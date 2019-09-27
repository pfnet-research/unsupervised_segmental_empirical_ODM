#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=chainer
stage=0       # start from -1 if you need to start from data download
stop_stage=100
ngpu=0         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false
score_wer=false
preprocess_config=
###
train_config=conf/tuning/train_unsupervised.v1.yaml
decode_config=conf/tuning/decode_unsupervised.v1.yaml
###
lm_chr_config=conf/lm_char.yaml
lm_phn_config=conf/lm_phon.yaml
###
train_boundary_config=conf/train_boundary_rnn.yaml
decode_boundary_config=conf/decode_boundary.yaml

# rnnlm related
do_charlm=false
paired_chr=false
lm_chr_resume=        # specify a snapshot file to resume LM training
lm_chr_tag=            # tag for managing LMs
paired_phn=false
lm_phn_resume=
lm_phn_tag=

# decoding parameter
recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
tag="" # tag for managing experiments.
n_utters=5  # in k
njobs=4

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_trim_${n_utters}k
train_dev=dev_trim
recog_set="dev test"


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data Download"
    local/download_data.sh
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    BEEP_URL="http://www.openslr.org/resources/14/beep.tar.gz"
    echo "stage 0: Data preparation"
    local/wsj_prepare_beep_dict.sh --BEEP_URL ${BEEP_URL} --TEDLIUM_DIR db
    local/prepare_data.sh
    for dset in dev test train; do
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
    done
    utils/subset_data_dir.sh data/train $((${n_utters} * 1000)) data/train_5k
fi

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in test dev train_5k; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
        utils/fix_data_dir.sh data/${x}
    done

    # remove utt having > 2000 frames or < 10 frames or
    # remove utt having > 400 characters or 0 characters
    remove_longshortdata.sh --maxchars 400 data/train_${n_utters}k data/${train_set}
    rm -rf data/train_${n_utters}k
    remove_longshortdata.sh --maxchars 400 data/dev data/${train_dev}

    utils/fix_data_dir.sh data/${train_set}

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    # dump features for training
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

phonedir=data/local/phone_dict
phone_dict=data/lang_phone/phone_units.txt
phone_nlsyms=data/lang_phone/non_lang_phone.txt
dict=data/lang_1char/train_units.txt
nlsyms=data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    echo " make a non-lingustic phoneme list"

    mkdir -p data/lang_phone/    
    echo "make a non-linguistic phoneme list"
    cat ${phonedir}/silence_phones.txt > ${phone_nlsyms}
    cat ${phone_nlsyms}

    echo "make a phoneme dictionary"
    cat ${phonedir}/silence_phones.txt | awk '{print tolower($0) " " NR+0}' > ${phone_dict}
    lines=$(wc -l ${phone_dict} | cut -f1 -d' ')
    cat ${phonedir}/nonsilence_phones.txt | awk '{print tolower($0) " " NR+'"${lines}"'}' >> ${phone_dict}
    wc -l ${phone_dict}

    # Add phoneme to dirs
    for rtask in train_trim_${n_utters}k dev dev_trim test; do
        local/text2phn.py -n 1 -s 1 -l ${phone_nlsyms} data/${rtask}/text --add-header \
            --lexicon ${phonedir}/lexicon.txt > data/${rtask}/phn
    done 

    mkdir -p data/lang_1char/
    cut -f1 -d$'\t' ${phonedir}/lexicon.txt > data/lang_phone/input_org.txt
    cat data/lang_phone/input_org.txt | tr " " "\n" | sort | uniq | grep "<" | grep ">" > ${nlsyms}
    # cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    # make json labels
    ./local/data2json.sh --feat ${feat_tr_dir}/feats.scp  --phndic ${phone_dict} \
         data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    ./local/data2json.sh --feat ${feat_dt_dir}/feats.scp --phndic ${phone_dict} \
         data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        ./local/data2json.sh --feat ${feat_recog_dir}/feats.scp --phndic ${phone_dict} \
            data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lm_chr_tag} ]; then
    lm_chr_tag=$(basename ${lm_chr_config%.*})
fi
lm_chr_expname=train_lm_${backend}_${lm_chr_tag}
if ${paired_chr}; then
    lm_chr_expname=${lm_chr_expname}_paired
else
    lm_chr_expname=${lm_chr_expname}_unpaired
fi
lm_chr_expdir=exp/${lm_chr_expname}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Character-LM Preparation"
    if ${do_charlm}; then
        mkdir -p ${lm_chr_expdir}
        if ${paired_chr}; then
            lmdatadir=data/local/lm_train_paired
            [ ! -e ${lmdatadir} ] && mkdir -p ${lmdatadir}
            gunzip -c db/TEDLIUM_release2/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py |\
            spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
            cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
            > ${lmdatadir}/valid.txt

        else
            lmdatadir=data/local/lm_train_unpaired
            [ ! -e ${lmdatadir} ] && mkdir -p ${lmdatadir}
            echo "The character language model will be trained with a different dataset (librispeech)"
            text2token.py -s 1 -n 1 -l ${nlsyms} txt_files/train_text \
                | cut -f 2- -d" " > ${lmdatadir}/train.txt
            text2token.py -s 1 -n 1 -l ${nlsyms} txt_files/dev_text \
                | cut -f 2- -d" " > ${lmdatadir}/valid.txt

        ${cuda_cmd} --gpu ${ngpu} ${lm_chr_expdir}/train.log \
            lm_train.py \
            --config ${lm_chr_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --verbose 1 \
            --outdir ${lm_chr_expdir} \
            --tensorboard-dir tensorboard/${lm_chr_expname} \
            --train-label ${lmdatadir}/train.txt \
            --valid-label ${lmdatadir}/valid.txt \
            --resume ${lm_chr_resume} \
            --dict ${dict}
        fi
    fi
fi

if [ -z ${lm_phn_tag} ]; then
    lm_phn_tag=$(basename ${lm_phn_config%.*})
fi
lm_phn_expname=train_lm_${backend}_${lm_phn_tag}
if ${paired_phn}; then
    lm_phn_expname=${lm_phn_expname}_paired
else
    lm_phn_expname=${lm_phn_expname}_unpaired
fi
lm_phn_expdir=exp/${lm_phn_expname}

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Phoneme-LM Preparation"
    if ${paired_phn}; then
        mkdir -p ${lm_phn_expdir}
        lmdatadir=data/local/lm_train
        [ ! -e ${lmdatadir} ] && mkdir -p ${lmdatadir}
        gunzip -c db/TEDLIUM_release2/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py |\
        spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
        cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
        > ${lmdatadir}/valid.txt
    else
        echo "The phonetic language model will be obtained from a different dataset (librispeech)"
        local/text2phn.py -n 1 -s 1 -l ${nlsyms} txt_files/train_text --lexicon ${phonedir}/lexicon.txt > data/local/train_phn.txt
        local/text2phn.py -n 1 -s 1 -l ${nlsyms} txt_files/dev_text --lexicon ${phonedir}/lexicon.txt > data/local/dev_phn.txt
    fi
    ${cuda_cmd} --gpu ${ngpu} ${lm_phn_expdir}/train.log \
        lm_train.py \
        --config ${lm_phn_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lm_phn_expdir} \
        --unk-val 'spn' \
        --tensorboard-dir tensorboard/${lm_phn_expname} \
        --train-label data/local/train_phn.txt \
        --valid-label data/local/dev_phn.txt \
        --resume ${lm_phn_resume} \
        --dict ${phone_dict}
    
fi

# exp for training boundary
if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_boundary_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then 
	expname=${expname}_$(basename ${preprocess_config%.*}) 
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Network Training - Boundary"
    mkdir -p ${expdir}
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --boundary true \
        --ngpu ${ngpu} \
        --config ${train_boundary_config} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "stage 6: Obtaining Initial Boundaries for training and dev sets"
    nj=${njobs}
    pids=() # initialize pids
    decode_ngpu=0
    for rtask in ${train_set} ${train_dev}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_boundary_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --boundary true \
            --config ${decode_boundary_config} \
            --ngpu ${decode_ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${feat_recog_dir}/init_boundaries \
            --model ${expdir}/results/${recog_model}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then 
	expname=${expname}_$(basename ${preprocess_config%.*}) 
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Training Acoustic Model in Unsupervised fashion"
    mkdir -p ${expdir}
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --unsupervised true \
        --ngpu ${ngpu} \
        --preprocess-conf ${preprocess_config} \
        --config ${train_config} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${phone_dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    echo "stage 8: Decoding phonemes"
    nj=${njobs}
    pids=() # initialize pids
    decode_ngpu=0
    for rtask in ${recog_set} ${train_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --unsupervised true \
            --config ${decode_config} \
            --ngpu ${decode_ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
        
        score_sclite.sh --wer false ${expdir}/${decode_dir} ${phone_dict}
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi