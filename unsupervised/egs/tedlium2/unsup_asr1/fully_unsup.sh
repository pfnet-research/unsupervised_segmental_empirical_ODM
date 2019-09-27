#!/bin/bash

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

preprocess_config=
train_config=conf/train_unsup_1.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml
njobs=4

# rnnlm related
lm_resume=        # specify a snapshot file to resume LM training
lmtag=            # tag for managing LMs

# decoding parameter
recog_model=model.loss.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

# exp tag
tag="" # tag for managing experiments.
n_utters=5  # in k

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_trim_${n_utters}k
train_dev=dev_trim
recog_set="dev test"

feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le -1 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "Extraction features should be obtained from main_run.sh"
    exit 0
fi

dict=data/lang_1char/train_units.txt
echo "dictionary: ${dict}"

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
mkdir -p ${expdir}

# It takes a few days. If you just want to end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=exp/${lmexpname}
mkdir -p ${lmexpdir}

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "skip 3"
    exit 0
    echo "stage 3: LM Preparation"
    lmdatadir=data/local/lm_train_${bpemode}${nbpe}
    [ ! -e ${lmdatadir} ] && mkdir -p ${lmdatadir}
    gunzip -c db/TEDLIUM_release2/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py |\
	spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece \
	> ${lmdatadir}/valid.txt
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --fully-unsupervised true \
        --ngpu ${ngpu} \
        --preprocess-conf ${preprocess_config} \
        --config ${train_config} \
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

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
#     echo "stage 5: Get boundaries"
#     nj=${njobs}

#     pids=() # initialize pids
#     for rtask in ${train_set} ${train_dev}; do
#     (
#         decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
#         feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

#         # split data
#         splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

#         #### use CPU for decoding
#         ngpu=0
#         ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
#             asr_recog.py \
#             --fully-unsup-boundaries true \
#             --config ${decode_config} \
#             --ngpu ${ngpu} \
#             --backend ${backend} \
#             --debugmode ${debugmode} \
#             --verbose ${verbose} \
#             --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
#             --result-label ${feat_recog_dir}/cnn_boundaries  \
#             --model ${expdir}/results/${recog_model}
        
#         ./local/score_boundaries.py ${feat_recog_dir}/init_boundaries ${feat_recog_dir}/cnn_boundaries
#     ) &
#     pids+=($!) # store background pids
#     done
#     i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
#     [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# fi

phone_dict=data/lang_phone/phone_units.txt
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=4

    pids=() # initialize pids
    for rtask in ${recog_set} ${train_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --fully-unsupervised true \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --debugmode ${debugmode} \
            --verbose ${verbose} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} 

        score_sclite.sh --wer false ${expdir}/${decode_dir} ${phone_dict}
        ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
