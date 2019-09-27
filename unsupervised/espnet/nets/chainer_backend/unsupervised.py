#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from distutils.util import strtobool
import logging
import math

import chainer
from chainer import initializers
from chainer import reporter
import numpy as np

from espnet.nets.asr_interface import ASRInterface

from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import get_model_conf
from espnet.nets.lm_interface import dynamic_import_lm

from chainer import links as L
from chainer import functions as F

CTC_LOSS_THRESHOLD = 10000
MIN_VALUE = float(np.finfo(np.float32).min)

from chainer import cuda

def beam_search(px, ilens, beam, xp, pb, dims, predictor):
    pb = xp.array(np.log(pb + 1e-20))
    batchsize = px.shape[0]

    best_states = None
    max_len = px.shape[1] - 1
    best_scores = xp.zeros((batchsize, beam))
    best_pred = xp.zeros((batchsize, max_len, beam))
    
    batch_mask = xp.array([np.full((beam), i) for i in range(batchsize)])
    for j in range(max_len):
        xs = px[:, j]
        pbt = pb[:, j]
        local_pred_idx = xp.argsort(xs, axis=1)[..., ::-1][..., :beam]
        local_pred_scores = xs[batch_mask, local_pred_idx]
        
        local_state, plm = predictor(best_states, local_pred_idx.reshape(-1))  # local_state & plm in shap: bs * beam x dims
        plm = xp.log(plm.reshape(batchsize, beam, -1) + 1e-20)
        local_score_0 = pbt[:, None] + local_pred_scores 
        local_score_1 = pbt.reshape(-1, 1, 1) + plm + local_pred_scores.reshape(batchsize, beam, 1)
        local_score = xp.concatenate([local_score_0[..., None], local_score_1], axis=-1).reshape(batchsize, -1)

        local_best_idx = xp.argsort(local_score, axis=-1)[..., ::-1][..., :beam]
        local_best_score = local_score[batch_mask, local_best_idx]
        local_idx = local_best_idx // dims
        local_best_pred = local_pred_idx[batch_mask, local_idx]
        next_idx = xp.fmod(local_best_idx, xp.full(local_best_idx.shape, dims))

        # Select best states
        new_states = dict()
        for key in local_state:
            new_values = list()
            for values in local_state[key]:
                values = values.reshape(batchsize, beam, -1).data
                dim_state = values.shape[-1]
                next_hidden = list()
                for s1 in range(batchsize):
                    for s2 in range(next_idx.shape[1]):
                        if local_idx[s1, s2] > 0:
                            next_hidden.append(values[s1, local_idx[s1, s2] - 1, :])
                        else:
                            next_hidden.append(xp.zeros((dim_state)))
                new_values.append(chainer.Variable(xp.stack(next_hidden, axis=0)))
            new_states[key] = new_values
        #if xp is not np:
        #    best_states = new_states  # Disable for testing code in CPU
        best_scores += local_best_score
        best_pred[:, j] = local_best_pred

    if xp is not np:
        best_scores = xp.asnumpy(best_scores)
        best_pred = xp.asnumpy(best_pred)
    return best_scores, best_pred


class AcousticModel(ASRInterface, chainer.Chain):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("unsupervised model setting")
        group.add_argument("--encoder-type", type=str, default="conv2d",
                           help='transformer input layer type')

        group.add_argument('--dropout-rate', default=0.3, type=float,
                           help='Dropout rate for the encoder')
        group.add_argument('--context-length', default=11, type=int,
                           help='')
        group.add_argument('--trade-off', default=1e-5, type=float,
                           help='')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        
        # PhoneLM
        group.add_argument("--phnlm-type", type=str, default="rnn",
                           help='')
        group.add_argument("--phnlm-config", type=str, default=None,
                           help='')
        group.add_argument("--phnlm-model", type=str, default=None,
                           help='')
        group.add_argument("--phnlm-model-module", type=str, default='default',
                           help='')
        # Boundary
        group.add_argument("--boundary-map-predict", type=strtobool, default=False,
                           help='')
        # Optimizer
        group.add_argument("--noam-dims", type=int, default=4096,
                           help='')
        group.add_argument("--noam-lr", type=float, default=1.,
                           help='')
        group.add_argument("--noam-warmup-steps", type=int, default=1000,
                           help='')

        # Model related
        group.add_argument("--beam-search", type=int, default=3,
                           help='')
        group.add_argument("--min-diff-frame", type=int, default=2,
                           help='')
        group.add_argument("--topk", type=int, default=3,
                           help='')
        return parser

    def __init__(self, idim, odim, args, flag_return=False, load_lm=True):
        chainer.Chain.__init__(self)
        in_channel = 1
        # Main Initialization
        odim =  len(args.char_list)
        stdv = 1. / np.sqrt(4096)
        with self.init_scope():
            self.pyx_1 = L.Linear(idim * args.context_length, 4096, initial_bias=initializers.Uniform(scale=stdv))
            #self.norm_1 = L.LayerNormalization(4096)
            self.pyx_2 = L.Linear(4096, 4096, initial_bias=initializers.Uniform(scale=stdv))
            # self.norm_2 = L.LayerNormalization(4096)
            self.pyx_out = L.Linear(4096, odim, initial_bias=initializers.Uniform(scale=stdv))
        if load_lm:
            self.load_lm(args)
        
        self.idim = idim
        self.in_channel = in_channel
        self.dropout = args.dropout_rate
        self.context_length = args.context_length
        self.odim = odim
        self.lamda = args.trade_off
        self.start = False
        self.beam = args.beam_search
        self.min_diff_frame = args.min_diff_frame
        self.topk = args.topk
        self.eos = self.odim - 1
        self.sos = self.odim - 1

    def load_lm(self, args):
        # Load RNNLM
        rnnlm_args = get_model_conf(args.phnlm_model, args.phnlm_config)
        if args.phnlm_model_module == 'default':
            import espnet.nets.chainer_backend.lm.default as lm_chainer
            rnnlm = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(
                len(rnnlm_args.char_list_dict), rnnlm_args.layer, rnnlm_args.unit))  # noqa pylint: disable=no-member
        else:
            lm_chainer = dynamic_import_lm(args.phnlm_model_module, args.backend)
            raise Exception
        chainer_load(args.phnlm_model, rnnlm)
        # _odim = len(rnnlm_args.char_list_dict)
        self.rnnlm = rnnlm

    def set_rnn_device(self, device):
        if not self.start:
            self.rnnlm.to_device(device)    
            self.start = True

    def prepare_context(self, xs, ilens):
        new_xs = list()
        for i in range(len(ilens)):
            x = xs[i]
            pad_left = x[0][None, :].repeat(self.context_length // 2, axis=0)
            pad_right = x[ilens[i] - 1][None, :].repeat(self.context_length // 2, axis=0)
            x = np.concatenate([pad_left, x, pad_right])
            indices = [np.arange(x, x + self.context_length) for x in range(ilens[i])]
            x = x.take(indices, axis=0)
            new_xs.append(x)
        xs = F.pad_sequence(new_xs).data
        return self.xp.array(xs)
        
    def forward(self, xs, ilens, ys, _boundaries, probs=None):
        """VGG2L forward propagation.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each features. (B,)

        Returns:
            chainer.Variable: Subsampled vector of xs.
            chainer.Variable: Subsampled vector of ilens.

        """
        batchsize = len(ilens)

        xp = self.xp
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        xs = self.prepare_context(xs, ilens)
        device = cuda.get_device_from_array(xs)  # noqa pylint: disable=no-member
        if device.id > -1:
            self.set_rnn_device(device)

        # x: utt x frames x context x dim
        # ####################
        # Forward p(y|x):
        # ####################
        xs = self.forward_px(xs, ilens)
        if _boundaries is None:
            # Optimize argmax_y p(y|x) = argmax_y p(y, x) in form: min(-log p(y|x))
            x = F.log_softmax(xs, axis=-1).data
            boundaries = self.predict_boundaries(x, ilens, probs)
        else:
            boundaries = list()
            for i in range(batchsize):
                idx = np.array(_boundaries[i])
                min_len = (idx[1:] - idx[:-1]) > self.min_diff_frame
                idx = idx[np.pad(min_len, [1, 0], mode='constant', constant_values=True)].tolist()
                boundaries.append(idx)
        xs = F.softmax(xs, axis=-1)

        # ####################
        # Inter segment loss:
        # ####################
        pxs = F.pad(xs, [[0, 0], [0, 1], [0, 0]], mode='constant', constant_values=1.)
        amax_len = np.amax([len(x) for x in boundaries]) - 1
        batch_segments  = list()
        batch_idx = list()
        for i in range(batchsize):
            _boundary = boundaries[i]
            smt_idx = xp.array(F.pad_sequence([np.arange(_boundary[j], _boundary[j + 1]) for j in range(len(_boundary) - 1)], padding=-1).data)
            idx = xp.array([_boundary[j] + (_boundary[j+1] - _boundary[j]) // 2 for j in range(len(_boundary) - 1)])
            pyz = F.cumsum(F.prod(pxs[i, smt_idx, :], axis=1), axis=0)
            weights = xp.arange(1, pyz.shape[0] + 1).reshape(-1, 1).astype(xp.float32)
            batch_segments.append(F.log((pyz / weights) + 1e-30))
            batch_idx.append(F.argmax(pxs[i, idx, :], axis=1).data)
        xs = F.pad_sequence(batch_segments)
        idx_xs = F.pad_sequence(batch_idx, padding=self.eos)
        h = xp.full([batchsize, 1], self.sos)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            state = None
            plm = list()
            for t in range(amax_len):
                state, _plm = self.rnnlm.predict(state, h)
                plm.append(_plm)
                h = idx_xs[:, t].reshape(-1, 1)
            del state
            plm = xp.stack(plm, axis=1)
        loss_inter = - F.sum(plm * xs) / batchsize


        # ###################        
        # Intra segment loss:
        # ###################
        loss_intra = 0.0
        for i in range(batchsize):
            _boundary = boundaries[i]
            _min = xp.array(F.pad_sequence([np.arange(_boundary[j], _boundary[j + 1] -1) for j in range(len(_boundary) - 1)], padding=-1).data)
            _max = xp.array(F.pad_sequence([np.arange(_boundary[j] + 1, _boundary[j + 1]) for j in range(len(_boundary) - 1)], padding=-1).data)
            loss_intra += F.sum((pxs[i, _max, :] - pxs[i, _min, :]) ** 2)
        loss_intra = loss_intra / float(batchsize)  # Normalized along batchsize

        self.loss = loss_inter + self.lamda * loss_intra
        loss_data = self.loss.data
        if not math.isnan(loss_data):
            reporter.report({'loss_inter': loss_inter}, self)
            reporter.report({'loss_intra': loss_intra}, self)
            logging.info('intra loss:' + str(loss_intra.data))
            logging.info('inter loss:' + str(loss_inter.data))
            logging.info('mtl loss:' + str(loss_data))
            reporter.report({'loss': self.loss}, self)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss
    
    def forward_px(self, xs, ilens):
        # x: utt x frames x context x dim
        bs, fr, _, _ = xs.shape
        xs = F.relu(self.pyx_1(xs.reshape(bs * fr, -1)))
        xs = F.relu(self.pyx_2(F.dropout(xs, self.dropout)))
        xs = self.pyx_out(F.dropout(xs, self.dropout)).reshape(bs, fr, -1)
        xs = F.pad_sequence([xs[i, :ilens[i]] for i in range(ilens.shape[0])])
        return xs
    
    def predict_boundaries(self, x, ilens, probs):
        dims = x.shape[-1] + 1
        _, hyps = beam_search(x, ilens, self.beam, self.xp, probs, dims, self.rnnlm.predict)
        boundaries = list()
        for i in range(x.shape[0]):
            utt_hyps = hyps[i, :ilens[i] - 1]
            utt_hyps = np.pad(utt_hyps, [[1, 0], [0, 0]], mode='edge')
            new_boundaries = utt_hyps[1:] != utt_hyps[:-1]
            new_boundaries = np.pad(new_boundaries, [[1, 0], [0, 0]], mode='constant', constant_values=True)
            new_boundaries[-1] = True
            for j in range(self.beam):
                idx = np.argwhere(new_boundaries[:, j])[:, 0]  # select top
                ### last clean up idx
                min_len = (idx[1:] - idx[:-1]) > self.min_diff_frame
                idx = idx[np.pad(min_len, [1, 0], mode='constant', constant_values=True)].tolist()
                if len(idx) > 3:    
                    break
            boundaries.append(idx)
        return boundaries
        

    def recognize(self, x, recog_args, char_list, rnnlm=None):
        """E2E greedy/beam search.

        Args:
            x (chainer.Variable): Input tensor for recognition.
            recog_args (parser.args): Arguments of config file.
            char_list (List[str]): List of Charactors.
            rnnlm (Module): RNNLM module defined at `espnet.lm.chainer_backend.lm`.

        Returns:
            List[Dict[str, Any]]: Result of recognition.

        """
        # subsample frame
        from itertools import groupby
        xp = self.xp
        ilens = xp.array([x.shape[0]])
        xs = self.prepare_context(x[None], ilens)
        xs = self.forward_px(xs, ilens)
        logging.info(xs.shape)
        xs = F.softmax(xs, axis=-1).data[0]
        best_idx = xp.argsort(xs, axis=-1)[:, ::-1]
        logging.info(best_idx.shape)
        y_hat = [int(x[0]) for x in groupby(best_idx[:, 0])]
        hyps = [{'score': 1.0, 'yseq': y_hat, 'c_prev': [None], 'z_prev': [None], 'a_prev': None}]
        return hyps

    def calculate_all_attentions(self, xs, ilens, ys):
        """E2E attention calculation.

        Args:
            xs (List): List of padded input sequences. [(T1, idim), (T2, idim), ...]
            ilens (np.ndarray): Batch of lengths of input sequences. (B)
            ys (List): List of character id sequence tensor. [(L1), (L2), (L3), ...]

        Returns:
            float np.ndarray: Attention weights. (B, Lmax, Tmax)

        """
        pass

    @staticmethod
    def CustomConverter(labels=False, folder=None, do_map=False):
        from espnet.nets.chainer_backend.net_unsupervised.training import CustomConverter
        return CustomConverter(labels, folder, do_map)
    
    @staticmethod
    def CustomUpdater(iterator, optimizer, converter, device='-1'):
        from espnet.nets.chainer_backend.net_unsupervised.training import CustomUpdater
        return CustomUpdater(iterator, optimizer, converter=converter, device=device)

    @staticmethod
    def CustomParallelUpdater(iterators, optimizer, converter, devices):
        from espnet.nets.chainer_backend.net_unsupervised.training import CustomParallelUpdater
        return CustomParallelUpdater(iterators, optimizer, converter=converter, devices=devices)

