#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from distutils.util import strtobool
import logging
import math

import chainer
from chainer import reporter
import numpy as np

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist

import espnet.nets.chainer_backend.unsup_cnn.basics as B

from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import get_model_conf
import espnet.lm.chainer_backend.lm as lm_chainer

from chainer import links as L
from chainer import functions as F

CTC_LOSS_THRESHOLD = 10000
MIN_VALUE = float(np.finfo(np.float32).min)


def detect_ups_downs(y):
    s0 = np.flatnonzero(y[1:] > y[:-1])+1
    s1 = np.flatnonzero(y[1:] < y[:-1])+1

    idx0 = np.searchsorted(s1,s0,'right')
    s0c = s0[np.r_[True,idx0[1:] > idx0[:-1]]]

    idx1 = np.searchsorted(s0c,s1,'right')
    s1c = s1[np.r_[True,idx1[1:] > idx1[:-1]]]
    s1c = np.concatenate(([0], s1c))
    j = 0
    s0c = [0]
    # Rearrange outputs for minimum distance (5 frames)
    for i in range(1, len(s1c)):
        if s1c[i] - s1c[j] > 5:
            s0c.append(s1c[i])
            j = i
    return s0c


class Conv2D(chainer.Chain):
    def __init__(self, n_in, n_out, ksize, stride, dropout, skip_connection, residual_scale=1.0, no_bias=True):
        super(Conv2D, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution1D(n_in, n_out, ksize, stride, nobias=True)
            self.norm = L.GroupNormalization(1, n_out)
        self.dropout = dropout
        self.sconn = skip_connection
        self.rscale = residual_scale
    
    def forward(self, x):
        x_dim = x.shape
        res_x = F.relu(self.norm(F.dropout(self.conv(x), self.dropout)))
        r_dim = res_x.shape
        if self.sconn and r_dim[1] == x_dim[1]:
            r_tsz = r_dim[2]
            tsz = x_dim[2]
            x = x[..., :: tsz // r_tsz][..., :r_tsz]
            res_x = (x + res_x) * self.rscale
        return res_x

class Wav2VecPredictor(chainer.Chain):
    def __init__(self, in_dim, out_dim,
                 prediction_steps=1,
                 n_negatives=1,
                 dropout=0.1,
                 offset=1):
        super(Wav2VecPredictor, self).__init__()
        with self.init_scope():
            self.project_to_steps = L.Deconvolution2D(in_dim, out_dim, (1, prediction_steps))
        self.dropout = dropout
        self.n_negatives = n_negatives
        self.offset = offset

    def sample_negatives(self, y):
        return

    def forward(self, x, y):
        negatives = self.sample_negatives(y)
        return x


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
        # RNN
        group.add_argument("--rnnlm-type", type=str, default="rnn",
                           help='')
        group.add_argument("--rnnlm-config", type=str, default=None,
                           help='')
        group.add_argument("--rnnlm-model", type=str, default=None,
                           help='')
        return parser

    def __init__(self, idim, odim, args, flag_return=True):
        chainer.Chain.__init__(self)
        in_channel = 1
        rnnlm_args = get_model_conf(args.rnnlm_model, args.rnnlm_config)
        rnnlm = lm_chainer.ClassifierWithState(lm_chainer.RNNLM(
            len(rnnlm_args.char_list_dict), rnnlm_args.layer, rnnlm_args.unit))
        chainer_load(args.rnnlm_model, rnnlm)

        _odim = len(rnnlm_args.char_list_dict)
        feat_enc = [[512, 3, x] for x in [2, 2, 1, 1, 1]]
        rin, jin = 0, 0
        for _, k, stride in feat_enc:
            if rin == 0:
                rin = k
            rin = rin + (k - 1) * jin
            if jin == 0:
                jin = stride
            else:
                jin *= stride
        offset = int(np.ceil(rin / jin))
        self.dropout = args.dropout_rate
        with self.init_scope():
            # MFCC to Vec Enc
            _idim = idim
            for n_out, k, stride in feat_enc:
                setattr(self, f'enc_conv{i}', Conv2D(_idim, n_out, k, stride, self.dropout, False))
                _idim = 512
            
            for i in range(9):
                setattr(self, f'agg_conv{i}', Conv2D(512, 512, 3, 1, self.dropout, False, no_bias=False))

            self.wav2vec_pred = Wav2VecPredictor(512, _odim, 
                                                 prediction_steps=2,
                                                 n_negatives=10,
                                                 dropout=self.dropout,
                                                 offset=offset)            
            self.rnnlm = rnnlm
        self.idim = idim
        self.in_channel = in_channel
        
        self.context_length = args.context_length
        self.odim = _odim
        self.lamda = args.trade_off
        self.started = False
    
    def forward_enc(self, xs):
        for i in range(5):
            xs = self[f'enc_conv{i}'](xs)
        return xs
    
    def forward_agg(self, xs):
        for i in range(9):
            xs = self[f'agg_conv{i}'](xs)
        return xs

    def forward(self, xs, ilens, ys, calculate_attentions=False):
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
        
        xs = xp.array(F.pad_sequence(xs).data.transpose(0, 2, 1))
        logging.info(xs.shape)
        feat_xs = self.forward_enc(xs)

        xs = self.forward_agg(F.dropout(feat_xs, self.dropout))
        
        # Set according ilens
        ilens = np.ceil((np.ceil(ilens.astype(np.float) / 2) - 1) / 2 - 1) - (2 * 12)
        ilens = ilens.astype(np.int)
        logging.info(xs.shape)
        logging.info(ilens)

        


        # self.started = True

        # ####################
        # Inter segment loss:
        # ####################
        x_len = xs.shape[1]
        amax_len = np.amax([len(x) for x in init_boundaries]) - 1
        mask = np.zeros((batchsize, amax_len, x_len, self.odim), dtype=np.bool)
        weight = xp.ones((batchsize, amax_len, 1), dtype=np.int32)
        # mask_xs = np.zeros((len(ilens), np.amax([len(x) for x in init_boundaries]) - 1, self.odim), dtype=np.bool)
        for i in range(batchsize):
            for j in range(len(init_boundaries[i]) - 1):
                mask[i, j, init_boundaries[i][j]: init_boundaries[i][j + 1]] = True
                weight[i, j] = init_boundaries[i][j + 1] - init_boundaries[i][j]
            # mask_xs[i, :len(init_boundaries[i])] = True
        xs = F.expand_dims(xs, axis=1)
        pxs = F.broadcast_to(xs, mask.shape)
        xs = F.where(mask, pxs, xp.zeros(mask.shape, dtype=xp.float32))
        xs = F.sum(xs, axis=2) / weight
        # xs = F.where(mask_xs, xs, xp.full(mask_xs.shape, MIN_VALUE, 'f'))
        
        xs = F.log_softmax(xs, axis=2)
        logging.info(xs.shape)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            state = None
            plm = list()
            for t in range(amax_len):
                state, _plm = self.rnnlm.predict(state, F.argmax(xs[:, t], axis=1))
                plm.append(_plm)
            plm = F.stack(plm, axis=1)
        loss_inter = - F.sum(plm * xs) / batchsize

        # ###################        
        # Intra segment loss:
        # ###################
        mask_min = np.zeros((batchsize, amax_len, x_len, self.odim), dtype=np.bool)
        mask_max = np.zeros((batchsize, amax_len, x_len, self.odim), dtype=np.bool)
        for i in range(batchsize):
            for j in range(len(init_boundaries[i]) - 1):
                mask_min[i, j, init_boundaries[i][j]: init_boundaries[i][j + 1] - 1] = True
                mask_max[i, j, init_boundaries[i][j] + 1: init_boundaries[i][j + 1]] = True
        x_t = F.where(mask_min, pxs, xp.zeros(mask.shape, dtype=xp.float32))[:, :, :-1]
        x_t_1 = F.where(mask_max, pxs, xp.zeros(mask.shape, dtype=xp.float32))[:, :, 1:]
        logging.debug(x_t_1.shape)

        # This loss is multiply by the length because some frames are zeros
        loss_intra = F.mean_squared_error(x_t_1, x_t) * x_len

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
        pass

    def calculate_all_attentions(self, xs, ilens, ys):
        """E2E attention calculation.

        Args:
            xs (List): List of padded input sequences. [(T1, idim), (T2, idim), ...]
            ilens (np.ndarray): Batch of lengths of input sequences. (B)
            ys (List): List of character id sequence tensor. [(L1), (L2), (L3), ...]

        Returns:
            float np.ndarray: Attention weights. (B, Lmax, Tmax)

        """
        hs, ilens = self.enc(xs, ilens)
        att_ws = self.dec.calculate_all_attentions(hs, ys)

        return att_ws

    @staticmethod
    def CustomConverter(labels=False):
        from espnet.nets.chainer_backend.unsupervised_model.training import CustomConverter
        return CustomConverter(labels)
    
    @staticmethod
    def CustomUpdater(iterator, optimizer, converter, device='-1', accum_grad=1):
        from espnet.nets.chainer_backend.unsupervised_model.training import CustomUpdater
        return CustomUpdater(iterator, optimizer, converter=converter, device=device, accum_grad=accum_grad)
