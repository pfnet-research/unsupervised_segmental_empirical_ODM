#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import logging
import math

import chainer
from chainer import cuda
from chainer import reporter
from chainer import initializers as Init
import numpy as np

from espnet.nets.asr_interface import ASRInterface

from chainer import functions as F
from chainer import links as L

from espnet.asr.asr_utils import chainer_load
from espnet.asr.asr_utils import get_model_conf
from espnet.nets.lm_interface import dynamic_import_lm


class ConvLayer(chainer.Chain):
    def __init__(self, in_channels, out_channels, ksize, stride,
                 dropout=0.1):
        super(ConvLayer, self).__init__()
        stdv = 1. / np.sqrt(out_channels)
        with self.init_scope():
            self.conv = L.Convolution1D(in_channels, out_channels, ksize, stride,
                                        nobias=True, initialW=Init.Uniform(stdv))
            self.bn = L.GroupNormalization(1, out_channels)
        self.dropout = dropout
    
    def forward(self, x):
        x = self.conv(F.dropout(x, self.dropout))
        return F.relu(self.bn(x))


class E2E(ASRInterface, chainer.Chain):
    """E2E module for chainer backend.

    Args:
        idim (int): Dimension of the inputs.
        odim (int): Dimension of the outputs.
        args (parser.args): Training config.
        flag_return (bool): If True, train() would return
            additional metrics in addition to the training
            loss.

    """
    @staticmethod
    def add_arguments(parser):
        E2E.encoder_add_arguments(parser)
        E2E.decoder_add_arguments(parser)
        return parser

    @staticmethod
    def encoder_add_arguments(parser):
        group = parser.add_argument_group("E2E encoder setting")
        # encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        group.add_argument('--ekernel', default=3, type=int,
                           help='')
        group.add_argument('--dkernel', default=11, type=int,
                           help='')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        group.add_argument('--negatives-samples', default=3, type=int,
                           help='')
        group.add_argument('--scheduler-steps', type=int, default=1,
                            help='')
        group.add_argument('--trade-off', default=0.5, type=float,
                           help='')
        group.add_argument('--min-phi', default=0.6, type=float,
                           help='')
        group.add_argument('--min-diff-frame', default=2, type=int,
                           help='')
        # optim related
        group.add_argument('--noam-dims', type=int, default=512,
                            help='')
        group.add_argument('--noam-steps', type=int, default=5000,
                            help='')
        group.add_argument('--noam-scale', type=float, default=1.0,
                            help='')
        group.add_argument('--boundary-threshold', type=float, default=0.009,
                            help='')
        # PhoneLM
        group.add_argument("--phnlm-type", type=str, default="rnn",
                           help='')
        group.add_argument("--phnlm-config", type=str, default=None,
                           help='')
        group.add_argument("--phnlm-model", type=str, default=None,
                           help='')
        group.add_argument("--phn-file", type=str, default=None,
                           help='')
        group.add_argument("--phnlm-model-module", type=str, default='default',
                           help='')
        group.add_argument("--topk", type=int, default=3,
                           help='')
        return parser

    @staticmethod
    def decoder_add_arguments(parser):
        group = parser.add_argument_group("E2E encoder setting")
        group.add_argument('--dtype', default='lstm', type=str,
                           choices=['lstm', 'gru'],
                           help='Type of decoder network architecture')
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--dropout-rate-decoder', default=0.0, type=float,
                           help='Dropout rate for the decoder')
        group.add_argument('--sampling-probability', default=0.0, type=float,
                           help='Ratio of predicted labels fed back to decoder')
        return parser

    def __init__(self, idim, odim, args, flag_return=True, load_lm=True):
        chainer.Chain.__init__(self)
        self.dropout = args.dropout_rate
        self.verbose = args.verbose
        self.outdir = args.outdir
        self.ekernel = args.ekernel
        self.dkernel = args.dkernel
        self.scheduler_steps = args.scheduler_steps
        self.n_negatives = args.negatives_samples
        self.lamda = args.trade_off
        self.min_phi = args.min_phi
        self.min_diff_frame = args.min_diff_frame
        self.boundary_threshold = args.boundary_threshold
        self.topk = args.topk
        
        with open(args.phn_file, 'rb') as f:
            dictionary = f.readlines()
            phn_list = [entry.decode('utf-8').split(' ')[0]
                        for entry in dictionary]
            phn_list.insert(0, '<blank>')
            phn_list.append('<eos>')
            args.phn_list = phn_list

        # below means the last number becomes eos/sos ID
        # note that sos/eos IDs are identical
        self.sos = odim - 1
        self.eos = odim - 1
        if load_lm:
            self.load_lm(args)
        with self.init_scope():
            # encoder
            self.in_conv = ConvLayer(idim, 512, self.ekernel, 1, dropout=0.0)
            self.res_cn1 = ConvLayer(512, 512, 1, 1, dropout=self.dropout)
            self.res_cn2 = ConvLayer(512, 512, 1, 1, dropout=self.dropout)

            self.res_cn3 = ConvLayer(512, 512, 1, 1, dropout=self.dropout)
            self.res_cn4 = ConvLayer(512, 512, 1, 1, dropout=self.dropout)
            self.out_cnv = ConvLayer(512, idim, 1, 1, dropout=0.0)

            # z to phone
            odim = len(args.phn_list)
            self.dec_proj_1 = L.Convolution1D(512, odim, self.dkernel, 1, nobias=True)
            self.dec_proj_2 = L.Convolution1D(odim, odim, self.dkernel, 1, nobias=True)
            self.norm_p1 = L.GroupNormalization(1, odim)
            self.norm_p2 = L.GroupNormalization(1, odim)

        self.acc = None
        self.idim = idim
        self.loss = None
        self.flag_return = flag_return
        self.t = 0
        self.start = False
        self.odim = odim

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

    def forward_gen(self, xs, ilens, get_boundaries=False):
        xp = self.xp
        _xs = list()
        amax = max(ilens)
        batchsize = len(ilens)
        for i in range(batchsize):
            padd = amax - ilens[i] + 1
            x = np.pad(xs[i], [[0, padd], [0, 0]], mode='edge')
            _xs.append(x)
        xs = xp.array(np.stack(_xs, axis=0).transpose(0, 2, 1))
        xs_x = xs[:, :, :-1]
        h = self.in_conv(xs_x)
        res_1 = self.res_cn1(h)
        h += res_1
        res_2 = self.res_cn2(h)
        latent_z = h + res_2
        res_3 = self.res_cn3(latent_z)
        h = latent_z + res_3
        res_4 = self.res_cn4(h)
        h = self.out_cnv(h + res_4)
        
        # negatives
        y_t = xs[:, :, self.ekernel:]
        negatives = self.sample_negatives(y_t)
        y = F.expand_dims(y_t, axis=0)
        targets = F.concat([y, negatives], axis=0)
        n_targets = targets.shape[0]
        predictions = F.repeat(F.expand_dims(h, axis=0), n_targets, axis=0)
        labels = xp.zeros([n_targets, batchsize, h.shape[-1]], dtype=xp.float32)
        labels[0] = 1
        for i in range(batchsize):
            labels[:, i, ilens[i] - self.ekernel:] = -1
        labels = F.flatten(labels)
        predictions = F.sum(targets.data * predictions, axis=2)
        predictions = F.flatten(predictions)
        loss = F.sigmoid_cross_entropy(predictions, labels.data)

        if get_boundaries:
            with chainer.no_backprop_mode():
                residuals = xp.mean(F.concat([res_1, res_2, res_3, res_4], axis=1).data, axis=1)
                cpe = F.sum(((y_t - h) ** 2).transpose(0, 2, 1), axis=-1).data
                assert cpe.shape == residuals.shape
                if xp is not np:
                    residuals = xp.asnumpy(residuals)
                    cpe = xp.asnumpy(cpe)
                boundaries = self.get_boundaries(residuals, cpe, threshold=self.boundary_threshold)
        else:
            boundaries = None
        return loss, latent_z, boundaries

    def forward(self, xs, ilens, ys=None):
        """E2E forward propagation.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each input batch. (B,)
            ys (chainer.Variable): Batch of padded target features. (B, Lmax, odim)

        Returns:
            float: Loss that calculated by attention and ctc loss.
            float (optional): Ctc loss.
            float (optional): Attention loss.
            float (optional): Accuracy.

        """
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))
        if chainer.config.train:
            self.t += 1
        phi = np.amin([np.exp((float(self.t) - self.scheduler_steps) / float(self.t)), self.min_phi])
        if phi > 0.01:
            get_boundaries = True
        else:
            get_boundaries = False
        # xp = self.xp
        # 1. encoder
        # Generation Loss
        loss_gen, latent_z, boundaries = self.forward_gen(xs, ilens, get_boundaries=get_boundaries)
        loss_gen_data = loss_gen.data

        # loss_gen = F.mean_squared_error(xs_t * mask, h * mask)
        if get_boundaries:
            loss_inter, loss_intra = self.odm(latent_z, boundaries)
            self.loss = (1. - phi) * loss_gen + phi * (loss_inter + self.lamda * loss_intra)
            loss_inter_data = loss_inter.data
            loss_intra_data = loss_intra.data
        else:
            self.loss = loss_gen
            loss_inter_data = None
            loss_intra_data = None

        if not math.isnan(self.loss.data):
            if loss_inter_data is not None:
                logging.info('Inter loss:' + str(loss_inter_data))
                logging.info('Intra loss:' + str(loss_intra_data))
            logging.info('mtl loss:' + str(self.loss.data))
            reporter.report({'loss': self.loss}, self)
            reporter.report({'loss_gen': loss_gen_data}, self)
            reporter.report({'loss_inter': loss_inter_data}, self)
            reporter.report({'loss_intra': loss_intra_data}, self)
        else:
            logging.warning('loss (=%f) is not correct', self.loss.data)
        return self.loss

    def sample_negatives(self, y):
        batch, dims, frames = y.shape
        # utt x dim x frame => dim x utt x frame
        # dim x (utt * frame)
        y = y.transpose(1, 0, 2).reshape(dims, -1)
        high = frames
        neg_idxs = np.random.randint(low=0, high=high, size=(batch, self.n_negatives * frames))
        for i in range(1, batch):
            neg_idxs[i] += i * high
        negs = y[..., neg_idxs.reshape(-1)]
        negs = negs.reshape(dims, batch, self.n_negatives, frames).transpose(2, 1, 0, 3)  # to Neg x utt x dim x frame
        return negs
    
    def odm(self, xs, boundaries):
        xp = self.xp
        # Check device
        device = cuda.get_device_from_array(xs.array)  # noqa pylint: disable=no-member
        if device.id > -1:
            self.set_rnn_device(device)

        batchsize = xs.shape[0]
        xs = F.pad(xs, [[0, 0], [0, 0], [self.dkernel // 2, self.dkernel // 2]], mode='edge')
        # ####################
        # Forward p(y|x)
        # ####################
        xs = F.relu(self.norm_p1(self.dec_proj_1(F.dropout(xs, self.dropout))))
        xs1 = F.pad(xs, [[0, 0], [0, 0], [self.dkernel // 2, self.dkernel // 2]], mode='edge')
        xs = F.relu(self.norm_p2(self.dec_proj_2(F.dropout(xs1, self.dropout)))) + xs
        xs = F.softmax(xs, axis=1).transpose(0, 2, 1)

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
        loss_inter = - F.sum(plm * xs) / float(batchsize * amax_len)

        # ###################        
        # Intra segment loss:
        # ###################
        loss_intra = 0.
        for i in range(batchsize):
            _boundary = boundaries[i]
            _min = xp.array(F.pad_sequence([np.arange(_boundary[j], _boundary[j + 1] -1) for j in range(len(_boundary) - 1)], padding=-1).data)
            _max = xp.array(F.pad_sequence([np.arange(_boundary[j] + 1, _boundary[j + 1]) for j in range(len(_boundary) - 1)], padding=-1).data)
            loss_intra += F.sum((pxs[i, _max, :] - pxs[i, _min, :]) ** 2) / float(_min.shape[0])
        return loss_inter, loss_intra

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
        xp = self.xp
        from itertools import groupby
        logging.info(x.shape)
        h = self.in_conv(x[None, :-1].transpose(0, 2, 1))
        res_1 = self.res_cn1(h)
        h += res_1
        res_2 = self.res_cn2(h)
        latent_z = h + res_2
        res_3 = self.res_cn1(latent_z)
        h = latent_z + res_3
        res_4 = self.res_cn1(h)
        h = self.out_cnv(h + res_4)

        # ## Get boundaries
        residuals = xp.mean(F.concat([res_1, res_2, res_3, res_4], axis=1).data, axis=1)
        cpe = F.sum(((x[None, self.ekernel:].transpose(0, 2, 1) - h) ** 2).transpose(0, 2, 1), axis=-1).data
        assert cpe.shape == residuals.shape
        if xp is not np:
            residuals = xp.asnumpy(residuals)
            cpe = xp.asnumpy(cpe)
        boundaries = self.get_boundaries(residuals, cpe, threshold=self.boundary_threshold)
        
        xs = F.pad(latent_z, [[0, 0], [0, 0], [self.dkernel // 2, self.dkernel // 2]], mode='edge')
        xs = F.relu(self.norm_p1(self.dec_proj_1(xs)))
        xs1 = F.pad(xs, [[0, 0], [0, 0], [self.dkernel // 2, self.dkernel // 2]], mode='edge')
        xs = F.relu(self.norm_p2(self.dec_proj_2(xs1))) + xs
        xs = F.softmax(xs, axis=1).transpose(0, 2, 1)

        pxs = F.pad(xs, [[0, 0], [0, 1], [0, 0]], mode='constant', constant_values=1.)
        idx = xp.array([boundaries[0][j] + (boundaries[0][j+1] - boundaries[0][j]) // 2 for j in range(len(boundaries[0]) - 1)])
        idx = F.argmax(pxs[0, idx, :], axis=1).data
        y_hat = [int(x[0]) for x in groupby(idx)]
        # logging.info(idx)
        hyps = [{'score': 1.0, 'yseq': y_hat, 'c_prev': [None], 'z_prev': [None], 'a_prev': None}]
        return hyps
        
    def get_boundaries(self, residual, cpe, threshold=0.007, weight=0.5):
        # GAS + RPE (Residual Predictive Error)
        dgas = residual[:, 1:] - residual[:, :-1]
        dcpe = cpe[:, 1:] - cpe[:, :-1]
        signal = (1 - weight) * dcpe + weight * dgas  # I_t
        local_maxima = (signal[:, 2:] < signal[:, 1:-1]) * (signal[:, 1:-1] > signal[:, :-2])
        local_maxima = np.pad(local_maxima, [[0, 0], [1, 1]], mode='constant', constant_values=False)
        boundaries = (local_maxima * (signal >= threshold))
        batchsize = boundaries.shape[0]
        _boundaries = np.pad(boundaries, [[0, 0], [1, 0]], mode='constant', constant_values=True)
        boundaries = list()
        for i in range(batchsize):
            idx = np.argwhere(_boundaries[i])[:, 0]
            ### last clean up idx
            min_len = (idx[1:] - idx[:-1]) > self.min_diff_frame
            idx = idx[np.pad(min_len, [1, 0], mode='constant', constant_values=True)]
            idx[1:] += int(self.ekernel)
            boundaries.append(idx.tolist())
        return boundaries

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
    def CustomConverter(labels=False):
        from espnet.nets.chainer_backend.fully_unsup1.training import CustomConverter
        return CustomConverter(labels=labels)

    @staticmethod
    def CustomUpdater(
            iterator, optimizer, converter, device='-1', accum_grad=1):
        from espnet.nets.chainer_backend.fully_unsup1.training import CustomUpdater
        return CustomUpdater(
            iterator, optimizer, converter=converter, device=device, accum_grad=accum_grad)

    @staticmethod
    def CustomParallelUpdater(iterators, optimizer, converter, devices):
        from espnet.nets.chainer_backend.fully_unsup1.training import CustomParallelUpdater
        return CustomParallelUpdater(iterators, optimizer, converter=converter, devices=devices)
