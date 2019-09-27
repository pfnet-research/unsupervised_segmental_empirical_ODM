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

def tolerance_precision(bounds, seg_bnds, tolerance_window):
    #Precision
    hit = 0.0
    for bound in seg_bnds:
        for l in range(tolerance_window + 1):
            if (bound + l in bounds) and (bound + l > 0):
                hit += 1
                break
            elif (bound - l in bounds) and (bound - l > 0):
                hit += 1
                break
    return (hit / (len(seg_bnds)))


def tolerance_recall(bounds, seg_bnds, tolerance_window):
    #Recall
    hit = 0.0
    for bound in bounds:
        for l in range(tolerance_window + 1):
            if (bound + l in seg_bnds) and (bound + l > 0):
                hit += 1
                break
            elif (bound - l in seg_bnds) and (bound - l > 0):
                hit += 1
                break

    return (hit / (len(bounds)))



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


class RES_B(chainer.Chain):
    def __init__(self, feats, dims, dropout=0.3,
                 kernel=3,
                 negatives_samples=3,
                 min_diff_frame=2,
                 boundary_threshold=0.007,
                 tolerance_window=1):

        super(RES_B, self).__init__()
        self.dropout = dropout
        self.kernel = kernel
        self.n_negatives = negatives_samples
        self.min_diff_frame = min_diff_frame
        self.boundary_threshold = boundary_threshold
        self.tolerance_window = tolerance_window 

        with self.init_scope():
            self.in_conv = ConvLayer(feats, dims, kernel, 1, dropout=0.0)
            self.res_cn1 = ConvLayer(dims, dims, 1, 1, dropout=self.dropout)
            self.res_cn2 = ConvLayer(dims, dims, 1, 1, dropout=self.dropout)

            self.res_cn3 = ConvLayer(dims, dims, 1, 1, dropout=self.dropout)
            self.res_cn4 = ConvLayer(dims, dims, 1, 1, dropout=self.dropout)
            self.out_cnv = ConvLayer(dims, feats, 1, 1, dropout=0.0)
        self.acc = None
        self.feats = feats
        self.loss = None

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
        y_t = xs[:, :, self.kernel:]
        negatives = self.sample_negatives(y_t)
        y = F.expand_dims(y_t, axis=0)
        targets = F.concat([y, negatives], axis=0)
        n_targets = targets.shape[0]
        predictions = F.repeat(F.expand_dims(h, axis=0), n_targets, axis=0)
        labels = xp.zeros([n_targets, batchsize, h.shape[-1]], dtype=xp.float32)
        labels[0] = 1
        for i in range(batchsize):
            labels[:, i, ilens[i] - self.kernel:] = -1
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
        get_boundaries = True
        # xp = self.xp
        # 1. encoder
        # Generation Loss
        loss, latent_z, boundaries = self.forward_gen(xs, ilens, get_boundaries=get_boundaries)
        if ys is not None:
            batchsize = len(xs)
            precision = np.zeros((batchsize))
            recall = np.zeros((batchsize))
            for i in range(batchsize):
                precision[i] = tolerance_precision(ys[i], boundaries[i], self.tolerance_window)
                recall[i] = tolerance_recall(ys[i], boundaries[i], self.tolerance_window)
            precision = np.mean(precision)
            recall = np.mean(recall)
            return loss, precision, recall
        return loss, latent_z, boundaries

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
            idx[1:] += int(self.kernel)
            boundaries.append(idx.tolist())
        return boundaries

    def get_boundary(self, xs, weight, b_type):
        logging.info(f'Forwarding data with length: {xs.shape[0]}')
        xp = self.xp
        if weight == 1.0:
            logging.info('Pure GAS')
        elif weight == 0.0:
            logging.info('Pure RPM')
        else:
            logging.info('RPM + GAS')

        if b_type == 'up':
            layer_idx = [0, 1]
        elif b_type == 'all':
            layer_idx = [0, 1, 2, 3]
        else:
            raise Exception('not implemented')

        do_gas, do_rpm = False, False
        if weight > 0.0:
            do_gas = True
        if weight < 1.0:
            do_rpm = True

        residuals = [None] * 4

        # Forward
        logging.info(xs.shape)
        xs_x = xp.pad(xs[None, :-1], [[0, 0], [self.kernel - 1, 0], [0, 0]], mode='edge')
        xs_y = xs[None, 1:]
        h = self.in_conv(xs_x.transpose(0, 2, 1))
        for i in range(4):
            residuals[i] = self[f'res_cn{i + 1}'](h)
            h += residuals[i]
        h = self.out_cnv(h)

        # if do_rpm:
        #     rpm = F.stack(residuals[layer_idx], axis=0).data
        #     logging.info(rpm.shape)
        #     exit(1)

        #     drpm = rpm[1:] - rpm[:-1]
        #     drpm = np.pad(drpm, [2, 0], mode='edge')
        # else:
        #     drpm = 0.0

        if do_gas:
            gas = xp.mean((F.concat([residuals[x] for x in layer_idx], axis=1).data), axis=1)[0]
            dgas = gas[1:] - gas[:-1]
            dgas = np.pad(dgas, [2, 0], mode='edge')
            assert dgas.shape[0] == xs.shape[0]
        else:
            dgas = 0.
        exit(1)
        I_t = (1 - weight) * drpm + weight * dgas
        return None, I_t
