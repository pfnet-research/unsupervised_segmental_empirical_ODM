
import logging
import math

import chainer
import numpy as np


from chainer import links as L
from chainer import functions as F


class Wav2VecPredictor(chainer.Chain):
    def __init__(self, in_dim, out_dim,
                 prediction_steps=1,
                 n_negatives=1,
                 cross_sample_negatives=False,
                 sample_distance=None,
                 dropout=0.1,
                 offset=1,
                 balanced_classes=False):
        super(Wav2VecPredictor, self).__init__()
        
        with self.init_scope():
            self.project_to_steps = L.Deconvolution2D(in_dim, out_dim, (1, prediction_steps))
        
        self.n_negatives = n_negatives
        self.cross_sample_negatives = cross_sample_negatives
        self.sample_distance = None if sample_distance is None else int(sample_distance)
        self.dropout = dropout
        self.offset = offset
        self.balanced_classes = balanced_classes

    def sample_negatives(self, y):
        bsz, fsz, tsz = y.shape
        # utt x dim x frame => dim x utt x frame
        # dim (utt x frame)
        y = y.transpose(1, 0, 2).reshape(fsz, -1)

        if self.cross_sample_negatives:
            high = tsz * bsz
            assert self.sample_distance is None, 'sample distance is not supported with cross sampling'
        else:
            high = tsz if self.sample_distance is None else min(tsz, self.sample_distance)

        neg_idxs = np.random.randint(low=0, high=high, size=(bsz, self.n_negatives * tsz))
        if self.sample_distance is not None and self.sample_distance < tsz:
            raise NotImplementedError('no needed')
        if not self.cross_sample_negatives:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        negs = y[..., neg_idxs.reshape(-1)]
        negs = negs.reshape(fsz, bsz, self.n_negatives, tsz).transpose(2, 1, 0, 3)  # to Neg x utt x dim x frame
        return negs   
    
    def forward(self, x, y):
        xp = self.xp
        negatives = self.sample_negatives(y)
        y = F.expand_dims(y, axis=0)
        targets = F.concat([y, negatives], axis=0)
        n_targets = targets.shape[0]
        x = F.expand_dims(x, axis=-1)
        x = self.project_to_steps(x)  # utt x dim x Len x steps
        x = F.dropout(x, self.dropout)
        x = F.repeat(F.expand_dims(x, axis=0), n_targets, axis=0)

        copies, bsz, _, tsz, steps = x.shape
        steps = min(steps, tsz - self.offset)
        predictions = list()
        labels = xp.zeros(bsz * copies * (tsz - self.offset + 1) * steps - ((steps + 1) * steps // 2) * copies * bsz)
        start, end = 0, 0
        for i in range(steps):
            offset = i + self.offset
            end = start + (tsz - offset) * bsz * copies
            pos_num = (end - start) // copies
            _prediction = F.flatten(F.sum(x[..., :-offset, i] * targets[..., offset:], axis=2))
            predictions.append(_prediction)
            labels[start:start + pos_num] = 1.
            start = end
        predictions = F.hstack(predictions)
        assert end == predictions.shape[0], '{} != {}'.format(end, predictions.shape[0])
        # no weighted
        # if weights is not None:
        #     labels = (labels, weights)
        return predictions, labels
