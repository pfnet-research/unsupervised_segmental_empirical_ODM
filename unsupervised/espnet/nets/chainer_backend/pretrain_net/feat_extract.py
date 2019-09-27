
import logging

import chainer
import numpy as np

from chainer import links as L
from chainer import functions as F


class ConvLayer(chainer.Chain):
    def __init__(self, n_in, n_out, ksize, stride,
                 dropout=0.0):
        super(ConvLayer, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution1D(n_in, n_out, ksize, stride, nobias=True)
            self.norm = L.GroupNormalization(1, n_out)
        self.dropout = dropout
    
    def forward(self, x):
        x = F.relu(self.norm(F.dropout(self.conv(x), self.dropout)))
        return x


class ConvFeatureExtractionModel(chainer.Chain):
    def __init__(self, idim, conv_layers, dropout=0.1,
                 skip_connections=False,
                 residual_scale=1.0):
        super(ConvFeatureExtractionModel, self).__init__()
        in_d = idim
        with self.init_scope():
            for i, (dim, k, stride) in enumerate(conv_layers):
                setattr(self, f'conv{i}', ConvLayer(in_d, dim, k, stride, dropout))
                in_d = dim    
        self.dropout = dropout
        self.nlayers = len(conv_layers)
        self.skip_connections = skip_connections
        self.residual_scale = residual_scale
        
    def forward(self, x):
        # Batch x Frames x Feats => Batch x Feats x Frames
        for i in range(self.nlayers):
            residual = x
            res_shape = residual.shape
            x = self[f'conv{i}'](x)
            x_shape = x.shape
            if self.skip_connections and x_shape[1] == res_shape[1]:
                tsz = x_shape[2]
                r_tsz = res_shape[2]
                residual = residual[..., :: r_tsz // tsz][..., : tsz]
                x = (x + residual) * self.residual_scale
        return x

