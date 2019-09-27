

import logging

import chainer
import numpy as np

from chainer import links as L
from chainer import functions as F


class ConvLayer(chainer.Chain):
    def __init__(self, n_in, n_out, ksize, stride,
                 dropout=0.0, nobias=False):
        super(ConvLayer, self).__init__()
        ka = ksize // 2
        kb = ka - 1 if ksize % 2 == 0 else ka

        with self.init_scope():
            self.conv = L.Convolution1D(n_in, n_out, ksize, stride, nobias=nobias)
            self.norm = L.GroupNormalization(1, n_out)
        self.dropout = dropout
        self.pad = ka + kb
    
    def forward(self, x):
        # padding dims only really make sense for stride = 1
        x = F.pad(x, [[0, 0], [0, 0], [self.pad, 0]], mode='edge')
        x = F.relu(self.norm(F.dropout(self.conv(x), self.dropout)))
        return x


class ConvAggregator(chainer.Chain):
    def __init__(self, embed, conv_layers, dropout=0.1,
                 skip_connections=False,
                 residual_scale=1.0,
                 nobias=False,
                 zero_pad=False):
        super(ConvAggregator, self).__init__()
        in_d = embed
        self.conv_layers = list()
        self.residual_proj = list()
        with self.init_scope():
            for i, (dim, k, stride) in enumerate(conv_layers):
                res_name = None
                if in_d != dim and skip_connections:
                    res_name = f'res{i}'
                    setattr(self, res_name, L.Convolution1D(in_d, dim, 1, nobias=nobias))
                self.residual_proj.append(res_name)
                name = f'conv{i}'
                setattr(self, name, ConvLayer(in_d, dim, k, stride, dropout))
                self.conv_layers.append(name)
                in_d = dim
        self.dropout = dropout
        self.skip_connections = skip_connections
        self.residual_scale = residual_scale
        
    def forward(self, x):
        for rproj, conv in zip(self.residual_proj, self.conv_layers):
            residual = x
            x = self[conv](x)
            if self.skip_connections:
                if rproj is not None:
                    residual = self[rproj](residual)
                x = (x + residual) * self.residual_scale
        return x
