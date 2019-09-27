import chainer
from chainer import functions as F
from chainer import reporter

import logging
import math
import numpy as np

from espnet.nets.asr_interface import ASRInterface

class Boundary(ASRInterface, chainer.Chain):
    @staticmethod
    def add_arguments(parser):
        """Customize flags for transformer setup.

        Args:
            parser (Namespace): Training config.

        """
        group = parser.add_argument_group("Boundary model setting")
        group.add_argument("--layer-type", type=str, default="rnn",
                           choices=["rnn", "res"],
                           help='Type of boundary model')
        group.add_argument('--dropout-rate', default=0.3, type=float,
                           help='Dropout rate for the encoder')
        # RNN Based Arguments
        group.add_argument('--bound-rnn-cells', default=64, type=int,
                           help='')
        group.add_argument('--bound-rnn-units', default=32, type=int,
                           help='')
        # RES Based Arguments
        group.add_argument('--bound-res-units', default=512, type=int,
                           help='')
        group.add_argument('--bound-res-kernel', default=3, type=int,
                           help='')
        group.add_argument('--negatives-samples', default=3, type=int,
                           help='')
        group.add_argument('--min-diff-frame', default=1, type=int,
                           help='')
        group.add_argument('--boundary-threshold', type=float, default=0.009,
                            help='')
        return parser

    def __init__(self, idim, odim, args, flag_return=True):
        #(self, dims, units, cells):
        chainer.Chain.__init__(self)
        self.is_rnn = False
        with self.init_scope():
            if args.layer_type == "rnn":
                self.is_rnn = True
                from espnet.nets.chainer_backend.net_boundary.rnn_boundary import RNN_B
                self.model = RNN_B(idim, args.bound_rnn_cells, args.bound_rnn_units,
                                   args.dropout_rate)
            else:
                from espnet.nets.chainer_backend.net_boundary.res_boundary import RES_B
                self.model = RES_B(idim, args.bound_res_units,
                                 args.dropout_rate,
                                 args.bound_res_kernel,
                                 args.negatives_samples,
                                 args.min_diff_frame,
                                 args.boundary_threshold)
        self.layer_type = args.layer_type
    
    def forward_rnn(self, xs, ilens):
        xp = self.xp
        amax = min(ilens)
        with chainer.no_backprop_mode():
            xs = xp.array(F.pad_sequence(xs).data[:, :amax])
        logging.info(xs.shape)

        self.loss = 0
        state = None
        for i in range(amax - 1):
            state, ys = self.model(xs[:, i], state)
            self.loss += F.mean_squared_error(xs[:, i + 1], ys)
        loss_data = self.loss.data
        self.loss /= len(ilens)
        if not math.isnan(loss_data):
            logging.info('loss:' + str(loss_data))
            reporter.report({'loss': loss_data}, self)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss
    
    def forward_res(self, xs, ilens, ys):
        loss, precision, recall = self.model(xs, ilens, ys)
        self.loss = loss
        loss_data = self.loss.data

        if not math.isnan(loss_data):
            logging.info('loss:' + str(loss_data))
            logging.info(f'precision: {precision}')
            logging.info(f'recall: {recall}')
            reporter.report({'loss': loss_data}, self)
            reporter.report({'precision': precision}, self)
            reporter.report({'recall': recall}, self)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def forward(self, xs, ilens, ys=None):
        if self.layer_type == 'rnn':
            return self.forward_rnn(xs, ilens)
        else:
            return self.forward_res(xs, ilens, ys)

    def get_boundary(self, xs, boundary_threshold, args, rebound=False):
        if not rebound:
            self.probs = None
            self.signal = None
            self.local_maxima = None
            with chainer.no_backprop_mode():
                probs, signal = self.model.get_boundary(xs, args.boundary_weight, args.boundary_type)
            local_maxima = (signal[2:] < signal[1:-1]) * (signal[1:-1] > signal[:-2])
            local_maxima = np.pad(local_maxima, 1, mode='constant', constant_values=False)
            self.probs = probs
            self.signal = signal
            self.local_maxima = local_maxima
        boundaries = (self.local_maxima * (self.signal >= boundary_threshold))
        boundaries[0] = True
        assert (boundaries.shape[0] == xs.shape[0])
        return probs, boundaries
    
    def serialize(self, serializer):
        """Serialize state dict."""
        # type=(chainer AbstractSerializer) -> NoneW
        super(chainer.Chain, self).serialize(serializer) 
        d = self.model.__dict__
        for name in self.model._children:
            d[name].serialize(serializer[name])

    @staticmethod
    def CustomConverter(type_model):
        from espnet.nets.chainer_backend.net_boundary.training import CustomConverter
        return CustomConverter()

    @staticmethod
    def CustomUpdater(
            iterator, optimizer, converter, device='-1', accum_grad=1):
        from espnet.nets.chainer_backend.net_boundary.training import CustomUpdater
        return CustomUpdater(
            iterator, optimizer, converter=converter, device=device)
    
    @staticmethod
    def CustomParallelUpdater(
            iterators, optimizer, converter, devices):
        from espnet.nets.chainer_backend.net_boundary.training import CustomParallelUpdater
        return CustomParallelUpdater(
            iterators, optimizer, converter=converter, devices=devices)
