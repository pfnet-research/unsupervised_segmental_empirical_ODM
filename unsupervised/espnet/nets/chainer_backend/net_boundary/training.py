# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Class Declaration of Transformer's Training Subprocess."""
from __future__ import division

import collections
import logging
import math
import six

# chainer related
from chainer import cuda
from chainer import training

from chainer import functions as F

from chainer.training.updaters.multiprocess_parallel_updater import gather_grads
from chainer.training.updaters.multiprocess_parallel_updater import gather_params
from chainer.training.updaters.multiprocess_parallel_updater import scatter_grads

from chainer.training import extension

import numpy as np


# copied from https://github.com/chainer/chainer/blob/master/chainer/optimizer.py
def sum_sqnorm(arr):
    """Calculate the norm of the array.

    Args:
        arr (numpy.ndarray)

    Returns:
        Float: Sum of the norm calculated from the given array.

    """
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device_from_array(x) as dev:
            if x is not None:
                x = x.ravel()
                s = x.dot(x)
                sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


class CustomUpdater(training.StandardUpdater):
    """Custom updater for chainer.

    Args:
        train_iter (iterator | dict[str, iterator]): Dataset iterator for the
            training dataset. It can also be a dictionary that maps strings to
            iterators. If this is just an iterator, then the iterator is
            registered by the name ``'main'``.
        optimizer (optimizer | dict[str, optimizer]): Optimizer to update
            parameters. It can also be a dictionary that maps strings to
            optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter (espnet.asr.chainer_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.
        device (int or dict): The destination device info to send variables. In the
            case of cpu or single gpu, `device=-1 or 0`, respectively.
            In the case of multi-gpu, `device={"main":0, "sub_1": 1, ...}`.
        accum_grad (int):The number of gradient accumulation. if set to 2, the network
            parameters will be updated once in twice, i.e. actual batchsize will be doubled.

    """

    def __init__(self, train_iter, optimizer, converter, device):
        """Initialize Custom Updater."""
        super(CustomUpdater, self).__init__(
            train_iter, optimizer, converter=converter, device=device)
        self.forward_count = 0
        self.start = True
        self.device = device
        self.is_rnn = optimizer.target.is_rnn
        logging.debug('using custom converter for transformer')

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Process main update routine for Custom Updater."""
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get batch and convert into variables
        batch = train_iter.next()
        x = self.converter(batch, self.device)
        if self.start:
            optimizer.target.cleargrads()
            self.start = False

        # Compute the loss at this time step and accumulate it
        loss = optimizer.target(*x)
        loss.backward()  # Backprop
        if self.is_rnn:
            loss.unchain_backward()  # Truncate the graph
        # compute the gradient norm to check if it is normal or not
        grad_norm = np.sqrt(sum_sqnorm(
            [p.grad for p in optimizer.target.params(False)]))
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.update()
        optimizer.target.cleargrads()  # Clear the parameter gradients


class CustomParallelUpdater(training.updaters.MultiprocessParallelUpdater):
    """Custom Parallel Updater for chainer.

    Defines the main update routine.

    Args:
        train_iter (iterator | dict[str, iterator]): Dataset iterator for the
            training dataset. It can also be a dictionary that maps strings to
            iterators. If this is just an iterator, then the iterator is
            registered by the name ``'main'``.
        optimizer (optimizer | dict[str, optimizer]): Optimizer to update
            parameters. It can also be a dictionary that maps strings to
            optimizers. If this is just an optimizer, then the optimizer is
            registered by the name ``'main'``.
        converter (espnet.asr.chainer_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.
        device (torch.device): Device to which the training data is sent. Negative value
            indicates the host memory (CPU).
        accum_grad (int):The number of gradient accumulation. if set to 2, the network
            parameters will be updated once in twice, i.e. actual batchsize will be doubled.

    """

    def __init__(self, train_iters, optimizer, converter, devices):
        """Initialize custom parallel updater."""
        from cupy.cuda import nccl
        super(CustomParallelUpdater, self).__init__(
            train_iters, optimizer, converter=converter, devices=devices)
        self.is_rnn = optimizer.target.is_rnn
        self.nccl = nccl
        logging.debug('using custom parallel updater for transformer')

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Process main update routine for Custom Parallel Updater."""
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):
            # For reducing memory
            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            x = self.converter(batch, self._devices[0])

            loss = self._master(*x)
            loss.backward()
            if self.is_rnn:
                loss.unchain_backward()  # Truncate the graph
            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.comm is not None:
                gg = gather_grads(self._master)
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 self.nccl.NCCL_FLOAT,
                                 self.nccl.NCCL_SUM,
                                 0, null_stream.ptr)
                scatter_grads(self._master, gg)
                del gg

            # update parameters
            # check gradient value
            grad_norm = np.sqrt(sum_sqnorm(
                [p.grad for p in optimizer.target.params(False)]))
            logging.info('grad norm={}'.format(grad_norm))

            # update
            if math.isnan(grad_norm):
                logging.warning('grad norm is nan. Do not update model.')
            else:
                optimizer.update()
            self._master.cleargrads()

            if self.comm is not None:
                gp = gather_params(self._master)
                self.comm.bcast(gp.data.ptr, gp.size, self.nccl.NCCL_FLOAT,
                                0, null_stream.ptr)


class CustomConverter(object):
    """Custom Converter.

    Args:
        subsampling_factor (int): The subsampling factor.

    """

    def __init__(self, subsampling_factor=1):
        """Initialize subsampling."""
        pass

    def __call__(self, batch, device):
        """Perform subsampling.

        Args:
            batch (list): Batch that will be sabsampled.
            device (chainer.backend.Device): CPU or GPU device.

        Returns:
            chainer.Variable: xp.array that are padded and subsampled from batch.
            xp.array: xp.array of the length of the mini-batches.
            chainer.Variable: xp.array that are padded and subsampled from batch.

        """
        # For transformer, data is processed in CPU.
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]
        
        # get batch of lengths of input sequences
        ilens = [x.shape[0] for x in xs]
        # xs = F.pad_sequence(xs, padding=-1).data
        return xs, ilens, ys
