# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Class Declaration of Transformer's Training Subprocess."""
from __future__ import division

import collections
import logging
import math
import multiprocessing
import os
import six

# chainer related
import chainer
from chainer import cuda
from chainer import reporter
from chainer import training

from chainer import functions as F
from chainer import optimizers as O

from chainer.training.updaters.multiprocess_parallel_updater import gather_grads
from chainer.training.updaters.multiprocess_parallel_updater import gather_params
from chainer.training.updaters.multiprocess_parallel_updater import scatter_grads
from chainer.training.updaters.multiprocess_parallel_updater import scatter_params
from chainer.training.updaters.multiprocess_parallel_updater import _get_nccl_data_type
from chainer.training.updaters.multiprocess_parallel_updater import _calc_loss
from chainer.training import extension

import numpy as np

try:
    from cupy.cuda import nccl
    _available = True
except Exception:
    _available = False


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
        with cuda.get_device_from_array(x) as dev:  # noqa pylint: disable=no-member
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

    def __init__(self, train_iter, optimizer, converter, device, accum_grad=1):
        """Initialize Custom Updater."""
        super(CustomUpdater, self).__init__(
            train_iter, optimizer, converter=converter, device=device)
        logging.debug('using custom converter for transformer')

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Process main update routine for Custom Updater."""
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Get batch and convert into variables
        batch = train_iter.next()
        x = self.converter(batch, self.device)
        
        # #####################
        # ODM optimization
        # ######################
        optimizer.target.cleargrads()

        # Compute the loss at this time step and accumulate it
        loss = optimizer.target(*x)
        loss.backward()  # Backprop

        # compute the gradient norm to check if it is normal or not
        grad_norm = np.sqrt(sum_sqnorm(
            [p.grad for p in optimizer.target.params(False)]))
        logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.update()
        optimizer.target.cleargrads()  # Clear the parameter gradients


class _Worker(multiprocessing.Process):

    def __init__(self, proc_id, pipe, master):
        super(_Worker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.converter = master.converter
        self.model = master._master
        self.device = master._devices[proc_id]
        self.iterator = master._mpu_iterators[proc_id]
        self.n_devices = len(master._devices)

    def setup(self):
        _, comm_id = self.pipe.recv()
        self.comm = nccl.NcclCommunicator(self.n_devices, comm_id,
                                          self.proc_id)

        self.model.to_gpu(self.device)
        self.reporter = reporter.Reporter()
        self.reporter.add_observer('main', self.model)
        self.reporter.add_observers('main',
                                    self.model.namedlinks(skipself=True))

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        self.setup()
        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                dev.synchronize()
                break
            if job == 'do_map':
                self.converter.do_map = True
            if job == 'update':
                # For reducing memory
                self.model.cleargrads()

                batch = self.converter(self.iterator.next(), self.device)
                with self.reporter.scope({}):  # pass dummy observation
                    loss = _calc_loss(self.model, batch)

                self.model.cleargrads()
                loss.backward()
                del loss

                gg = gather_grads(self.model)
                nccl_data_type = _get_nccl_data_type(gg.dtype)
                null_stream = cuda.Stream.null
                self.comm.reduce(gg.data.ptr, gg.data.ptr, gg.size,
                                 nccl_data_type, nccl.NCCL_SUM, 0,
                                 null_stream.ptr)
                del gg
                self.model.cleargrads()
                gp = gather_params(self.model)
                nccl_data_type = _get_nccl_data_type(gp.dtype)
                self.comm.bcast(gp.data.ptr, gp.size, nccl_data_type, 0,
                                null_stream.ptr)
                scatter_params(self.model, gp)
                del gp


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
        from cupy.cuda import nccl  # noqa pylint: disable=import-error
        super(CustomParallelUpdater, self).__init__(
            train_iters, optimizer, converter=converter, devices=devices)
        self.nccl = nccl
        logging.debug('using custom parallel updater for transformer')
        self.do_map = True

    def setup_workers(self):
        if self._initialized:
            return
        self._initialized = True

        self._master.cleargrads()
        for i in six.moves.range(1, len(self._devices)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with cuda.Device(self._devices[0]):
            self._master.to_gpu(self._devices[0])
            if len(self._devices) > 1:
                comm_id = nccl.get_unique_id()
                self._send_message(('set comm_id', comm_id))
                self.comm = nccl.NcclCommunicator(len(self._devices),
                                                  comm_id, 0)

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Process main update routine for Custom Parallel Updater."""
        self.setup_workers()

        self._send_message(('update', None))
        with cuda.Device(self._devices[0]):  # noqa pylint: disable=no-member
            # For reducing memory
            optimizer = self.get_optimizer('main')
            batch = self.get_iterator('main').next()
            x = self.converter(batch, self._devices[0])

            loss = self._master(*x)
            loss.backward()

            # NCCL: reduce grads
            null_stream = cuda.Stream.null  # noqa pylint: disable=no-member
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
    
    def update(self):
        self.update_core()
        self.iteration += 1
        #if self.is_new_epoch:
        #    self.converter.do_map = self.do_map
        #    self._send_message(('do_map', None))


class VaswaniRule(extension.Extension):
    """Trainer extension to shift an optimizer attribute magically by Vaswani.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, d, warmup_steps=4000,
                 init=None, target=None, optimizer=None,
                 scale=1.):
        """Initialize Vaswani rule extension."""
        self._attr = attr
        self._d_inv05 = d ** (-0.5) * scale
        self._warmup_steps_inv15 = warmup_steps ** (-1.5)
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        """Initialize Optimizer values."""
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            self._init = self._d_inv05 * (1. * self._warmup_steps_inv15)
        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        """Forward extension."""
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self._d_inv05 * \
            min(self._t ** (-0.5), self._t * self._warmup_steps_inv15)
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        """Serialize extension."""
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)

    def _get_optimizer(self, trainer):
        """Obtain optimizer from trainer."""
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        """Update requested variable values."""
        setattr(optimizer, self._attr, value)
        self._last_value = value


class CustomConverter(object):
    """Custom Converter.

    Args:
        subsampling_factor (int): The subsampling factor.

    """

    def __init__(self, labels=False, folder=None, do_map=False):
        """Initialize subsampling."""
        self.labels = labels
        self.folder = folder
        self.boundaries_dir = 'init_boundaries'
        self.do_map = do_map
        pass

    def get_boundaries(self, names):
        boundaries = list()
        folder = self.folder
        bdir = self.boundaries_dir
        for i in range(len(names)):
            b = np.load(os.path.join(folder, bdir, f'{names[i]}_lbl.npy')).astype(np.bool)
            boundaries.append(np.argwhere(b)[:, 0].tolist())
        return boundaries

    def get_probs(self, names):
        folder = self.folder
        bdir = self.boundaries_dir
        pbs = F.pad_sequence([np.load(os.path.join(folder, bdir, f'{x}_pbs.npy')) for x in names], padding=0).data
        return pbs

    def __call__(self, batch, device=None):
        """Perform subsampling.

        Args:
            batch (list): Batch that will be sabsampled.
            device (chainer.backend.Device): CPU or GPU device.

        Returns:
            chainer.Variable: xp.array that are padded and subsampled from batch.
            xp.array: xp.array of the length of the mini-batches.
            chainer.Variable: xp.array that are padded and subsampled from batch.

        """
        assert len(batch) == 1
        batch, names = batch[0]
        xs, _ = batch

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs], dtype=np.int)
        if self.do_map:
            probs = self.get_probs(names)
            return xs, ilens, None, None, probs
        else:
            boundaries = self.get_boundaries(names)
            return xs, ilens, None, boundaries, None

    def return_probs(self, batch):
        batch, names = batch
        xs, _ = batch

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs], dtype=np.int)
        boundaries = self.get_boundaries(names)
        probs = self.get_probs(names)
        return xs, ilens, probs, boundaries, names
