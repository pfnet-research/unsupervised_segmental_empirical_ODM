import multiprocessing

import chainer

from chainer.training import extension
from chainer.backends import cuda
from chainer import functions as F
import logging

import numpy as np
import os
import six
import time

try:
    from cupy.cuda import nccl
    _available = True
except Exception:
    _available = False


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
        local_score_0 = pbt + local_pred_scores 
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


def process_batch(batches, converter, model, beam, min_diff_frame):
    init_dir = 'init_boundaries'
    refined_dir = 'refined_boundaries'
    converter.boundaries_dir = init_dir
    save_folder = os.path.join(converter.folder, refined_dir)
    os.makedirs(save_folder, exist_ok=True)
    xp = model.xp
    for batch in batches:
        x, ilens, probs, _, names = converter.return_probs(batch)
        x = model(x, ilens, None, None, do_map=True)
        dims = x.shape[-1] + 1
        _, hyps = beam_search(x.data, ilens, beam, xp, probs, dims, model.rnnlm.predict)
        for i in range(x.shape[0]):
            utt_hyps = hyps[i, :ilens[i] - 1]
            utt_hyps = np.pad(utt_hyps, [[1, 0], [0, 0]], mode='edge')
            new_boundaries = utt_hyps[1:] != utt_hyps[:-1]
            new_boundaries = np.pad(new_boundaries, [[1, 0], [0, 0]], mode='constant', constant_values=True)
            idx = np.argwhere(new_boundaries[:, 0])[:, 0]  # select top
            ### last clean up idx
            min_len = (idx[1:] - idx[:-1]) > min_diff_frame
            idx = idx[np.pad(min_len, [1, 0], mode='constant', constant_values=True)].tolist()
            np.save(os.path.join(save_folder, f'{names[i]}_lbl'), idx)
        #     if len(lens) < 1:
        #         np.save(os.path.join(save_folder, f'{names[i]}_lbl'), idx[0])
        #     else:
        #         np.save(os.path.join(save_folder, f'{names[i]}_lbl'), idx[lens[-1]])
        #     logging.info(len(new_boundaries))
    converter.boundaries_dir = refined_dir


class _Worker(multiprocessing.Process):

    def __init__(self, proc_id, pipe, master):
        super(_Worker, self).__init__()
        self.proc_id = proc_id
        self.pipe = pipe
        self.model = master.model
        self.train_iter = master.train_iters[proc_id]
        self.converter = master.converter
        self.valid_iter = master.valid_iters[proc_id]
        self.valid_converter = master.valid_converter
        self.device = master.devices[proc_id]
        self.n_devices = len(master.devices)
        self.beam = master.beam
        self.min_diff_frame = master.min_diff_frame

    def setup(self):
        _, comm_id = self.pipe.recv()
        self.comm = nccl.NcclCommunicator(self.n_devices, comm_id,
                                          self.proc_id)
        self.model.to_device(self.device)
        
    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        self.setup()
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            process_batch(self.train_iter, self.converter, self.model, self.beam, self.min_diff_frame)
            process_batch(self.valid_iter, self.valid_converter, self.model, self.beam, self.min_diff_frame)
        self.device.device.synchronize()


class BoundaryUpdate(extension.Extension):
    """An extension enabling shuffling on an Iterator"""

    def __init__(self, model, train_iters,
                 converter, valid_iters, valid_converter,
                 devices, train_updater, min_diff_frame=2):
        """ """
        if not _available:
            raise Exception('NCCL not found')
        self.train_iters = train_iters
        self.converter = converter
        self.model = model
        self.valid_iters = valid_iters
        self.valid_converter = valid_converter
        self.beam = 3
        if isinstance(devices, dict):
            devices = devices.copy()
            main = devices.pop('main')
            devices = list(six.itervalues(devices))
            devices = [main] + devices
        elif isinstance(devices, (list, tuple)):
            devices = list(devices)

        self.devices = devices
        self._pipes = []
        self._workers = []
        self.comm = None
        self.min_diff_frame = min_diff_frame

    def __call__(self, trainer):
        """Calls the enabler on the given iterator

        :param trainer: The iterator
        """
        # Update boundaries for train data
        logging.info('Initializing refining')
        if isinstance(self.devices, int):
            self.run_single_process()
        else:
            self.run_multi_process()
        logging.info('Refined finished')
        exit(1)
    
    def run_single_process(self):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            process_batch(self.train_iters[0], self.converter, self.model, self.beam, self.min_diff_frame)
            process_batch(self.valid_iters[0], self.valid_converter, self.model, self.beam, self.min_diff_frame)
    
    def run_multi_process(self):
        # self.setup_workers()
        #self._send_message(('update', None))
        # with chainer.using_device(self.devices[0]):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for i in range(len(self.train_iters)):
                process_batch(self.train_iters[i], self.converter, self.model, self.beam, self.min_diff_frame)
                process_batch(self.valid_iters[i], self.valid_converter, self.model, self.beam, self.min_diff_frame)
        #self._send_message(('finalize', None))
        #for worker in self._workers:
        #    worker.join()

    def _send_message(self, message):
        for pipe in self._pipes:
            pipe.send(message)

    def setup_workers(self):
        for i in six.moves.range(1, len(self.devices)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with chainer.using_device(self.devices[0]):
            self.model.to_device(self.devices[0])
            if len(self.devices) > 1:
                comm_id = nccl.get_unique_id()
                self._send_message(('set comm_id', comm_id))
                self.comm = nccl.NcclCommunicator(
                    len(self.devices), comm_id, 0)
