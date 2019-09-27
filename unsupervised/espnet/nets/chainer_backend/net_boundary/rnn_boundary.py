import chainer
from chainer import links as L
from chainer import functions as F

import logging
import numpy as np

from chainer.functions.activation import lstm


class RNN_B(chainer.Chain):
    def __init__(self, dims, units, cells, dropout=0.3):
        super(RNN_B, self).__init__()
        with self.init_scope():
            self.dnn_in = L.Linear(dims, units)
            self.rnn_enc_1 = L.StatelessLSTM(units, cells)
            self.rnn_enc_2 = L.StatelessLSTM(cells, cells)
            self.rnn_dec_1 = L.StatelessLSTM(cells, cells)
            self.rnn_dec_2 = L.StatelessLSTM(cells, units)
            self.dnn_out = L.Linear(units, dims)
            self.out = L.Linear(dims, dims)
        self.dropout = dropout
        self.cells = cells
        self.units = units
        self.n_layers = 4

    def forward(self, x, state):
        if state is None:
            state = {'c': [None] * self.n_layers, 'h': [None] * self.n_layers}
        c = [None] * self.n_layers
        h = [None] * self.n_layers
        x = F.relu(self.dnn_in(F.dropout(x, self.dropout)))
        c[0], h[0] = self.rnn_enc_1(state['c'][0], state['h'][0], F.dropout(x, self.dropout))
        c[1], h[1] = self.rnn_enc_2(state['c'][1], state['h'][1], F.dropout(h[0], self.dropout))
        c[2], h[2] = self.rnn_dec_1(state['c'][2], state['h'][2], F.dropout(h[1], self.dropout))
        c[3], h[3] = self.rnn_dec_2(state['c'][3], state['h'][3], F.dropout(h[2], self.dropout))
        state = {'c': c, 'h': h}
        x = F.relu(self.dnn_out(h[3]))
        x = self.out(x)
        return state, x

    def get_boundary(self, xs, weight, b_type):
        logging.info(f'Forwarding data with length: {xs.shape[0]}')
        if weight == 1.0:
            logging.info('Pure GAS')
        elif weight == 0.0:
            logging.info('Pure RPM')
        else:
            logging.info('RPM + GAS')
        
        layers = ['rnn_enc_1', 'rnn_enc_2', 'rnn_dec_1', 'rnn_dec_2']
        n_layers = len(layers)
        
        if b_type == 'up':
            layer_idx = [0, 1]
        elif b_type == 'all':
            layer_idx = [0, 1, 2, 3]
        else:
            raise Exception('not implemented')

        do_gas, do_rpm = False, False
        if weight > 0.0:
            gas = list()
            do_gas = True
        if weight < 1.0:
            rpm = list()
            do_rpm = True

        hs = [None] * n_layers
        cs = [None] * n_layers
        fg = [None] * n_layers
        ig = [None] * n_layers
        og = [None] * n_layers
        probs = 0
        # Forward
        x = F.relu(self.dnn_in(xs))
        for i in range(x.shape[0] - 1):
            hx = x[i][None, :]
            for l in range(n_layers):
                ig[l], fg[l], og[l], cs[l], hs[l] = self.get_rnn_forward(layers[l], cs[l], hs[l], hx, do_gas)
                hx = hs[l]
            if do_gas:
                fg = np.mean(np.concatenate([fg[n] for n in layer_idx], axis=-1))
                ig = np.mean(np.concatenate([ig[n] for n in layer_idx], axis=-1))
                og = np.mean(np.concatenate([og[n] for n in layer_idx], axis=-1))
                _gas = np.stack([ig, fg, og])
                gas.append(_gas)
                
                fg = [None] * n_layers
                ig = [None] * n_layers
                og = [None] * n_layers
            if do_rpm:
                y = F.mean_squared_error(self.out(F.relu(self.dnn_out(hx))), xs[i + 1][None, :])
                rpm.append(y.data)
        if do_rpm:
            rpm = np.stack(rpm, axis=0)
            drpm = rpm[1:] - rpm[:-1]
            drpm = np.pad(drpm, [2, 0], mode='edge')
        else:
            drpm = 0.0

        if do_gas:
            gas = np.stack(gas, axis=0)
            probs = F.softmax(gas, axis=-1).data[:, 1]
            gas = gas[:, 1]
            dgas = gas[1:] - gas[:-1]
            dgas = np.pad(dgas, [2, 0], mode='edge')
        else:
            dgas = 0.
        I_t = (1 - weight) * drpm + weight * dgas
        return probs, I_t

    def get_rnn_forward(self, layer, c, h, x, do_gas=False):
        # a: cell input
        # i: input gate
        # f: forget gate
        # o: output gate
        xp = self.xp
        if c is None:
            c = chainer.Variable(
                    xp.zeros((1, self[layer].state_size), dtype=x.dtype))
        lstm_in = self[layer].upward(x)
        if h is not None:
            lstm_in += self[layer].lateral(h)
        cs, hs = lstm.lstm(c, lstm_in)

        if do_gas:
            _, i, f, o = lstm._extract_gates(lstm_in)
            i = lstm._sigmoid(i.data)
            f = lstm._sigmoid(f.data)
            o = lstm._sigmoid(o.data)
        else:
            f = None
        return i, f, o, cs, hs
