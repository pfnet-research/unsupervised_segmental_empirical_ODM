
from distutils.util import strtobool
import logging
import math

import chainer
from chainer import links as L
from chainer import functions as F
from chainer import reporter

from espnet.nets.asr_interface import ASRInterface

from espnet.nets.chainer_backend.pretrain_net.aggregator import ConvAggregator
from espnet.nets.chainer_backend.pretrain_net.feat_extract import ConvFeatureExtractionModel
from espnet.nets.chainer_backend.pretrain_net.wav2vec_prediction import Wav2VecPredictor

import numpy as np


class AcousticModel(ASRInterface, chainer.Chain):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group("unsupervised model setting")
        group.add_argument('--prediction-steps', type=int, default=12,
                           help='number of steps ahead to predict')
        group.add_argument('--sample-distance', type=str, default=None,
                           help='sample distance from target. does not work properly with cross-sampling')
        group.add_argument('--cross-sample-negatives', type=strtobool, default=False,
                           help='whether to sample negatives across examples in the same batch')
        group.add_argument('--num-negatives', type=int, default=1,
                           help='number of negative examples')
        group.add_argument('--conv-feature-layers', type=str, metavar='EXPR',
                            help='convolutional feature extraction layers [(dim, kernel_size, stride), ...]')
        group.add_argument('--conv-aggregator-layers', type=str, metavar='EXPR',
                            help='convolutional feature extraction layers [(dim, kernel_size, stride), ...]')
        group.add_argument('--dropout-rate', type=float, default=0.0,
                           help='dropout to apply within the model')
        group.add_argument('--dropout-features', type=float, default=0.0,
                           help='dropout to apply to the features')
        group.add_argument('--dropout-agg', type=float, default=0.0,
                           help='dropout to apply after aggregation step')
        group.add_argument('--encoder', type=str, choices=['cnn'], default='cnn',
                           help='type of encoder to use')
        group.add_argument('--aggregator', type=str, choices=['cnn', 'gru'], default='cnn',
                            help='type of aggregator to use')
        group.add_argument('--gru-dim', type=int, default=512,
                           help='GRU dimensionality')
        group.add_argument('--no-conv-bias', type=strtobool, default=False,
                            help='if set, does not learn bias for conv layers')
        group.add_argument('--agg-zero-pad', type=strtobool, default=False,
                            help='if set, zero pads in aggregator instead of repl pad')
        group.add_argument('--skip-connections-feat', type=strtobool, default=False,
                            help='if set, adds skip connections to the feature extractor')
        group.add_argument('--skip-connections-agg', type=strtobool, default=False,
                            help='if set, adds skip connections to the aggregator')
        group.add_argument('--residual-scale', type=float, default=0.5,
                            help='scales residual by sqrt(value)')
        group.add_argument('--log-compression', type=strtobool, default=False,
                            help='if set, adds a log compression to feature extractor')
        group.add_argument('--balanced-classes', type=strtobool, default=False,
                            help='if set, loss is scaled to balance for number of negatives')
        group.add_argument('--project-features', type=str, default='none', 
                            choices=['none', 'same', 'new'],
                            help='if not none, features are projected using the (same or new) aggregator')
        group.add_argument('--non-affine-group-norm', type=strtobool, default=False,
                            help='if set, group norm is not affine')
        group.add_argument('--offset', type=str, default='auto',
                           help='if set, introduces an offset from target to predictions. '
                                             'if set to "auto", it is computed automatically from the receptive field')
        # optim related
        group.add_argument('--noam-dims', type=int, default=512,
                            help='')
        group.add_argument('--noam-steps', type=int, default=25000,
                            help='')
        group.add_argument('--noam-scale', type=float, default=1.0,
                            help='')
        return parser

    def __init__(self, idim, odim, args, flag_return=True):
        chainer.Chain.__init__(self)
        self.prediction_steps = args.prediction_steps
        self.dropout = args.dropout_rate
        feature_enc_layers = eval(args.conv_feature_layers)

        offset = args.offset
        if offset == 'auto':
            assert args.encoder == 'cnn'
            jin = 0
            rin = 0
            for _, k, stride in feature_enc_layers:
                if rin == 0:
                    rin = k
                rin = rin + (k - 1) * jin
                if jin == 0:
                    jin = stride
                else:
                    jin *= stride
            offset = math.ceil(rin / jin)
            self.max_ilen = rin * jin // 2
        offset = int(offset)

        with self.init_scope():
            if args.encoder == 'cnn':
                self.feature_extractor = ConvFeatureExtractionModel(
                    idim,
                    conv_layers=feature_enc_layers,
                    dropout=self.dropout,
                    skip_connections=args.skip_connections_feat,
                    residual_scale=args.residual_scale
                )
                embed = feature_enc_layers[-1][0]
            else:
                raise Exception('unknown encoder type ' + args.encoder)
            
            if args.aggregator == 'cnn':
                agg_layers = eval(args.conv_aggregator_layers)
                agg_dim = agg_layers[-1][0]
                self.feature_aggregator = ConvAggregator(
                    embed=embed,
                    conv_layers=agg_layers,
                    dropout=self.dropout,
                    skip_connections=args.skip_connections_agg,
                    residual_scale=args.residual_scale,
                    nobias=args.no_conv_bias,
                    zero_pad=args.agg_zero_pad
                )
            else:
                raise Exception('unknown aggregator type ' + args.aggregator)
            
            self.wav2vec_predict = Wav2VecPredictor(
                in_dim=agg_dim,
                out_dim=embed,
                prediction_steps=args.prediction_steps,
                n_negatives=args.num_negatives,
                cross_sample_negatives=args.cross_sample_negatives,
                sample_distance=args.sample_distance,
                dropout=self.dropout,
                offset=offset,
                balanced_classes=args.balanced_classes
            )

    def forward(self, xs, ilens, ys, calculate_attentions=False):
        """VGG2L forward propagation.

        Args:
            xs (chainer.Variable): Batch of padded charactor ids. (B, Tmax)
            ilens (chainer.Variable): Batch of length of each features. (B,)

        Returns:
            chainer.Variable: Subsampled vector of xs.
            chainer.Variable: Subsampled vector of ilens.

        """
        # batchsize = len(ilens)
        xp = self.xp
        logging.info(self.__class__.__name__ + ' input lengths: ' + str(ilens))

        # x: utt x frame x dim
        # => utt x dim x frame
        xs = F.pad_sequence(xs).transpose(0, 2, 1)
        if xs.shape[2] < self.max_ilen:
            # dirty enlargement
            xs = F.repeat(xs, 3, axis=2)
        xs = xp.array(xs.data)
        logging.info(xs.shape)

        features = self.feature_extractor(xs)
        logging.info(features.shape)
        xs = F.dropout(features, self.dropout)
        xs = self.feature_aggregator(xs)
        logging.info(xs.shape)
        xs = F.dropout(xs, self.dropout)

        # if self.project_features is not None:
        #     features = self.project_features(features)

        predictions, labels = self.wav2vec_predict(xs, features)
        self.loss = F.sigmoid_cross_entropy(predictions, labels)

        loss_data = self.loss.data
        if not math.isnan(loss_data):
            logging.info('mtl loss:' + str(loss_data))
            reporter.report({'loss': self.loss}, self)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)

        return self.loss

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

        # x: utt x frame x dim
        # => utt x dim x frame
        x = x[None, :].transpose(0, 2, 1)
        z = self.feature_extractor(x).data
        c = self.feature_aggregator(z).data
        return c, z

    @staticmethod
    def CustomConverter(labels=False):
        from espnet.nets.chainer_backend.pretrain_net.training import CustomConverter
        return CustomConverter(labels)
    
    @staticmethod
    def CustomUpdater(iterator, optimizer, converter, device='-1', accum_grad=1):
        from espnet.nets.chainer_backend.pretrain_net.training import CustomUpdater
        return CustomUpdater(iterator, optimizer, converter=converter, device=device, accum_grad=accum_grad)
