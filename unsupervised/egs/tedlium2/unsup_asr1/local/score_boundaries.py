#!/usr/bin/env python
# encoding: utf-8

# Copyright 2018 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import glob
import logging
import os
import sys

from distutils.util import strtobool

from espnet.utils.cli_utils import get_commandline_args

import math
import numpy as np


def r_val_eval(u_p, u_r):
    if u_r == 0 or u_p == 0:
        u_f = -1.
        u_r_val = -1.
    else:
        u_f = 2 * u_p * u_r / (u_p + u_r)
        u_os = (u_r/u_p - 1) * 100
        u_r_val = 1 - (math.fabs(math.sqrt((100-u_r)*(100-u_r) + \
         math.pow(u_os, 2))) + math.fabs( (u_r - 100 - u_os)/math.sqrt(2))) / 200
    return u_r_val * 100, u_f


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


def get_parser():
    parser = argparse.ArgumentParser(
        description='add multiple json values to an input or output value',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('true_folder', type=str,
                        help='')
    parser.add_argument('predicted_folder', type=str,
                        help='')
    parser.add_argument('--verbose', '-V', default=1, type=int,
                        help='Verbose option')
    parser.add_argument('--tolerance-window', default=2, type=int,
                        help='')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = '%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s'
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(
            level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    # make intersection set for utterance keys
    true_files = glob.glob(os.path.join(args.true_folder, '*lbl.npy'))
    precision = np.zeros((len(true_files)))
    recall = np.zeros((len(true_files)))
    for i, true_file in enumerate(true_files):
        true_labels = np.load(true_file).astype(np.bool)
        true_labels = np.array([x for x in np.argwhere(true_labels)[:, 0]])
        filename = os.path.basename(true_file)
        predicted_labels = np.load(os.path.join(args.predicted_folder, filename)).astype(np.bool)
        predicted_labels = np.array([x for x in np.argwhere(predicted_labels)[:, 0]])
        precision[i] = tolerance_precision(true_labels, predicted_labels, args.tolerance_window)
        recall[i] = tolerance_recall(true_labels, predicted_labels, args.tolerance_window)
    files = precision.shape[0]
    precision = np.mean(precision) * 100.
    recall = np.mean(recall) * 100.

    if recall == 0. and precision == 0.:
        Fscore = -1.
        Rval = -1.
    else:
        Rval, Fscore = r_val_eval(precision, recall)
    logging.info(f'Results= Files: {files}, Precision:{precision}, Recall: {recall}, F-Score: {Fscore}, R-Value: {Rval}')

