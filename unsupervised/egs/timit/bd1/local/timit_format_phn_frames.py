#!/usr/bin/env python3

import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Obtain frames for timit phonemes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('phn_list', type=str,
                        help='list of phoneme files')
    args = parser.parse_args()
    with open(args.phn_list) as f:
        list_files = f.readlines()
    list_files = [l.rstrip('\n') for l in list_files]
    for phnfl in list_files:
        dots = np.loadtxt(phnfl, usecols=0)
        dots = dots.astype(np.float) / 160.  # conversion to frames in base shift 10 ms 
        dots = ' '.join([str(x) for x in dots.astype(np.int)])
        filename = phnfl.split('/')
        filename = '{}_{}'.format(filename[-2], filename[-1].split('.')[0])
        print(f'{filename}\t{dots}')
    