#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import print_function
from __future__ import unicode_literals

import argparse
import codecs
import re
import sys

is_python2 = sys.version_info[0] == 2


def exist_or_not(i, match_pos):
    start_pos = None
    end_pos = None
    for pos in match_pos:
        if pos[0] <= i < pos[1]:
            start_pos = pos[0]
            end_pos = pos[1]
            break

    return start_pos, end_pos


def get_parser():
    parser = argparse.ArgumentParser(
        description='convert raw text to tokenized text',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--nchar', '-n', default=1, type=int,
                        help='number of characters to split, i.e., \
                        aabb -> a a b b with -n 1 and aa bb with -n 2')
    parser.add_argument('--skip-ncols', '-s', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('--space', default='<space>', type=str,
                        help='space symbol')
    parser.add_argument('--non-lang-syms', '-l', default=None, type=str,
                        help='list of non-linguistic symobles, e.g., <NOISE> etc.')
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    parser.add_argument('--lexicon', type=str, default=False, nargs='?',
                        help='lexicon')
    parser.add_argument('--add-header', action='store_true',
                        help='')
    parser.add_argument('--trans_type', '-t', type=str, default="phn",
                        help="""Transcript type. phn. e.g., for TIMIT FADG0_SI1279 -
                        read from SI1279.PHN file -> "sil b r ih sil k s aa r er n aa l
                        sil t er n ih sil t ih v sil" """)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    #if args.non_lang_syms is not None:
    #    with codecs.open(args.non_lang_syms, 'r', encoding="utf-8") as f:
    #        nls = [x.rstrip() for x in f.readlines()]
    #        rs = [re.compile(re.escape(x)) for x in nls]

    _lexicon = dict()
    with open(args.lexicon, encoding="utf-8") as f:
        for ln in f.readlines():
            wrd, phn = ln.split('\n')[0].split('\t')
            _lexicon[wrd] = phn.split()
    unk = _lexicon.get('<UNK>')
    sil = _lexicon.get('!SIL')
    if args.text:
        f = codecs.open(args.text, encoding="utf-8")
    else:
        f = codecs.getreader("utf-8")(sys.stdin if is_python2 else sys.stdin.buffer)

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout if is_python2 else sys.stdout.buffer)
    for line in f.readlines():
        x = line.split()

        # print(' '.join(x[:args.skip_ncols]), end=" ")
        snt = list()
        snt += sil
        for lbl in x[args.skip_ncols:]:
            phn = _lexicon.get(lbl, unk)
            snt += phn + sil
        # b = ' '.join([_lexicon.get(lbl, unk) ])
        a = ' '.join(snt).lower()
        if args.add_header:
            a = f'{x[0]} {a}'
        print(a)


if __name__ == '__main__':
    main()
