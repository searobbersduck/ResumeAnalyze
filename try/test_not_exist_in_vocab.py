# !/usr/bin/env python2
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

infile = 'out/eduexpr_x.txt'
invocab = '../extsrc/ttt_vocab.txt'

def loadvocab(vocabfile, vocab):
    idx = 0
    with open(vocabfile, 'r') as f:
        for line in f.readlines():
            if not line:
                continue
            line = line.strip()
            # line = line.decode('utf8')
            ss = line.split(' ')
            if len(ss) != 2:
                continue
            idx += 1
            vocab[ss[0].decode('utf8')] = idx

vocab = {}

loadvocab(invocab, vocab)

vocab_no = {}

with open(infile, 'r') as f:
    for line in f.readlines():
        if not line:
            continue
        line = line.strip()
        if line is None:
            continue
        line = line.decode('utf8')
        for s in range(len(line)):
            if line[s] in vocab:
                continue
            else:
                if line[s] in vocab_no:
                    vocab_no[line[s]] += 1
                else:
                    vocab_no[line[s]] = 1

for key,val in vocab_no.iteritems():
    print(key+'\t'+'{}'.format(val))

print('hello world!')
