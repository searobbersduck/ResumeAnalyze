# /user/bin/env python2
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

src_file = '../extsrc/result.txt'

edu_list = []

with open(src_file, 'r') as f:
    for line in f.readlines():
        if not line:
            continue
        line = line.strip()
        if line is None:
            continue
        ss = line.split('\t')
        for s in ss:
            tss = s.split('\x01')
            if tss[0] != '5':
                continue
            edu_content = []
            tss = tss[1:]
            for ts in tss:
                edu_content.append(ts)
            if len(edu_content) > 0:
                edu_list.append(edu_content)

def __makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

outdir = './data'
__makedir(outdir)

outfile = os.path.join(outdir, 'eduexpr_x.txt')

with open(outfile, 'w') as f:
    cnt = 0
    for l in edu_list:
        for llist in l:
            f.write(llist)
            f.write('\n')
        f.write('\n\n\n\n\n\n\n')
        cnt += 1
        if cnt == 120:
            break

vocab = {}
for l in edu_list:
    for llist in l:
        ustr = llist.decode('utf8')
        for ichar in range(len(ustr)):
            if ustr[ichar] in vocab:
                vocab[ustr[ichar]] += 1
            else:
                vocab[ustr[ichar]] = 1

import operator
sorted_vocab = sorted(vocab.iteritems(), key=operator.itemgetter(1))

with open(os.path.join(outdir, 'eduexpr_vocab.txt'), 'w') as f:
    for key, val in sorted_vocab:
        f.write(key.encode('utf8'))
        f.write('\t')
        f.write('{}'.format(val))
        f.write('\n')


