# !/usr/bin/env python2
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os

lines_list = []

with open('result.txt', 'r') as f:
    for line in f.readlines():
        if not line:
            continue
        line = line.strip()
        if line is None:
            continue
        ss = line.split('\t')
        for s in ss:
            tss = s.split('\x01')
            if len(tss) < 2:
                continue
            tss = tss[1:]
            for ts in tss:
                str_line = ''
                uts = ts.decode('utf8')
                for i in range(len(uts)):
                    str_line += uts[i]
                    str_line += u' '
                lines_list.append(str_line)

with open('result_seg.txt','w') as f:
    for line in lines_list:
        f.write(line.encode('utf8'))
        f.write('\n')

