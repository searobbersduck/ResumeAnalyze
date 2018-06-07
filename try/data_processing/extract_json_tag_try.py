# !/usr/bin/env python2
# -*- coding:utf-8 -*-

import json

json_file = '/Users/higgs/beast/code/work/ResumeAnalyze/try/data/resume_json/33526.docx_json.txt'

json_cont = None
with open(json_file) as f:
    json_cont = json.load(f)

def extract_key_from_json(json_cont, vocab):
    for i in range(len(json_cont)):
        sub_para = json_cont[i]
        for key in sub_para.keys():
            sub_para_str = sub_para[key]
            sub_para_json = json.loads(sub_para_str)
            for sub_key in sub_para_json.keys():
                vocab[sub_key] = 1

vocab = {}

extract_key_from_json(json_cont, vocab)

print(vocab.keys())

key_file = '/Users/higgs/beast/code/work/ResumeAnalyze/try/data/resume_json/basic_keyword.txt'

with open(key_file) as f:
    for line in f.readlines():
        line = line.strip()
        ss = line.split()
        if len(ss) != 2:
            continue
        vocab[ss[1].decode('utf8')] = 1

print(vocab)

'''

'''