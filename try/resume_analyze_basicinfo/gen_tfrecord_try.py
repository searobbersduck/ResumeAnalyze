# !/usr/bin/env python2
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from glob import glob
import json
import numpy as np
import re
import operator

# resumes_dir = '../data/resume_json'
#  resumes_dir = './data'
resumes_dir = './resume_output'
# resumes_dir = './resume_pred'
tag_file = '../data/resume_json/basic_keyword.txt'
vocab_chars_file = '../../extsrc/ttt_vocab.txt'
vec_chars_file = '../../extsrc/ttt_vec.txt'
out_tfdata_dir = './tfdata3'

MAX_LEN = 10000


def loadvocab(invoab, vocab):
    with open(invoab, 'r') as f:
        id = 0
        for line in f.readlines():
            if not line:
                continue
            line = line.strip()
            if line is None:
                continue
            ss = line.split(' ')
            vocab[ss[0].decode('utf8')] = id
            id += 1

def switch_vocab_keyval(invocab, outvocab):
    for key, value in invocab.iteritems():
        outvocab[value] = key

def w2id(vocab, char, vocab_len):
    if char == ' ' or char == u' ':
        return vocab_len
    if char in vocab:
        return int(vocab[char])
    else:
        return vocab_len+1

def loadw2v(txt):
    mat_list = []
    with open(txt, 'r') as f:
        line = f.readline()
        line = line.strip()
        ss = line.split(' ')
        char_num = int(ss[0])
        dim = int(ss[1])
        for line in f.readlines():
            arr_list = []
            line = line.strip()
            ss = line.split(' ')
            if len(ss) != (dim + 1):
                print('error')
                continue
            for s in ss[1:]:
                arr_list.append(float(s))
            if len(arr_list) != (dim):
                print('arr_list error!')
                continue
            mat_list.append(arr_list)
        # add blank
        blank_list = [1/200. for i in range(dim)]
        # add unknown
        unknown_list = [2/200. for i in range(dim)]
        mat_list.append(blank_list)
        mat_list.append(unknown_list)
    return np.asarray(mat_list, np.float32)

def get_resume_list(dir):
    resumes_list = glob(os.path.join(dir, '*_origin.txt'))
    resumes_list = [i.replace('_origin.txt', '') for i in resumes_list]
    return resumes_list

def extract_key_from_json(json_cont, vocab):
    for i in range(len(json_cont)):
        sub_para = json_cont[i]
        for key in sub_para.keys():
            sub_para_str = sub_para[key]
            sub_para_json = json.loads(sub_para_str)
            for sub_key in sub_para_json.keys():
                vocab[sub_key] = 1

# extract tag from 'tag file'&'json file', the tags should contain all tags in the two files
def get_tags_from_files(in_tag_file, in_json_file_list, vocab):
    with open(in_tag_file) as f:
        for line in f.readlines():
            if line is None:
                continue
            line = line.strip()
            if len(line) == 0:
                continue
            ss = line.split()
            if len(ss) != 2:
                continue
            vocab[ss[1].decode('utf8')] = 1
    # for json_file in in_json_file_list:
    #     with open(json_file) as f:
    #         json_cont = json.load(f)
    #         extract_key_from_json(json_cont, vocab)
    sorted_x = sorted(vocab.items(), key=operator.itemgetter(0))
    idx = 1
    for key, val in sorted_x:
        vocab[key] = idx
        idx += 1
    # vocab['xxx'] = idx

def get_tags_all(in_vocab_tag, out_vocab_tag_all):
    sorted_x = sorted(in_vocab_tag.items(), key=operator.itemgetter(1))
    for key, val in sorted_x:
        out_vocab_tag_all[key+'-b'] = 2*val
        out_vocab_tag_all[key+'-i'] = 2*val-1
    out_vocab_tag_all[u'other'] = len(in_vocab_tag)*2+1

# find all sub_str in string
def find_all_str(sub_str, str):
    pos_list = []
    str_b = 0
    str_e = len(str)
    while True:
        pos = str.find(sub_str, str_b, str_e)
        if pos == -1:
            break
        pos_list.append(pos)
        str_b = pos + 1
    return pos_list


# annotation content according to json annotation
def annotation_cont(in_cont, in_json, vocab_chars, vocab_tags, vocab_tags_all, max_len):
    cont_arr = np.zeros(max_len, dtype=np.int64)
    ann_arr = np.zeros(max_len, dtype=np.int64)
    ass_arr = np.zeros(len(in_cont), dtype=np.int64)
    for i in range(len(ass_arr)):
        ass_arr[i] = vocab_tags_all['other']
    for key in in_json.keys():
        if not key in vocab_tags:
            continue
        key_vert = in_json[key]
        if key_vert == u'':
            continue
        key_vert_list = []
        if isinstance(key_vert, list):
            key_vert_list = key_vert
        else:
            key_vert_list.append(key_vert)
        for kv in key_vert_list:
            key_b = '{}-b'.format(key)
            key_i = '{}-i'.format(key)
            key_pos = find_all_str(kv, in_cont)
            for sub_key_pos in key_pos:
                ass_arr[sub_key_pos] = vocab_tags_all[key_b]
                for i in range(1, len(kv)):
                    ass_arr[sub_key_pos+i] = vocab_tags_all[key_i]
    cont_len = len(in_cont)
    if cont_len > max_len:
        cont_len = max_len
    vocab_chars_len = len(vocab_chars)
    for i in range(cont_len):
        cont_arr[i] = w2id(vocab_chars, in_cont[i], vocab_chars_len)
    ann_arr[:cont_len] = ass_arr[:cont_len]
    return cont_arr, ann_arr


# get content and annotation
def get_cont_and_annotation(in_cont_file, in_json_file, vocab_chars, vocab_tags, vocab_tags_all, max_len):
    cont_txt = ''
    cont = None
    ann = None
    with open(in_cont_file, 'r') as f:
        json_cont_list = json.load(f)
        for json_cont in json_cont_list:
            if json_cont['label'] != 'BASIC_INFO':
                continue
            cont = json_cont['text']
    basic_info_json = None
    if cont is not None:
        with open(in_json_file, 'r') as f:
            json_ann_list = json.load(f)
            for json_ann in json_ann_list:
                for key in json_ann.keys():
                    if key != 'basic':
                        continue
                    sub_para_str = json_ann[key]
                    sub_para_json = json.loads(sub_para_str)
                    basic_info_json = sub_para_json
    return annotation_cont(cont, basic_info_json, vocab_chars,
                           vocab_tags, vocab_tags_all, max_len)



# generate tf item
def gen_tfitem(cont_arr, tag_arr):
    assert len(cont_arr) == len(tag_arr)
    example = tf.train.Example(
        features = tf.train.Features(
            feature={
                'target': tf.train.Feature(
                    int64_list = tf.train.Int64List(value=tag_arr)
                ),
                'chars': tf.train.Feature(
                    int64_list = tf.train.Int64List(value=cont_arr)
                )
            }
        )
    )
    return example


# generate tfrecord for tensorflow
def generate_tfrecord():
    if not os.path.isdir(out_tfdata_dir):
        os.makedirs(out_tfdata_dir)
    # 1. load chars vocab
    vocab_chars = {}
    loadvocab(vocab_chars_file, vocab_chars)
    # 2. get tags vocab
    resumes_list = get_resume_list(resumes_dir)
    json_file_list = [i + '_json.txt' for i in resumes_list]
    vocab_tags = {}
    get_tags_from_files(tag_file, json_file_list, vocab_tags)
    vocab_tags_all = {}
    get_tags_all(vocab_tags, vocab_tags_all)
    # 3. tf writer
    train_writer = tf.python_io.TFRecordWriter(os.path.join(out_tfdata_dir, 'tfrecord.train'))
    test_writer = tf.python_io.TFRecordWriter(os.path.join(out_tfdata_dir, 'tfrecord.test'))
    val_writer = tf.python_io.TFRecordWriter(os.path.join(out_tfdata_dir, 'tfrecord.val'))
    #
    writer = train_writer
    resumes_list = get_resume_list(resumes_dir)
    for ii in range(1):
        for resume_file in resumes_list:
            cont_file = resume_file+'_origin.txt'
            json_file = resume_file+'_json.txt'
            x1, x2 = get_cont_and_annotation(cont_file, json_file, vocab_chars,
                                             vocab_tags, vocab_tags_all, MAX_LEN)
            rand_num = np.random.random()
            if rand_num < -0.8:
                writer = train_writer
            elif rand_num < 0.9:
                writer = val_writer
            else:
                writer = test_writer
            record = gen_tfitem(x1, x2)
            writer.write(record.SerializeToString())
    train_writer.close()
    test_writer.close()
    val_writer.close()

# stat maximum length of all resumes
def stat_max_len_of_resumes(resume_dir):
    resumes_list = glob(os.path.join(resumes_dir, '*_origin_text.txt'))
    max_cnt = 0
    above_5000 = 0
    above_10000 = 0
    for resume_file in resumes_list:
        cnt = 0
        with open(resume_file) as f:
            for line in f.readlines():
                line = line.decode('utf8')
                cnt += len(line)
        if max_cnt < cnt:
            max_cnt = cnt
        if cnt > 10000:
            above_10000 += 1
    print('the max chars number is: {}'.format(max_cnt))
    print('the chars number above 10000 resume number is: {}'.format(above_10000))

def test_get_resume_list():
    resumes_list = get_resume_list(resumes_dir)
    for res in resumes_list:
        print('directory: {}\tfilename: {}'.format(
            os.path.dirname(res), os.path.basename(res)))

def test_get_tags_from_files():
    resumes_list = get_resume_list(resumes_dir)
    json_file_list = [i+'_json.txt' for i in resumes_list]
    vocab = {}
    get_tags_from_files(tag_file, json_file_list, vocab)
    print('====> out tag num: {}'.format(len(vocab)))
    print(vocab.keys())
    sorted_x = sorted(vocab.items(), key=operator.itemgetter(0))
    print(sorted_x)

def test_get_cont_and_annotation():
    # 1. load chars vocab
    vocab_chars = {}
    loadvocab(vocab_chars_file, vocab_chars)
    # 2. get tags vocab
    resumes_list = get_resume_list(resumes_dir)
    json_file_list = [i + '_json.txt' for i in resumes_list]
    vocab_tags = {}
    get_tags_from_files(tag_file, json_file_list, vocab_tags)
    vocab_tags_all = {}
    get_tags_all(vocab_tags, vocab_tags_all)
    #
    in_cont_file = os.path.join(resumes_dir, '33526.docx_origin.txt')
    in_json_file = os.path.join(resumes_dir, '33526.docx_json.txt')
    x1, x2 = get_cont_and_annotation(in_cont_file, in_json_file, vocab_chars,
                            vocab_tags, vocab_tags_all, 10000)
    print(x1)


def test_generate_tfrecord():
    generate_tfrecord()

if __name__ == '__main__':
    # test_get_resume_list()
    # test_get_tags_from_files()
    # test_get_cont_and_annotation()
    test_generate_tfrecord()
    # stat_max_len_of_resumes(resumes_dir)