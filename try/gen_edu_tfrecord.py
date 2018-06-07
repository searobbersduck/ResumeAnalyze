# !/usr/bin/env python2
# -*- coding:utf-8 -*-

import tensorflow as tf
from utils import AverageMeter, read_ann_file, \
    loadvocab, gen_whole_tags, gen_dataset_item, \
    loadw2v, edu_tag, get_val_datalist, \
    switch_vocab_keyval, merge_cont_and_tag

import numpy as np

MAX_SENTENCES = 20
MAX_CHARS = 40

train_file = './data/eduexpr_x_ann.txt'
docs_train = read_ann_file(train_file)

char_vocab = {}
char_vocab_file = '../extsrc/ttt_vocab.txt'
loadvocab(char_vocab_file, char_vocab)

tag_vocab = {}
gen_whole_tags(edu_tag, tag_vocab)

def gen_tfitem(cont_o_arr, tag_o_arr):
    assert (len(cont_o_arr)) == MAX_SENTENCES * MAX_CHARS
    example = tf.train.Example(
        features = tf.train.Features(
            feature={
                'target': tf.train.Feature(
                    int64_list = tf.train.Int64List(value=tag_o_arr)
                ),
                'chars': tf.train.Feature(
                    int64_list = tf.train.Int64List(value=cont_o_arr)
                )
            }
        )
    )
    return example

import os

outdir = './tfdata'
if not os.path.isdir(outdir):
    os.makedirs(outdir)

train_writer = tf.python_io.TFRecordWriter('./tfdata/tfrecord.train')
test_writer = tf.python_io.TFRecordWriter('./tfdata/tfrecord.test')

writer = train_writer

for doc in docs_train:
    cont_o_arr, tag_o_arr = \
        gen_dataset_item(doc, 20, 40, char_vocab, tag_vocab)
    cont_o_arr = cont_o_arr.astype(np.int64).reshape([-1])
    tag_o_arr = tag_o_arr.astype(np.int64).reshape([-1])
    if np.random.random() < 0.8:
        writer = train_writer
    else:
        writer = test_writer
    record = gen_tfitem(cont_o_arr, tag_o_arr)
    writer.write(record.SerializeToString())

train_writer.close()
test_writer.close()

