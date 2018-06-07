# !/usr/bin/env python2
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import math
import numpy as np
from model_tf import parse_tfrecord_function, ResumeExtractorModel
from utils import AverageMeter, read_ann_file, \
    loadvocab, gen_whole_tags, gen_dataset_item, \
    loadw2v, edu_tag, get_val_datalist, \
    switch_vocab_keyval, merge_cont_and_tag

import tensorflow as tf
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='resume extract')
    parser.add_argument('--train_file', default=None)
    parser.add_argument('--val_file', default=None)
    parser.add_argument('--test_file', default=None)
    parser.add_argument('--model', default='bilstm',
                        choices=['bilstm', 'bilstm-crf'])
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)
    parser.add_argument('--iternum', default=10000, type=int)
    parser.add_argument('--phase', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--output',default='./out')
    parser.add_argument('--char_vector_file', required=True)
    parser.add_argument('--char_vocab_file', required=True)
    parser.add_argument('--tag_vocab_file', default=None)
    parser.add_argument('--hidden_size', default=4, type=int)
    parser.add_argument('--optimizer', default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--weights', default=None)
    parser.add_argument('--tfdata', default='./tfdata')
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--batch_size', default=2)
    return parser.parse_args()

def make_feed_dict(input_c, input_d, inputs, droprate=0,
                   input_y=None, input_lr=None, lr=None):
    cx = inputs[0]
    y = inputs[1]
    if input_y is None and input_lr is None:
        return {
            input_c: cx, input_d: droprate
        }
    else:
        return {
            input_c: cx, input_d: droprate, input_y:y, input_lr: lr
        }


def load_test_dataset(sess, test_input, testDatas):
    while True:
        try:
            features = sess.run(test_input)
            testDatas.append(features)
        except:
            break

def test_eval(sess, unary_score, test_squence_length,
              transMatrix, model, testDatas):
    batch = args.batch_size
    len_test = len(testDatas)
    numbatch = int((len_test-1)/args.batch_size)
    correct_labels = 0
    total_labels = 0
    for i in range(numbatch):
        endOff = (i+1)*batch
        if endOff > len_test:
            endOff = len_test
        data = testDatas[i*batch:endOff]
        inputs = []
        inputs.append([b[0] for b in data])
        inputs.append([b[1] for b in data])
        feed_dict = make_feed_dict(model.chars_inp, model.drop_inp, inputs)
        y = inputs[1]
        length= sess.run(
            [test_squence_length], feed_dict
        )
        unary_score_val, length = sess.run(
            [unary_score, test_squence_length], feed_dict
        )
        for unary_, y_, l_ in zip(
            unary_score_val, y, length
        ):
            unary_ = unary_[:l_]
            y_ = y_[:l_]
            viterbi_s, _ = tf.contrib.crf.viterbi_decode(unary_, transMatrix)
            correct_labels += np.sum(np.equal(viterbi_s, y_))
            total_labels += l_
    accuracy = 100.0*correct_labels/float(total_labels)
    print('Accuracy: {:.3f}'.format(accuracy))
    return accuracy

def val(val_file, char_vocab, sess, unary_score, test_squence_length,
              transMatrix, model, id2char_vocab, id2tag_vocab):
    datalist = get_val_datalist(val_file, char_vocab)
    for i in range(len(datalist)):
        datalist[i] = datalist[i].astype(np.int64)
        in_data = tf.convert_to_tensor(datalist[i])
        inputs = []
        inputs.append([in_data])
        inputs.append([])
        feed_dict = make_feed_dict(model.chars_inp,
                                   model.drop_inp,
                                   inputs)
        unary_score_val, length = sess.run(
            [unary_score, test_squence_length], feed_dict
        )
        for unary_, l_ in zip(unary_score_val, length):
            unary_1 = unary_[:l_]
            viterbi_s, _ = tf.contrib.crf.viterbi_decode(unary_, transMatrix)
            res = np.zeros([len(unary_)])
            res[:l_] = np.array(viterbi_s, dtype=np.int64)
            merge_cont_and_tag(datalist[i], res,
                               id2char_vocab, id2tag_vocab, id2tag_vocab)


def test_eval1(sess, unary_score, test_squence_length,
              transMatrix, model, testDatas, id2char_vocab, id2tag_vocab):
    batch = 1
    len_test = len(testDatas)
    numbatch = len_test
    correct_labels = 0
    total_labels = 0
    for i in range(numbatch):
        endOff = (i+1)*batch
        if endOff > len_test:
            endOff = len_test
        data = testDatas[i*batch:endOff]
        inputs = []
        inputs.append([b[0] for b in data])
        inputs.append([b[1] for b in data])
        feed_dict = make_feed_dict(model.chars_inp, model.drop_inp, inputs)
        y = inputs[1]
        length= sess.run(
            [test_squence_length], feed_dict
        )
        unary_score_val, length = sess.run(
            [unary_score, test_squence_length], feed_dict
        )
        for unary_, y_, l_ in zip(
            unary_score_val, y, length
        ):
            unary_ = unary_[:l_]
            y_ = y_[:l_]
            viterbi_s, _ = tf.contrib.crf.viterbi_decode(unary_, transMatrix)
            res = np.zeros([20*40], dtype=np.int64)
            res[:l_] = viterbi_s
            res = np.reshape(res, [20,40])
            merge_cont_and_tag(data[0][0], res,
                               id2char_vocab, id2tag_vocab, id2tag_vocab)



def main(args):
    tfdatapath = args.tfdata
    train_file = os.path.join(tfdatapath, 'tfrecord.train')
    test_file = os.path.join(tfdatapath, 'tfrecord.test')
    graph = tf.Graph()
    testDatas = []
    with graph.as_default():
        datasetTrain = tf.contrib.data.TFRecordDataset(train_file)
        datasetTrain = datasetTrain.map(parse_tfrecord_function)
        datasetTrain = datasetTrain.repeat(args.iternum)
        datasetTrain = datasetTrain.shuffle(buffer_size=1024)
        datasetTrain = datasetTrain.batch(args.batch_size)
        iterator = datasetTrain.make_one_shot_iterator()
        batch_inputs = iterator.get_next()
        datasetTest = tf.contrib.data.TFRecordDataset(test_file)
        datasetTest = datasetTest.map(parse_tfrecord_function)
        iteratorTest = datasetTest.make_initializable_iterator()
        test_input = iteratorTest.get_next()

        # 1. load char vocab
        char_vocab = {}
        loadvocab(args.char_vocab_file, char_vocab)
        id2char_vocab = {}
        switch_vocab_keyval(char_vocab, id2char_vocab)
        # 2. load tag vocab
        tag_vocab = {}
        gen_whole_tags(edu_tag, tag_vocab)
        id2tag_vocab = {}
        switch_vocab_keyval(tag_vocab, id2tag_vocab)
        # 3. load char vector
        char_vector = loadw2v(args.char_vector_file)
        # 4. build model
        print('====> building resume extract model: ')
        model = ResumeExtractorModel(
            [20, 50, 30], [2,3,5],
        char_vector, char_vector.shape[1],
        len(tag_vocab)+1)
        pred, pLen = model.inference()
        loss = model.loss(pred, pLen)
        train_op = model.train(loss)
        sv = tf.train.Supervisor(graph=graph, logdir=args.log_dir)
        with sv.managed_session(master='') as sess:
            sess.run(iteratorTest.initializer)
            load_test_dataset(sess, test_input, testDatas)
            bestAcc = -float('inf')
            trackHist = 0
            steps = 100000
            for i in range(steps):
                if sv.should_stop():
                    break
                try:
                    inputs = sess.run(batch_inputs)
                    feeddict = make_feed_dict(
                        model.chars_inp,
                        model.drop_inp,
                        inputs,
                        droprate=0.5,
                        input_y = model.y_inp,
                        input_lr = model.lr_inp,
                        lr = args.lr
                    )
                    trainLoss, transMatrix, _ = sess.run(
                        [loss, model.transition_params, train_op], feeddict)
                    if (i+1)%100 == 0:
                        print('[{}]\tloss: {:.4f}'.format(i+1, trainLoss))
                    if (i+1)%1 == 0:
                        # acc = test_eval(sess,
                        #                 pred, pLen, transMatrix,
                        #                 model, testDatas)
                        test_eval1(sess, pred, pLen, transMatrix,
                            model, testDatas, id2char_vocab, id2tag_vocab)
                except Exception as e:
                    print(e)
                    continue

if __name__ == '__main__':
    args = parse_args()
    main(args)
