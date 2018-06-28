# !/usr/bin/env python2
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import math
import numpy as np

import tensorflow as tf
import argparse
from glob import glob

import json

from model_tf import parse_tfrecord_function, ResumeExtractorModel
from gen_tfrecord_try import resumes_dir, tag_file, \
    vec_chars_file, out_tfdata_dir, loadvocab, \
    switch_vocab_keyval, w2id, get_resume_list, get_tags_from_files, \
    get_tags_all, loadw2v, get_cont_from_file, \
    get_cont_except_workexpr

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
    parser.add_argument('--tfdata', default='./tfdata2')
    parser.add_argument('--log_dir', default='./log')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--track_history', default=100)
    return parser.parse_args()

args = parse_args()


def merge_cont_and_tag(cont, tag, id2char_vocab, id2tag_vocab):
    row_cnt = len(tag)
    row_str = ''
    pre_tag = ''
    for i in range(row_cnt):
        cur_tag = tag[i]
        if cur_tag == 0:
            if pre_tag != '' and pre_tag_type != 'other':
                pre_tag_type = id2tag_vocab[pre_tag]
                row_str += '${}$'.format(pre_tag_type.replace('-b', '').replace('-i', ''))
            break
        pre_tag_type = id2tag_vocab[pre_tag] if (pre_tag != '') else ''
        cur_tag_type = id2tag_vocab[cur_tag]
        if pre_tag != '':
            if (cur_tag_type == 'other'):
                if (pre_tag == 0):
                    print('predict tag error!')
                    # return False
                    continue
            elif (cur_tag % 2 == 1) and (not ((cur_tag == pre_tag) or (cur_tag == pre_tag - 1))):
                print('predict tag error!')
                # return False
                continue
            elif (cur_tag % 2 == 0) and (pre_tag == 0):
                print('predict tag error!')
                # return False
                continue
            if (cur_tag % 2 == 0):
                if pre_tag_type != 'other':
                    row_str += '${}$'.format(pre_tag_type.replace('-b', '').replace('-i', ''))
                row_str += '${}$'.format(cur_tag_type.replace('-b', '').replace('-i', ''))
            if (cur_tag_type == 'other'):
                if pre_tag_type != 'other':
                    row_str += '${}$'.format(pre_tag_type.replace('-b', '').replace('-i', ''))
        else:
            pre_tag = cur_tag
            if (cur_tag % 2 == 1) and (cur_tag_type != 'other'):
                print('predict tag error!')
                return False
            if (cur_tag % 2 == 0):
                row_str += '${}$'.format(cur_tag_type.replace('-b', '').replace('-i', ''))
        pre_tag = cur_tag
        if cont[i] >= len(id2char_vocab) - 1:
            row_str += ' '
        else:
            row_str += id2char_vocab[cont[i]]
    print(row_str)

def cont_id2char(cont, id2char_vocab):
    row_str = ''
    for i in range(len(cont)):
        if cont[i] >= len(id2char_vocab) - 1:
            row_str += ' '
        else:
            row_str += id2char_vocab[cont[i]]
    return row_str


def extract_cont_to_json(cont, tag, id2char_vocab, id2tag_vocab, resume_json):
    row_cnt = len(tag)
    key_start = False
    cur_key = None
    # resume_json = {}
    i = 0
    while (i < len(tag)):
        cur_tag = tag[i]
        cur_tag_type = id2tag_vocab[cur_tag]
        if cur_tag == 0:
            break
        if cur_tag_type == 'other':
            i += 1
            continue
        if cur_tag % 2 == 0:
            pos_list = []
            cur_tag_b = cur_tag
            cur_tag_i = cur_tag - 1
            j = i
            tmpi = 0
            pos_list.append(i)
            for j in range(i+1, len(tag)):
                if tag[j] == cur_tag_i:
                    pos_list.append(j)
                else:
                    tmpi = j
                    break
            cur_tag_type = id2tag_vocab[cur_tag].replace('-b','').replace('-i','')
            if cur_tag_type in resume_json:
                resume_json[cur_tag_type].append(cont_id2char(cont[i:j], id2char_vocab))
            else:
                resume_json[cur_tag_type] = [cont_id2char(cont[i:j], id2char_vocab)]
            i = tmpi
        else:
            i += 1
    print('hello world')

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

def test_eval1(sess, unary_score, test_squence_length,
              transMatrix, model, testDatas, id2char_vocab, id2tag_vocab):
    batch = 1
    len_test = len(testDatas)
    numbatch = len_test
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
        unary_score_val, length = sess.run(
            [unary_score, test_squence_length], feed_dict
        )
        for unary_, y_, l_ in zip(
            unary_score_val, y, length
        ):
            unary_ = unary_[:l_]
            viterbi_s, _ = tf.contrib.crf.viterbi_decode(unary_, transMatrix)
            merge_cont_and_tag(data[0][0], viterbi_s, id2char_vocab,
                               id2tag_vocab)
            print('hello world')

def predict(sess, unary_score, test_squence_length,
              transMatrix, model, predictDir,
            id2char_vocab, id2tag_vocab, vocab_chars):
    txts = glob(os.path.join(predictDir, '*.txt'))
    testDatas = []
    for txt in txts:
        testDatas.append(get_cont_from_file(txt, vocab_chars, 10000))
    batch = 1
    len_test = len(testDatas)
    numbatch = len_test
    for i in range(numbatch):
        endOff = (i+1)*batch
        if endOff > len_test:
            endOff = len_test
        data = testDatas[i*batch:endOff]
        inputs = []
        inputs.append([b for b in data])
        inputs.append([b for b in data])
        feed_dict = make_feed_dict(model.chars_inp, model.drop_inp, inputs)
        unary_score_val, length = sess.run(
            [unary_score, test_squence_length], feed_dict
        )
        for unary_, l_ in zip(
            unary_score_val, length
        ):
            unary_ = unary_[:l_]
            viterbi_s, _ = tf.contrib.crf.viterbi_decode(unary_, transMatrix)
            merge_cont_and_tag(data[0], viterbi_s, id2char_vocab,
                               id2tag_vocab)
            print('hello world')

def predict1(sess, unary_score, test_squence_length,
              transMatrix, model, predictDir,
            id2char_vocab, id2tag_vocab, vocab_chars):
    txts = glob(os.path.join(predictDir, '*_origin.txt'))
    testDatas = []
    for txt in txts:
        testDatas.append(get_cont_except_workexpr(txt, vocab_chars, 10000))
    batch = 1
    len_test = len(testDatas)
    numbatch = len_test
    json_list = []
    for i in range(numbatch):
        endOff = (i+1)*batch
        if endOff > len_test:
            endOff = len_test
        data = testDatas[i*batch:endOff]
        inputs = []
        inputs.append([b for b in data])
        inputs.append([b for b in data])
        feed_dict = make_feed_dict(model.chars_inp, model.drop_inp, inputs)
        unary_score_val, length = sess.run(
            [unary_score, test_squence_length], feed_dict
        )
        for unary_, l_ in zip(
            unary_score_val, length
        ):
            unary_ = unary_[:l_]
            viterbi_s, _ = tf.contrib.crf.viterbi_decode(unary_, transMatrix)
            json_dict = {}
            try:
                extract_cont_to_json(data[0], viterbi_s, id2char_vocab,
                                   id2tag_vocab, json_dict)
            except:
                continue
            json_list.append(json_dict)
    out_dir = predictDir
    for i in range(len(txts)):
        basename = os.path.basename(txts[i]).split('.')[0]
        basename = basename+'_pred.json'
        r = json.dumps(json_list[i])
        with open(os.path.join(out_dir, basename), 'w') as f:
            f.write(r)
    print('hello world!')



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
        resumes_list = get_resume_list(resumes_dir)
        json_file_list = [i + '_json.txt' for i in resumes_list]
        vocab_tags = {}
        get_tags_from_files(tag_file, json_file_list, vocab_tags)
        vocab_tags_all = {}
        get_tags_all(vocab_tags, vocab_tags_all)
        vocab_id2tag = {}
        switch_vocab_keyval(vocab_tags_all, vocab_id2tag)
        # 3. load char vector
        char_vector = loadw2v(args.char_vector_file)
        # 4. build model
        print('====> building resume extract model: ')
        model = ResumeExtractorModel(
            [20, 50, 30], [2,3,5],
        char_vector, char_vector.shape[1],
        len(vocab_tags_all)+1)
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
                    # trainLoss, transMatrix, _ = sess.run(
                    #     [loss, model.transition_params, train_op], feeddict)
                    trainLoss, transMatrix = sess.run(
                        [loss, model.transition_params], feeddict)
                    if (i+1)%10 == 0:
                        print('[{}]\tloss: {:.4f}'.format(i+1, trainLoss))
                    if (i+1)%1 == 0:
                    #     acc = test_eval(sess,
                    #                     pred, pLen, transMatrix,
                    #                     model, testDatas)
                    #     if acc > bestAcc:
                    #         print('====> Current best accuracy: {:.3f}'.format(acc))
                    #         bestAcc = acc
                    #         trackHist = 0
                    #         sv.saver.save(sess, args.log_dir + '/best_model')
                    #     else:
                    #         if trackHist > args.track_history:
                    #             print('====> Alaways not better in last {} histories. '
                    #                   'Best Accuracy: {:.3f}'.format(trackHist, bestAcc))
                    #             break
                    #         else:
                    #             trackHist += 1
                        # test_eval1(sess, pred, pLen, transMatrix,
                        #     model, testDatas, id2char_vocab, vocab_id2tag)
                        # predict(sess, pred, pLen, transMatrix,
                        #            model, '/Users/higgs/beast/code/work/ResumeAnalyze/try/resume_analyze_basicinfo/predict_data',
                        #         id2char_vocab, vocab_id2tag, char_vocab)
                        predict1(sess, pred, pLen, transMatrix,
                                model,
                                '/Users/higgs/beast/code/work/ResumeAnalyze/try/resume_analyze_basicinfo/test_output',
                            id2char_vocab, vocab_id2tag, char_vocab)
                        # print('hello world!')
                except KeyboardInterrupt as e:
                    sv.saver.save(
                        sess, args.log_dir + '/model', global_step=(i + 1))
                    raise e
                except Exception as e:
                    print(e)
                    continue
        sv.saver.save(sess, args.log_dir + '/finnal-model')




if __name__ == '__main__':
    main(args)