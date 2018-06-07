# !/usr/bin/env python2
# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import os
import torch.nn as nn
import torch.optim as optim

import argparse

import time
import numpy as np

from model import bilstm_crf

from utils import AverageMeter, read_ann_file, \
    loadvocab, gen_whole_tags, gen_dataset_item, \
    loadw2v, edu_tag, get_val_datalist, \
    switch_vocab_keyval, merge_cont_and_tag

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
    return parser.parse_args()

def train(datalist, taglist, model, criterion, optimizer):
    model.train()
    data_len = len(datalist)
    indexs = [i for i in range(data_len)]
    np.random.shuffle(indexs)
    losses = AverageMeter()
    for index in indexs:
        data_in = datalist[index]
        data_in = torch.from_numpy(data_in)
        data_in = data_in.type(torch.LongTensor)
        tag_in = taglist[index]
        tag_in = torch.from_numpy(tag_in)
        tag_in = tag_in.type(torch.LongTensor)
        tag_in = tag_in.view(-1)
        o = model(data_in)
        loss = criterion(o, tag_in)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.data)
    return losses.avg

def predict(datalist, model):
    model.eval()
    data_len = len(datalist)
    losses = AverageMeter()
    o_list = []
    for index in range(data_len):
        data_in = datalist[index]
        data_in = torch.from_numpy(data_in)
        data_in = data_in.type(torch.LongTensor)
        o = model(data_in)
        o = torch.max(o,1)[1]
        o = torch.reshape(o, [20,40])
        o_list.append(o.numpy())
    return o_list

def main():
    logger = []
    args = parse_args()
    loginfo = '====> opts: '
    print(loginfo)
    print(args)
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    time_stamp = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    out_dir = os.path.join(args.output, '{}-{}-{}'.format(
        args.phase, time_stamp, args.model
    ))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    log_dir = os.path.join(out_dir, 'log.txt')
    # 1. gen data doc
    docs_train = read_ann_file(args.train_file) if args.train_file is not None else None
    docs_val = read_ann_file(args.val_file) if args.val_file is not None else None
    docs_test = read_ann_file(args.test_file) if args.test_file is not None else None

    # 2. load char vocab
    char_vocab = {}
    loadvocab(args.char_vocab_file, char_vocab)
    id2char_vocab = {}
    switch_vocab_keyval(char_vocab, id2char_vocab)
    # 3. load tag vocab
    tag_vocab = {}
    gen_whole_tags(edu_tag, tag_vocab)
    id2tag_vocab = {}
    switch_vocab_keyval(tag_vocab, id2tag_vocab)
    # 4. load char vector
    char_vector = loadw2v(args.char_vector_file)
    char_vector = torch.from_numpy(char_vector)
    # 5. load model
    print('====> building model: ')
    model = bilstm_crf(char_vector, args.hidden_size,
                       tag_vocab, len(tag_vocab)+1, args.weights)
    # 6. set relate params
    loss_weight = torch.FloatTensor([1, 40, 10, 40, 10, 40, 10, 40, 10, 40, 10, 5])
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    # do train/val/test
    if args.phase == 'train':
        train_datalist = []
        val_datalist = []
        train_taglist = []
        val_taglist = []
        if docs_train is not None:
            for doc in docs_train:
                cont_o_arr, tag_o_arr = \
                    gen_dataset_item(doc, 20, 40, char_vocab, tag_vocab)
                train_datalist.append(cont_o_arr)
                train_taglist.append(tag_o_arr)
        if docs_val is not None:
            for doc in docs_val:
                cont_o_arr, tag_o_arr = \
                    gen_dataset_item(doc, 20, 40, char_vocab, tag_vocab)
                val_datalist.append(cont_o_arr)
                val_taglist.append(tag_o_arr)
        best_loss = 0.3
        for i in range(args.iternum):
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.lr, momentum=args.mom,
                                        weight_decay=args.wd, nesterov=True)
            if args.optimizer == 'adam':
                print('No adam')
            loss = train(train_datalist, train_taglist,
                         model, criterion, optimizer)
            loginfo = '[{}]:\tloss {}'.format(i, loss)
            logger.append(loginfo)
            print(loginfo)
            if loss < best_loss:
                best_loss = loss
                out_model_dir = os.path.join(out_dir, 'model-{}.pth'.format(i))
                torch.save(model.cpu().state_dict(),
                           out_model_dir)
                loginfo = '====> Current best loss: {}\t\t Save model to {}'.format(
                    best_loss, out_model_dir
                )
                logger.append(loginfo)
                print(loginfo)

    elif args.phase == 'test':
        test_datalist = get_val_datalist(args.test_file, char_vocab)
        o_list = predict(test_datalist, model)
        for i in range(len(o_list)):
            merge_cont_and_tag(test_datalist[i], o_list[i],
                               id2char_vocab, id2tag_vocab, id2tag_vocab)
        print('No test')
    else:
        print('====> error phase!')


if __name__ == '__main__':
    main()


# train script
# python --train_file




