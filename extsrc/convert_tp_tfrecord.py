#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Project: aiparser
# File: convert_tp_tfrecord.py
# Created Date: 2018-05-09 03:41:57
# Author: Koth Chen
# Last Modified: 2018-05-09 03:41:57
# Modified By:
# Copyright (c) 2018
#

import fire
import kaka
import w2v
import random

import tensorflow as tf
from cdocument import Document, DataPack

MAX_SENTENCES = 400
MAX_TOKENS = 32
MAX_CHARS = 64


def convertPack2tf(pack):
    if len(pack.tags) != MAX_SENTENCES:
        print("pack tags:%d" % (len(pack.tags)))
        assert (False)
    assert (len(pack.chars) == MAX_CHARS * MAX_SENTENCES)
    assert (len(pack.words) == MAX_TOKENS * MAX_SENTENCES)
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "target":
                tf.train.Feature(
                    int64_list=tf.train.Int64List(value=pack.tags)),
                "chars":
                tf.train.Feature(
                    int64_list=tf.train.Int64List(value=pack.chars)),
                "words":
                tf.train.Feature(
                    int64_list=tf.train.Int64List(value=pack.words))
            }))
    return example

def doConvert2(inputPath, charVobPath, wordVocabPath, tfPath):
    seg = kaka.Tokenizer("/var/local/kakaseg/conf.json")
    fp = open(inputPath, "r")
    recordWriter = tf.python_io.TFRecordWriter(tfPath)
    cvob = w2v.Word2vecVocab()
    cvob.Load(charVobPath)

    wvob = w2v.Word2vecVocab()
    wvob.Load(wordVocabPath)
    total = 0
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        ss = line.split('\t')
        document = Document()
        for s in ss:
            tss = s.split('\x01')
            tag = int(tss[0])
            tss = tss[1:]
            document.addParagraph(tag)
            for ts in tss:
                document.addLine(ts)
        pack = DataPack()
        paras = document.makeNormParas(MAX_SENTENCES)
        if len(paras) == 0:
            continue
        if random.random() <= 0.3:
            newparas = document.augumentEx(paras)
            document.genTrain(newparas, seg, cvob, wvob, MAX_SENTENCES,
                              MAX_TOKENS, MAX_CHARS, pack)
            total += 1
      
            if total % 1000 == 0:
                print("processed %d..." % (total))
            #write pack
            record = convertPack2tf(pack)
            recordWriter.write(record.SerializeToString())
            pack.reset()

        document.genTrain(paras, seg, cvob, wvob, MAX_SENTENCES, MAX_TOKENS,
                          MAX_CHARS, pack)
        #write pack
        record = convertPack2tf(pack)
        recordWriter.write(record.SerializeToString())
        total += 1
        if total % 1000 == 0:
            print("processed %d..." % (total))
    print("totally %d docs!" % (total))

def doConvert(inputPath, charVobPath, wordVocabPath, trainTfrPath,
              testTfrPath):
    seg = kaka.Tokenizer("/var/local/kakaseg/conf.json")
    fp = open(inputPath, "r")
    writerTrain = tf.python_io.TFRecordWriter(trainTfrPath)
    writerTest = tf.python_io.TFRecordWriter(testTfrPath)
    cvob = w2v.Word2vecVocab()
    cvob.Load(charVobPath)

    wvob = w2v.Word2vecVocab()
    wvob.Load(wordVocabPath)

    testNum = 0
    total = 0
    while True:
        line = fp.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        ss = line.split('\t')
        document = Document()
        for s in ss:
            tss = s.split('\x01')
            tag = int(tss[0])
            tss = tss[1:]
            document.addParagraph(tag)
            for ts in tss:
                document.addLine(ts)
        pack = DataPack()
        forTest = False
        recordWriter = writerTrain
        if random.random() <= 0.04:
            recordWriter = writerTest
            forTest = True
        paras = document.makeNormParas(MAX_SENTENCES)
        if len(paras) == 0:
            continue
        if random.random() <= 0.3:
            newparas = document.augumentEx(paras)
            document.genTrain(newparas, seg, cvob, wvob, MAX_SENTENCES,
                              MAX_TOKENS, MAX_CHARS, pack)
            total += 1
            if forTest:
                testNum += 1

            if total % 1000 == 0:
                print("processed %d..., test=%d" % (total, testNum))
            #write pack
            record = convertPack2tf(pack)
            recordWriter.write(record.SerializeToString())
            pack.reset()

        document.genTrain(paras, seg, cvob, wvob, MAX_SENTENCES, MAX_TOKENS,
                          MAX_CHARS, pack)
        #write pack
        record = convertPack2tf(pack)
        recordWriter.write(record.SerializeToString())
        total += 1
        if forTest:
            testNum += 1
        if total % 1000 == 0:
            print("processed %d..., test=%d" % (total, testNum))
    print("totally %d docs, test=%d!" % (total, testNum))


def main():
    fire.Fire()


if __name__ == '__main__':
    main()