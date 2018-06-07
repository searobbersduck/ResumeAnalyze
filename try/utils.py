# !/usr/bin/env python2
# -*- coding:utf8 -*-

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np

import torch

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

def w2id(vocab, char, vocab_len):
    if char == ' ' or char == u' ':
        return vocab_len
    if char in vocab:
        return int(vocab[char])
    else:
        return vocab_len+1

def tag2id(vocab, tag, vocab_len):
    if tag in vocab:
        return vocab[tag]
    else:
        return vocab_len

# edu_tag = {
#     u'startedAt' : 1,
#     u'endAt' : 2,
#     u'school' : 3,
#     u'major' : 4,
#     u'degree' : 5,
#     u'other' : 11,
# }

edu_tag = {
    u'startedAt' : 1,
    u'endAt' : 2,
    u'school' : 3,
    u'major' : 4,
    u'degree' : 5,
    u'other' : 11,
}

def get_tags(ann_file):
    with open(ann_file, 'r') as f:
        for line in f.readlines():
            if not line:
                continue
            line = line.strip()
            line = line.decode('utf8')
            ann_flag_list = []
            ann_i = 0
            while True:
                line.find(ann_i)

            # # $tag(state1)$cont(state2)$tag(state3)$other_cont(state4)
            # tag_begin = False
            # tag_end = False
            # cont_begin = False
            # cont_end = False
            # ann_flag = 0
            # uchar_tag_list = []
            # for i in range(len(line)):
            #     if line[i] == u'$':
            #         ann_flag += 1
            #         continue
            #     if ann_flag == 1:
            #         ...
            #     elif ann_flag == 2:
            #         ...
            #     elif ann_flag == 3:
            #         ...
            #     elif ann_flag == 4:
            #         ...


def test_findall():
    str = u'$school$南京建工学院$school$ | $major$建筑工程$major$ | $degree$大专$degree$'
    ann_id = 0
    ann_id_list = []
    while True:
        id = str.find(u'$', ann_id)
        if id == -1:
            break
        ann_id = id + 1
        ann_id_list.append(id)
    assert len(ann_id_list) % 4 == 0
    tags_num = len(ann_id_list) // 4
    cont_all = u''
    tag_all = []
    tags = []
    for i in range(tags_num):
        assert str[ann_id_list[i*4]+1:ann_id_list[i*4+1]] == str[ann_id_list[i*4+2]+1:ann_id_list[i*4+3]]
        tag = str[ann_id_list[i*4]+1:ann_id_list[i*4+1]]
        cont = str[ann_id_list[i*4+1]+1:ann_id_list[i*4+2]]
        cont_all += cont
        for j in range(len(cont)):
            tag_get = ''
            if j == 0:
                tag_get = tag+u'-b'
                tag_all.append(tag_get)
            else:
                tag_get = tag+u'-i'
                tag_all.append(tag+u'-i')
            if tag_get not in tags:
                tags.append(tag_get)
        if i+1 < tags_num:
            other_cont = str[ann_id_list[i*4+3]+1:ann_id_list[(i+1)*4]]
            cont_all += other_cont
            for j in range(len(other_cont)):
                tag_all.append(u'other')
                if u'other' not in tags:
                    tags.append(u'other')
        print(u'{}\t{}'.format(tag, cont))
    print(cont_all)
    print(tag_all)
    print(tags)

def gen_whole_tags(tags_vocab, whole_tags_vocab):
    for key,val in tags_vocab.iteritems():
        if key == u'other':
            whole_tags_vocab[key] = val
            continue
        whole_tags_vocab[key+u'-b'] = val*2
        whole_tags_vocab[key + u'-i'] = val * 2 - 1

def normalize_sentences(max_sentences, max_chars, sentences, vocab):
    o_arr = np.zeros((max_sentences, max_chars), dtype=np.int32)
    for i in range(len(sentences)):
        sent = sentences[i]
        sent = sent.strip()
        if sent is None:
            continue
        sent = sent.decode('utf8')
        for j in range(len(sent)):
            id = w2id(vocab, sent[j], len(vocab))
            o_arr[i][j] = id
    return o_arr

# mat = loadw2v('../extsrc/ttt_vec.txt')

str = '成功细中取，  富贵险中求！'

def test_normalize_sentences():
    vocab_file = '../extsrc/ttt_vocab.txt'
    vec_file = '../extsrc/ttt_vec.txt'
    vocab = {}
    loadvocab(vocab_file, vocab)
    str = '成功细中取，  富贵险中求！\n云想衣裳花想容\n长风破浪会有时，   直挂云帆济沧海\n'
    max_s = 20
    max_chars = 40
    ss = str.split('\n')
    sentences = []
    for s in ss:
        if len(s) < 1:
            continue
        sentences.append(s)
    tensor = normalize_sentences(max_s, max_chars, sentences, vocab)
    print(tensor.shape)


def get_cont_and_tag_from_ann(cont):
    ann_id = 0
    ann_id_list = []
    while True:
        id = cont.find(u'$', ann_id)
        if id == -1:
            break
        ann_id = id+1
        ann_id_list.append(id)
    assert len(ann_id_list) % 4 == 0
    tags_num = len(ann_id_list) // 4
    cont_all = u''
    tag_all = []
    for i in range(tags_num):
        assert cont[ann_id_list[i*4]+1:ann_id_list[i*4+1]] == cont[ann_id_list[i*4+2]+1:ann_id_list[i*4+3]]
        tag_name = cont[ann_id_list[i*4]+1:ann_id_list[i*4+1]]
        cont_sub = cont[ann_id_list[i*4+1]+1:ann_id_list[i*4+2]]
        for j in range(len(cont_sub)):
            tag_name_loc = tag_name + u'-i'
            if j == 0:
                tag_name_loc = tag_name + u'-b'
            else:
                tag_name_loc = tag_name + u'-i'
            tag_all.append(tag_name_loc)
        cont_all += cont_sub
        if i+1 < tags_num:
            other_cont = cont[ann_id_list[i*4+3]+1:ann_id_list[(i+1)*4]]
            for j in range(len(other_cont)):
                tag_name_loc = 'other'
                tag_all.append('other')
            cont_all += other_cont
    # 改行数据没有标签时
    if tags_num == 0:
        for i in range(len(cont)):
            cont_all += cont[i]
            tag_all.append('other')
    assert len(cont_all) == len(tag_all)
    return cont_all, tag_all



'''
输入一段文字（包括很多行），输出的是归一化的字符id和一一对应的tagid
1. 需要有tagid的vocab, 每种tag对应两个'tag-b'和'tag-i'，标志这个标志的开始和其余部分
params:
sentences: unicode string list, 标注的数据，可以是多行数据，每行是list的一项， 每一项有如下表述形式： $school$南京建工学院$school$ | $major$建筑工程$major$ | $degree$大专$degree$
max_s: 每一段标注数据，最多有max_s行
max_c: 每一行最多有max_c个字符
vocab: 字符的字典
vocab_tag: 标签的字典
'''
def gen_dataset_item(sentences, max_s, max_c, vocab, vocab_tag):
    cont_o_arr = np.zeros((max_s, max_c), dtype=np.int32)
    tag_o_arr = np.zeros((max_s, max_c), dtype=np.int32)
    for i in range(len(sentences)):
        if (i+1) > max_s:
            break
        sent = sentences[i]
        if not sent:
            continue
        sent = sent.strip()
        if sent is None or sent == '':
            continue
        # sent = sent.decode('utf8')
        # 1. get cont & tag
        conts, tags = get_cont_and_tag_from_ann(sent)
        # 2. convert to id and set to arr
        # todo: 解决超过40个字符的问题，这里先一并跳过
        if len(conts) > 40:
            continue
        for j in range(len(conts)):
            charid = w2id(vocab, conts[j], len(vocab))
            tagid = tag2id(vocab_tag, tags[j], len(vocab_tag))
            cont_o_arr[i][j] = charid
            tag_o_arr[i][j] = tagid
    return cont_o_arr, tag_o_arr

def gen_val_dataset_item(sentences, max_s, max_c, vocab):
    cont_o_arr = np.zeros((max_s, max_c), dtype=np.int32)
    for i in range(len(sentences)):
        if (i+1) > max_s:
            break
        sent = sentences[i]
        if not sent:
            continue
        sent = sent.strip()
        if sent is None or sent == '':
            continue
        # sent = sent.decode('utf8')
        # 1. get cont & tag
        conts = sent
        # 2. convert to id and set to arr
        # todo: 解决超过40个字符的问题，这里先一并跳过
        if len(conts) > 40:
            continue
        for j in range(len(conts)):
            charid = w2id(vocab, conts[j], len(vocab))
            cont_o_arr[i][j] = charid
    return cont_o_arr

def merge_cont_and_tag(cont, tag, id2char_vocab, id2tag_vocab, id2tag_ori_vocab):
    # assert cont.shape == tag.shape
    row_cnt = cont.shape[0]
    column_cnt = cont.shape[1]
    for i in range(row_cnt):
        pre_tag = ''
        row_str = ''
        for j in range(column_cnt):
            cur_tag = tag[i][j]
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
                        return False
                elif (cur_tag % 2 == 1) and  (not ((cur_tag == pre_tag) or (cur_tag == pre_tag-1))):
                    print('predict tag error!')
                    return False
                elif (cur_tag % 2 == 0) and (pre_tag == 0):
                    print('predict tag error!')
                    return False
                if (cur_tag % 2 == 0):
                    if pre_tag_type != 'other':
                        row_str += '${}$'.format(pre_tag_type.replace('-b','').replace('-i',''))
                    row_str += '${}$'.format(cur_tag_type.replace('-b','').replace('-i',''))
                if (cur_tag_type == 'other'):
                    if pre_tag_type != 'other':
                        row_str += '${}$'.format(pre_tag_type.replace('-b', '').replace('-i', ''))
            else:
                pre_tag = cur_tag
                if (cur_tag % 2 == 1) and (cur_tag_type != 'other'):
                    print('predict tag error!')
                    return False
                if (cur_tag % 2 == 0):
                    row_str += '${}$'.format(cur_tag_type.replace('-b','').replace('-i',''))
            pre_tag = cur_tag
            if cont[i][j] >= len(id2char_vocab)-1:
                row_str += ' '
            else:
                row_str += id2char_vocab[cont[i][j]]
        print(row_str)


def read_ann_file(ann_file):
    is_paras_begin = True
    parags = []
    docs = []
    with open(ann_file, 'r') as f:
        for line in f.readlines():
            if not line:
                continue
            line = line.strip()
            if line is None or line == '':
                is_paras_begin = True
                if len(parags) > 0:
                    docs.append(parags)
                    parags = []
                continue
            if is_paras_begin:
                parags.append(line.decode('utf8'))
    print('docs number: {}'.format(len(docs)))
    return docs

def get_val_datalist(infile, char_vocab):
    docs = read_ann_file(infile)
    data_list = []
    for doc in docs:
        cont_o_arr = gen_val_dataset_item(doc, 20, 40, char_vocab)
        data_list.append(cont_o_arr)
    return data_list


def test_merge_cont_and_tag():
    # 1. gen sentences
    ann_file = './data/eduexpr_x_ann.txt'
    docs = read_ann_file(ann_file)
    # 2. load char vocab
    char_vocab_file = '../extsrc/ttt_vocab.txt'
    char_vocab = {}
    loadvocab(char_vocab_file,char_vocab)
    id2char_vocab = {}
    switch_vocab_keyval(char_vocab, id2char_vocab)
    # 3. load tag vocab
    tag_vocab = {}
    gen_whole_tags(edu_tag, tag_vocab)
    id2tag_vocab = {}
    switch_vocab_keyval(tag_vocab, id2tag_vocab)
    # 4. gen data item
    cont_o_arr, tag_o_arr = gen_dataset_item(docs[2], 20, 40, char_vocab, tag_vocab)
    cont_o_in_list = []
    tag_o_in_list = []
    for doc in docs:
        cont_o_arr, tag_o_arr = gen_dataset_item(doc, 20, 40, char_vocab, tag_vocab)
        cont_o_in_list.append(cont_o_arr)
        tag_o_in_list.append(tag_o_arr)
    for i in range(len(cont_o_in_list)):
        merge_cont_and_tag(cont_o_in_list[i], tag_o_in_list[i],
                        id2char_vocab, id2tag_vocab, id2tag_vocab)


def test_gen_dataset_item():
    # 1. gen sentences
    ann_file = './out/eduexpr_ann1.txt'
    docs = read_ann_file(ann_file)
    # 2. load char vocab
    char_vocab_file = '../extsrc/ttt_vocab.txt'
    char_vocab = {}
    loadvocab(char_vocab_file,char_vocab)
    # 3. load tag vocab
    tag_vocab = {}
    gen_whole_tags(edu_tag, tag_vocab)
    # 4. gen data item
    cont_o_arr, tag_o_arr = gen_dataset_item(docs[2], 20, 40, char_vocab, tag_vocab)
    cont_o_in_list = []
    tag_o_in_list = []
    for doc in docs:
        cont_o_arr, tag_o_arr = gen_dataset_item(doc, 20, 40, char_vocab, tag_vocab)
        cont_o_in_list.append(cont_o_arr)
        tag_o_in_list.append(tag_o_arr)
    # 5. load char vector
    char_vec_file = '../extsrc/ttt_vec.txt'
    embeds = loadw2v(char_vec_file)
    embeds = torch.from_numpy(embeds)
    cont_o_arr = torch.from_numpy(cont_o_arr)
    cont_o_arr = cont_o_arr.type(torch.LongTensor)
    cont_o_embeds = embeds[cont_o_arr]
    # 6. train model
    from model import bilstm_crf
    import torch.nn as nn
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    model = bilstm_crf(embeds, 4, tag_vocab, len(tag_vocab)+1)
    optimizer = torch.optim.SGD(model.parameters(), 0.001)
    model.train()
    models_dir = './out/models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    for i in range(100000):
        randinx = np.random.randint(0, len(cont_o_in_list))
        cont_o_arr = cont_o_in_list[randinx]
        tag_o_arr = tag_o_in_list[randinx]
        cont_o_arr = torch.from_numpy(cont_o_arr)
        cont_o_arr = cont_o_arr.type(torch.LongTensor)
        o = model(cont_o_arr)
        tag_o_arr = torch.from_numpy(tag_o_arr)
        tag_o_arr = tag_o_arr.type(torch.LongTensor)
        tag_o_arr = tag_o_arr.view(-1)
        loss = criterion(o, tag_o_arr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100 == 0:
            print('[{}]\t\tlosss {:.4f}'.format(i,loss.data))
        if i%10000 == 0:
            out_model_dir = os.path.join(models_dir, 'edu-{}.pkt'.format(i))
            torch.save(model.cpu().state_dict(),
                       out_model_dir)
            print('====> Save model:\t{}'.format(out_model_dir))

        # print('hello world')

    print('hello world')

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


if __name__ == '__main__':
    # test_normalize_sentences()
    # test_findall()
    # test_gen_dataset_item()
    test_merge_cont_and_tag()