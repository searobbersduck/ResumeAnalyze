# !/usr/bin/env python2
# -*- coding:utf-8 -*-

import tensorflow as tf
MAX_SENTENCES = 20
MAX_CHARS = 40

def parse_tfrecord_function(example_proto):
    features = {
        'target': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'chars': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0)
    }
    features = tf.parse_single_example(example_proto, features)
    target = features['target']
    target.set_shape([MAX_SENTENCES*MAX_CHARS])
    chars = features['chars']
    chars.set_shape([MAX_SENTENCES*MAX_CHARS])
    chars = tf.reshape(chars, [MAX_SENTENCES, MAX_CHARS])
    return chars, target

class ResumeExtractorModel:
    def __init__(self, filterSizes, windowSizes, charEmb, charEmbSize,
                 numTags, maxChars=MAX_CHARS, max_sentences=MAX_SENTENCES,
                 lstmEmbSize=200):
        self.filter_sizes = filterSizes
        self.window_sizes = windowSizes
        self.char_emb = charEmb
        self.char_emb_size = charEmbSize
        self.max_sentences = max_sentences
        self.max_chars = maxChars
        self.num_tags = numTags
        self.lstm_emb_size = lstmEmbSize
        self.char_filters = []
        assert len(self.filter_sizes) == len(self.window_sizes)
        self.conv_out_size = 0
        for i in range(len(self.window_sizes)):
            filter_x = tf.get_variable('char_filter_{}'.format(i),
                                       [1, windowSizes[i], self.char_emb_size, self.filter_sizes[i]],
                                       initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                       regularizer=tf.contrib.layers.l2_regularizer(1e-4))
            self.char_filters.append(filter_x)
            self.conv_out_size += self.filter_sizes[i]
        self.W = tf.get_variable('weights',
                             shape=[self.lstm_emb_size*2, self.num_tags],
                             initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             regularizer=tf.contrib.layers.l2_regularizer(1e-4))
        self.b = tf.get_variable('bias',
                             shape=[self.num_tags])
        self.lr_inp = tf.placeholder(tf.float32, shape=None, name='lr')
        self.drop_inp = tf.placeholder(tf.float32, shape=None, name='droprate')
        self.chars_inp = tf.placeholder(tf.int64,
                                        shape=[None, self.max_sentences, self.max_chars],
                                        name='chars_inp')
        self.y_inp = tf.placeholder(tf.int64, shape=[None, self.max_sentences*self.max_chars],
                                    name='y_inp')

    def char_convolotion(self, vecs):
        vecs = tf.expand_dims(vecs, 1)
        res = []
        for i in range(len(self.char_filters)):
            conv = tf.nn.conv2d(vecs, self.char_filters[i],
                                [1,1,1,1], padding='SAME')
            conv = tf.nn.relu(conv)
            conv = tf.nn.dropout(conv, keep_prob=1-self.drop_inp)
            conv = tf.squeeze(conv)
            res.append(conv)
        return tf.concat(res, axis=-1)

    def do_bilstm(self, X, lengths, scope='bilstm'):
        with tf.variable_scope(scope) as scope:
            with tf.variable_scope(scope) as scope:
                bcell = tf.nn.rnn_cell.LSTMCell(
                    num_units=self.lstm_emb_size, state_is_tuple=True)
                fcell = tf.nn.rnn_cell.LSTMCell(
                    num_units=self.lstm_emb_size, state_is_tuple=True)
                bcell = tf.nn.rnn_cell.DropoutWrapper(
                    cell=bcell, output_keep_prob=1 - self.drop_inp)
                fcell = tf.nn.rnn_cell.DropoutWrapper(
                    cell=fcell, output_keep_prob=1 - self.drop_inp)
                outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    bcell,
                    fcell,
                    X,
                    sequence_length=lengths,
                    dtype=tf.float32,
                    time_major=False,
                    scope="LSTM")
        return tf.concat(outputs,2)

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=2)
        length = tf.reduce_sum(tf.sign(length), reduction_indices=1)
        length = tf.cast(length, tf.int32)*self.max_chars
        return length

    def loss(self, P, pLen):
        print(P.shape)
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, tf.cast(self.y_inp, tf.int32), pLen
        )
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.reduce_mean(-log_likelihood) + tf.reduce_sum(reg_losses)
        return loss

    def train(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_inp)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5.0)
                     for gradient in gradients]
        train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op

    def inference(self):
        charsInp = self.chars_inp
        lengths = self.length(charsInp)
        charsInp = tf.reshape(charsInp, [-1, self.max_sentences, self.max_chars])
        charsInp = tf.nn.embedding_lookup(self.char_emb, charsInp)
        charsInp = tf.reshape(charsInp, [-1, self.max_chars, self.char_emb_size])
        features = self.char_convolotion(charsInp)
        features = tf.reshape(features, [-1, self.max_sentences*self.max_chars, self.conv_out_size])
        out = self.do_bilstm(features, lengths)
        out = tf.reshape(out, shape=[-1, self.lstm_emb_size*2])
        out = tf.nn.xw_plus_b(out, self.W, self.b)
        scores = tf.reshape(out, [-1, self.max_sentences*self.max_chars, self.num_tags],
                            name='finalInference')
        return scores, lengths


def test_parse_tfrecord_function():
    from utils import AverageMeter, read_ann_file, \
        loadvocab, gen_whole_tags, gen_dataset_item, \
        loadw2v, edu_tag, get_val_datalist, \
        switch_vocab_keyval, merge_cont_and_tag
    import numpy as np
    infile = './tfdata/tfrecord.train'
    datasetTrain = tf.contrib.data.TFRecordDataset(infile)
    datasetTrain = datasetTrain.map(parse_tfrecord_function)
    iterator = datasetTrain.make_one_shot_iterator()
    batch_inputs = iterator.get_next()
    char_vocab_file = '../extsrc/ttt_vocab.txt'
    char_vocab = {}
    loadvocab(char_vocab_file, char_vocab)
    id2char_vocab = {}
    switch_vocab_keyval(char_vocab, id2char_vocab)
    # 3. load tag vocab
    tag_vocab = {}
    gen_whole_tags(edu_tag, tag_vocab)
    id2tag_vocab = {}
    switch_vocab_keyval(tag_vocab, id2tag_vocab)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            try:
                incont = batch_inputs[0]
                intag = batch_inputs[1]
                incont, intag = sess.run([incont, intag])
                intag = np.reshape(intag, [20, 40])
                merge_cont_and_tag(incont, intag,
                                   id2char_vocab, id2tag_vocab, id2tag_vocab)
            except:
                print('except!')


if __name__ == '__main__':
    test_parse_tfrecord_function()



