# !/usr/bin/env python2
# -*- coding:utf-8 -*-

import tensorflow as tf
MAX_LEN = 10000

def parse_tfrecord_function(example_proto):
    features = {
        'target': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
        'chars': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0)
    }
    features = tf.parse_single_example(example_proto, features)
    target = features['target']
    target.set_shape([MAX_LEN])
    chars = features['chars']
    chars.set_shape([MAX_LEN])
    return chars, target

class ResumeExtractorModel:
    def __init__(self, filterSizes, windowSizes,
                 charEmb, charEmbSize, numTags,
                 maxLen=MAX_LEN, lstmEmbSize=200):
        self.filter_sizes = filterSizes
        self.window_sizes = windowSizes
        self.char_emb = charEmb
        self.char_emb_size = charEmbSize
        self.max_len = maxLen
        self.num_tags = numTags
        self.lstm_emb_size = lstmEmbSize
        self.char_filters = []
        assert len(self.filter_sizes) == len(self.window_sizes)
        self.conv_out_size = 0
        for i in range(len(self.filter_sizes)):
            filter_x = tf.get_variable('char_filter_{}'.format(i),
                                       [1, self.window_sizes[i], self.char_emb_size, self.filter_sizes[i]],
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
                                        shape=[None, self.max_len],
                                        name='chars_inp')
        self.y_inp = tf.placeholder(tf.int64,
                                    shape=[None, self.max_len],
                                    name='y_inp')

    def char_convolution(self, vecs):
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
            fw = tf.nn.rnn_cell.LSTMCell(
                num_units=self.lstm_emb_size, state_is_tuple=True
            )
            bw = tf.nn.rnn_cell.LSTMCell(
                num_units=self.lstm_emb_size, state_is_tuple=True
            )
            fw = tf.nn.rnn_cell.DropoutWrapper(
                cell=fw, output_keep_prob=1-self.drop_inp
            )
            bw = tf.nn.rnn_cell.DropoutWrapper(
                cell=bw, output_keep_prob=1-self.drop_inp
            )
            outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                fw,
                bw,
                X,
                sequence_length=lengths,
                dtype=tf.float32,
                time_major=False,
                scope="LSTM")
        return tf.concat(outputs, 2)

    def length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        return length

    def loss(self, P, pLen):
        log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            P, tf.cast(self.y_inp, tf.int32), pLen
        )
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = tf.reduce_mean(-log_likelihood) + tf.reduce_sum(reg_losses)
        return loss

    def train(self, loss):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr_inp
        )
        gradients, vars = zip(*optimizer.compute_gradients(loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, 5.0)
                     for gradient in gradients]
        train_op = optimizer.apply_gradients(zip(gradients, vars))
        return train_op

    def inference(self):
        charsInp = self.chars_inp
        lengths = self.length(charsInp)
        charsInp = tf.reshape(charsInp, [-1, self.max_len])
        charsInp = tf.nn.embedding_lookup(self.char_emb, charsInp)
        charsInp = tf.reshape(charsInp, [-1, self.max_len, self.char_emb_size])
        features = self.char_convolution(charsInp)
        features = tf.reshape(features, [-1, self.max_len, self.conv_out_size])
        out = self.do_bilstm(features, lengths)
        out = tf.reshape(out, shape=[-1, self.lstm_emb_size*2])
        out = tf.nn.xw_plus_b(out, self.W, self.b)
        scores = tf.reshape(out, [-1, self.max_len, self.num_tags],
                            name='finalInference')
        return scores, lengths


def test_parse_tfrecord_function():
    from gen_tfrecord_try import loadvocab, \
        get_resume_list, vocab_chars_file, resumes_dir, \
        get_tags_from_files, tag_file, get_tags_all, \
        switch_vocab_keyval
    # 1. load chars vocab
    vocab_chars = {}
    loadvocab(vocab_chars_file, vocab_chars)
    vocab_id2char = {}
    switch_vocab_keyval(vocab_chars, vocab_id2char)
    # 2. get tags vocab
    resumes_list = get_resume_list(resumes_dir)
    json_file_list = [i + '_json.txt' for i in resumes_list]
    vocab_tags = {}
    get_tags_from_files(tag_file, json_file_list, vocab_tags)
    vocab_tags_all = {}
    get_tags_all(vocab_tags, vocab_tags_all)
    vocab_id2tag = {}
    switch_vocab_keyval(vocab_tags_all, vocab_id2tag)
    # 3. read
    infile = './tfdata/tfrecord.train'
    datasetTrain = tf.contrib.data.TFRecordDataset(infile)
    datasetTrain = datasetTrain.map(parse_tfrecord_function)
    iterator = datasetTrain.make_one_shot_iterator()
    batch_inputs = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            try:
                cont = batch_inputs[0]
                incont = sess.run([cont])
                outcont = ''
                for i in range(incont[0].shape[0]):
                    if incont[0][i] == 0:
                        break
                    elif (incont[0][i] >= len(vocab_id2char)):
                        outcont += ' '
                        continue
                    outcont += vocab_id2char[incont[0][i]]
                print(outcont)
            except Exception as e:
                break
                print(e)
                print('except!')

if __name__ == '__main__':
    test_parse_tfrecord_function()

