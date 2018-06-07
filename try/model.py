# /usr/bin/env python2
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division
from __future__ import  absolute_import

import torch
import torch.nn as nn

torch.manual_seed(1)

class bilstm_crf(nn.Module):
    def __init__(self, embedding, hidden_dim,
                 vocab_tag, tags_size, weights=None):
        super(bilstm_crf, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.vocab_tag = vocab_tag
        self.tags_size = tags_size
        self.char_embeddings = embedding
        self.embedding_size = embedding.size()[1]
        self.lstm = nn.LSTM(self.embedding_size,
                            hidden_dim, num_layers=1,
                            bidirectional=True)
        self.hidden2tag = nn.Linear(2*hidden_dim, self.tags_size)
        self.hidden = self.init_hidden()
        if weights is not None:
            self.load_state_dict(torch.load(weights))

    def init_hidden(self):
        return (torch.randn(2,1, self.hidden_dim),
                torch.randn(2,1,self.hidden_dim))

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.char_embeddings[sentence].view(-1, 1, self.embedding_size)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(-1, self.hidden_dim*2)
        lstm_out = self.hidden2tag(lstm_out)
        return lstm_out

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        return lstm_feats

