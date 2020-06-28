# -*- coding: utf-8 -*-
"""
 @Time    : 2020/6/24 下午6:18
 @FileName: model.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import warnings

import apex
import torch
import torch.nn as nn
from apex.contrib.multihead_attn import SelfMultiheadAttn
from apex.mlp import MLP
from torch.nn import functional as F

warnings.filterwarnings("ignore")

layer_norm = apex.normalization.FusedLayerNorm


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfMultiheadAttn(d_model, nhead, dropout=dropout, impl='fast')
        self.feed_forward = MLP([d_model, dim_feedforward, d_model])
        self.d_model = d_model
        self.norm1 = layer_norm(d_model)
        self.norm2 = layer_norm(d_model)

        self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        src = self.norm2(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, is_training=self.training)[0]

        src = src + src2
        src = self.norm1(src)
        src2 = self.feed_forward(src.view(-1, self.d_model)).view(src.size())
        src = src + src2
        return src


class SelfAttention(nn.Module):
    def __init__(self, n_hidden, n_layer, n_head=6):
        super().__init__()
        self.att = nn.ModuleList()
        for l in range(n_layer):
            en = TransformerEncoderLayer(n_hidden, n_head, n_hidden * 4)
            self.att.append(en)
        self.output_ln = layer_norm(n_hidden)

    def forward(self, representations):
        representations = representations.transpose(0, 1).contiguous()
        for one in self.att:
            representations = one(representations)
        return self.output_ln(representations.transpose(0, 1))


class BERTLSTM(nn.Module):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer, n_head):
        super().__init__()
        vocabulary_size = (2 + vocab_size // 8) * 8
        self.word_embedding = nn.Embedding(vocabulary_size, embedding_dim=n_embedding)
        self.encoder = nn.LSTM(input_size=n_embedding, hidden_size=n_hidden // 2, bidirectional=True, batch_first=True)
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.attention = SelfAttention(n_hidden, n_layer, n_head=n_head)
        self.output = nn.Sequential(nn.Linear(n_hidden, n_embedding),
                                    nn.LeakyReLU(inplace=True),
                                    apex.normalization.FusedLayerNorm(n_embedding))
        self.trans = nn.Linear(n_embedding, vocabulary_size, bias=False)
        self.word_embedding.weight = self.trans.weight

    def inference(self, seq):
        word_embedding = self.word_embedding(seq)
        encoder_representations, _ = self.encoder(word_embedding)
        encoder_representations = self.attention(encoder_representations)
        return encoder_representations


class BERT(BERTLSTM):
    def __init__(self, vocab_size, n_embedding, n_hidden, n_layer, n_head):
        super().__init__(vocab_size, n_embedding, n_hidden, n_layer, n_head)
        del self.trans
        del self.output
        self.prediction = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden // 2),
            nn.GELU(),
            nn.Linear(self.n_hidden // 2, 1, bias=False),
        )

    def forward(self, inputs):
        [seq, label] = inputs
        hidden = self.inference(seq)
        mask_idx = torch.eq(seq, 1)  # 1 is the index in the seq we separate each candidates.
        hidden = hidden.masked_select(mask_idx.unsqueeze(2).expand_as(hidden)).view(
            -1, 3, self.n_hidden)
        hidden = self.prediction(hidden).squeeze(-1)
        if label is None:
            return hidden.argmax(1)
        return F.cross_entropy(hidden, label)
