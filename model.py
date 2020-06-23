# -*- coding: utf-8 -*-
"""
 @Time    : 2020/6/23 上午10:13
 @FileName: model.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class Bert4ReCO(nn.Module):
    def __init__(self, model_type):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_type)
        self.n_hidden = self.encoder.config.hidden_size
        self.prediction = nn.Linear(self.n_hidden, 1, bias=False)

    def forward(self, inputs):
        [seq, label] = inputs
        hidden = self.encoder(seq)[0]
        mask_idx = torch.eq(seq, 1)  # 1 is the index in the seq we separate each candidates.
        hidden = hidden.masked_select(mask_idx.unsqueeze(2).expand_as(hidden)).view(
            -1, 3, self.n_hidden)
        hidden = self.prediction(hidden).squeeze(-1)
        if label is None:
            return hidden.argmax(1)
        return F.cross_entropy(hidden, label)


if __name__ == '__main__':
    model = Bert4ReCO('voidful/albert_chinese_xxlarge')
