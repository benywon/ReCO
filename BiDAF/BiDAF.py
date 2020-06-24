# -*- coding: utf-8 -*-
"""
 @Time    : 2019/11/21 下午4:42
 @FileName: BiDAF.py
 @author: 王炳宁
 @contact: wangbingning@sogou-inc.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDAF(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoder_size, drop_out=0.2):
        super(BiDAF, self).__init__()
        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding(vocab_size, embedding_size)
        self.a_encoder = nn.GRU(input_size=embedding_size, hidden_size=embedding_size // 2, batch_first=True,
                                bidirectional=True)
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        # 3. Contextual Embedding Layer
        self.context_LSTM = nn.LSTM(input_size=embedding_size,
                                    hidden_size=encoder_size,
                                    bidirectional=True,
                                    batch_first=True,
                                    )

        # 4. Attention Flow Layer
        self.att_weight_c = nn.Linear(encoder_size * 2, 1)
        self.att_weight_q = nn.Linear(encoder_size * 2, 1)
        self.att_weight_cq = nn.Linear(encoder_size * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM = nn.LSTM(input_size=encoder_size * 8,
                                     hidden_size=encoder_size,
                                     bidirectional=True,
                                     batch_first=True,
                                     num_layers=2
                                     )

        self.Wq = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.drop_out = drop_out

    def forward(self, inputs):
        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)
            cq = []
            for i in range(q_len):
                # (batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                # (batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        # 2. Word Embedding Layer
        [query, passage, answer, is_train] = inputs
        c_word = self.word_emb(passage)
        q_word = self.word_emb(query)
        a_embeddings = self.word_emb(answer)
        a_embedding, _ = self.a_encoder(a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3)))
        a_score = F.softmax(self.a_attention(a_embedding), 1)
        a_output = a_score.transpose(2, 1).bmm(a_embedding).squeeze()
        a_embedding = a_output.view(a_embeddings.size(0), 3, -1)

        # Highway network
        # 3. Contextual Embedding Layer
        c, _ = self.context_LSTM(c_word)
        q, _ = self.context_LSTM(q_word)
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        m, _ = self.modeling_LSTM(g)
        # 6. Output Layer
        sj = F.softmax(self.vp(self.Wp1(m)).transpose(2, 1), 2)
        rp = sj.bmm(m)
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)), self.drop_out)
        score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)
        if not is_train:
            return score.argmax(1)
        loss = -torch.log(score[:, 0]).mean()
        return loss
