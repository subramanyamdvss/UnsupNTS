# Copyright (C) 2018  Mikel Artetxe <artetxem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from undreamt import data
from undreamt.attention import GlobalAttention

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from undreamt.devices import gpu

import random
random.seed(7)
torch.manual_seed(7)
# torch.cuda.manual_seed_all(7)



class RNNAttentionDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, layers=1, dropout=0, input_feeding=True):
        super(RNNAttentionDecoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.special_embeddings = nn.Embedding(data.SPECIAL_SYMBOLS+1, embedding_size, padding_idx=0)
        self.attention = GlobalAttention(hidden_size, alignment_function='general')
        self.input_feeding = input_feeding
        self.input_size = embedding_size + hidden_size if input_feeding else embedding_size
        self.stacked_rnn = StackedGRU(self.input_size, hidden_size, layers=layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        #supposed to be a dictionary
        self.sosembeddings = nn.Embedding(3+1,embedding_size,padding_idx=0)

    def forward(self, ids, lengths, word_embeddings, hidden, context, context_mask, prev_output, generator,\
        att_embeddings=None,pass_att=False,pass_context=False,detach_encoder=False,ncontrol = None):
        if ncontrol is None:
            embeddings = word_embeddings(data.word_ids(ids)) + self.special_embeddings(data.special_ids(ids))
        else:
            embeddings = word_embeddings(data.word_ids(ids)) + self.special_embeddings(data.special_ids_nosos(ids)) + \
            self.sosembeddings(data.sos_ids(ids).div(3).mul(ncontrol))
        output = prev_output
        scores = []
        find_cosine= True if att_embeddings is not None else False
        cosineloss = Variable(gpu(torch.FloatTensor(1).fill_(0)))
        att_scores=[]
        att_contexts=[]
        for emb in embeddings.split(1):
            if self.input_feeding:
                input = torch.cat([emb.squeeze(0), output], 1)
            else:
                input = emb.squeeze(0)
            output, hidden = self.stacked_rnn(input, hidden)
            output,att_weights,weighted_context = self.attention(output, context, context_mask,pass_weights=True,pass_context=True,detach_encoder=detach_encoder)
            output = self.dropout(output)
            score = generator(output)   
            if pass_context:
                # print('weighted_context size:',weighted_context.size())
                att_contexts.append(weighted_context)
            if find_cosine:
                # print("att_weights:",att_weights.requires_grad)
                att_embeddings = att_embeddings.detach()
                # print("att_embeddings:",att_embeddings.requires_grad)
                weighted_embedd = att_weights.unsqueeze(1).bmm(att_embeddings.transpose(0,1)).squeeze(1)
                # print("score: ",score.exp())
                # print("special_embeddings: ",self.special_embeddings.weight.size())
                # print("word_embeddings: ",word_embeddings.weight.size())
                weighted_predembedd = score.exp().unsqueeze(1).matmul(torch.cat([self.special_embeddings.weight[1:],word_embeddings.weight[1:]])).squeeze(1)
                # print("weighted_predembedd: ",weighted_predembedd.size())
                att_cosine = torch.sum(F.cosine_similarity(weighted_embedd,weighted_predembedd))
                cosineloss+=att_cosine
            att_scores.append(att_weights)
            scores.append(score)

        if not pass_context:
            if not pass_att:
                if not find_cosine:
                    return torch.stack(scores), hidden, output
                else:
                    return torch.stack(scores), hidden, output, cosineloss
            else:
                att_scores = torch.stack(att_scores)
                if not find_cosine:
                    return torch.stack(scores), hidden, output, att_scores
                else:
                    return torch.stack(scores), hidden, output, cosineloss, att_scores
        else:
            att_contexts = torch.stack(att_contexts)
            # print('att_contexts size',att_contexts.size())
            if not pass_att:
                if not find_cosine:
                    return torch.stack(scores), hidden, output, att_contexts
                else:
                    return torch.stack(scores), hidden, output, cosineloss, att_contexts
            else:
                att_scores = torch.stack(att_scores)
                if not find_cosine:
                    return torch.stack(scores), hidden, output, att_scores, att_contexts
                else:
                    return torch.stack(scores), hidden, output, cosineloss, att_scores, att_contexts

    def initial_output(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False)


# Based on OpenNMT-py
class StackedGRU(nn.Module):
    def __init__(self, input_size, hidden_size, layers, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = layers
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
        h_1 = torch.stack(h_1)
        return input, h_1