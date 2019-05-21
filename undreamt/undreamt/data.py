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

import collections
import numpy as np
import torch
import torch.nn as nn
import nltk
import random
SPECIAL_SYMBOLS = 4
PAD, OOV, EOS, SOS = 0, 1, 2, 3

# random.seed(7)
# torch.manual_seed(7)
# torch.cuda.manual_seed_all(7)

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)



class Dictionary:
    def __init__(self, words):
        self.id2word = [None] + words
        self.word2id = {word: 1 + i for i, word in enumerate(words)}

    def sentence2ids(self, sentence, eos=False, sos=False,testing=False):
        tokens = tokenize(sentence)
        if not testing:
            ids = [SPECIAL_SYMBOLS + self.word2id[word] - 1 if word in self.word2id else OOV for word in tokens]
        if testing:
            tagged = nltk.pos_tag(tokens)
            # print(tagged)
            ftag = (tagged[0][1] =='NNP' or tagged[0][1] =='NNPS' )
            
            ids = [SPECIAL_SYMBOLS + self.word2id[(word.lower() if i==0 and not ftag else word )] - 1 if ((word.lower() if i==0 and not ftag else word ) in self.word2id and not hasNumbers(word)) else OOV for i,word in enumerate(tokens)]

        if eos:
            ids = ids + [EOS]
        if sos:
            ids = [SOS] + ids
        return ids

    def sentences2ids(self, sentences, eos=False, sos=False,testing=False):
        ids = [self.sentence2ids(sentence, eos=eos, sos=sos,testing=testing) for sentence in sentences]
        lengths = [len(s) for s in ids]
        ids = [s + [PAD]*(max(lengths)-len(s)) for s in ids]  # Padding
        ids = [[ids[i][j] for i in range(len(ids))] for j in range(max(lengths))]  # batch*len -> len*batch
        return ids, lengths

    def ids2sentence(self, ids,translation_att=None,sentence=None,pass_att=False,testing=False):
        if sentence is not None:
            sentence = sentence.strip().split()
            try:
                if not pass_att:
                    return ' '.join([sentence[findOOV(self,testing,sentence,translation_att[idx][0] if translation_att[idx][0]!=len(sentence) else translation_att[idx][0]-1)]\
                     if i == OOV else self.id2word[i - SPECIAL_SYMBOLS + 1] for idx,i in enumerate(ids) if i != EOS and i != PAD and i != SOS])
                else:
                    return ' '.join([sentence[findOOV(self,testing,sentence,translation_att[idx][0] if translation_att[idx][0]!=len(sentence) else translation_att[idx][0]-1)]\
                     if i == OOV else self.id2word[i - SPECIAL_SYMBOLS + 1] for idx,i in enumerate(ids) if i != EOS and i != PAD and i != SOS]),\
                    ' '.join([sentence[translation_att[idx][0] if translation_att[idx][0]!=len(sentence) else translation_att[idx][1]]\
                     for idx,i in enumerate(ids) if i != EOS and i != PAD and i != SOS])
            except IndexError:
                print(translation_att,sentence)
                exit(1)
        return ' '.join(['<OOV>' if i == OOV else self.id2word[i - SPECIAL_SYMBOLS + 1] for i in ids if i != EOS and i != PAD and i != SOS])

    def ids2sentences(self, ids,translations_att=None,sentences=None,pass_att=False,testing=False):
        if translations_att is None:
            return [self.ids2sentence(i,pass_att=pass_att,testing=testing) for i in ids]
        else:
            return [self.ids2sentence(i,translation_att=translations_att[idx],sentence=sentences[idx],pass_att=pass_att,testing=testing) for idx,i in enumerate(ids)]

    def size(self):
        return len(self.id2word) - 1


def special_ids(ids):
    return ids * (ids < SPECIAL_SYMBOLS).long()

def special_ids_nosos(ids):
    return ids * (ids < SPECIAL_SYMBOLS-1).long()
def sos_ids(ids):
    return ids * (ids == 3).long()


def word_ids(ids):
    return (ids - SPECIAL_SYMBOLS + 1) * (ids >= SPECIAL_SYMBOLS).long()

def findOOV(dic,testing, sentence, hintidx):
    ids = dic.sentence2ids(' '.join(sentence),testing=testing)
    # print(sentence,hintidx)
    windowsize=5
    for off in range(windowsize):
        low = hintidx-off
        high = hintidx+off
        if low < 0 and high >= len(sentence):
            break
        if high<len(sentence) and ids[high] == OOV:
            return high
        if low>=0 and ids[low] == OOV:
            return low
    return hintidx


class CorpusReader:
    def __init__(self, src_file, trg_file=None, max_sentence_length=80, cache_size=1000):
        self.src_file = src_file
        self.trg_file = trg_file
        self.epoch = 1
        self.pending = set()
        self.length2pending = collections.defaultdict(set)
        self.next = 0
        self.cache = []
        self.cache_size = cache_size
        self.max_sentence_length = max_sentence_length

    def _fill_cache(self):
        self.next = 0
        self.cache = [self.cache[i] for i in self.pending]
        self.pending = set()
        self.length2pending = collections.defaultdict(set)
        while len(self.cache) < self.cache_size:
            src = self.src_file.readline()
            trg = self.trg_file.readline() if self.trg_file is not None else src
            src_length = len(tokenize(src))
            trg_length = len(tokenize(trg))
            if src == '' and trg == '':
                print("this should not happen consecutively")
                self.epoch += 1
                self.src_file.seek(0)
                if self.trg_file is not None:
                    self.trg_file.seek(0)
            elif 0 < src_length <= self.max_sentence_length and 0 < trg_length <= self.max_sentence_length:
                self.cache.append(((src_length, trg_length), src.strip(), trg.strip()))
        for i in range(self.cache_size):
            self.pending.add(i)
            self.length2pending[self.cache[i][0]].add(i)

    def _remove(self, index):
        length = self.cache[index][0]
        self.pending.remove(index)
        self.length2pending[length].remove(index)

    def _score_length(self, src, trg, src_min, src_max, trg_min, trg_max):
        return max(abs(src - src_min),
                   abs(src - src_max),
                   abs(trg - trg_min),
                   abs(trg - trg_max))

    def next_batch(self, size, noop=False):
        if size > self.cache_size:
            raise ValueError('Cache size smaller than twice the batch size')

        if len(self.pending) < self.cache_size / 2:
            self._fill_cache()

        indices = [self.next]
        length = self.cache[self.next][0]
        target_length = length
        src_min = src_max = length[0]
        trg_min = trg_max = length[1]
        self._remove(self.next)
        while len(indices) < size:
            try:
                index = self.length2pending[target_length].pop()
                self.pending.remove(index)
                indices.append(index)
            except KeyError:
                candidates = [(self._score_length(k[0], k[1], src_min, src_max, trg_min, trg_max), k) for k, v in self.length2pending.items() if len(v) > 0]
                target_length = min(candidates)[1]
                src_min = min(src_min, target_length[0])
                src_max = max(src_max, target_length[0])
                trg_min = min(trg_min, target_length[1])
                trg_max = max(trg_max, target_length[1])

        indices = sorted(indices, key=lambda i: self.cache[i][0], reverse=True)

        for i in range(self.next, self.cache_size):
            if i in self.pending:
                self.next = i
                break

        return [self.cache[i][1] for i in indices], [self.cache[i][2] for i in indices]


class BacktranslatorCorpusReader:
    def __init__(self, corpus, translator,ncontrol=None):
        self.corpus = corpus
        self.translator = translator
        self.epoch = corpus.epoch
        self.ncontrol = ncontrol

    def next_batch(self, size, noop=False):
        src, trg = self.corpus.next_batch(size)
        if not noop:
            src = self.translator.greedy(trg, train=False, no_noise=False, ncontrol=self.ncontrol)
        else:
            src=trg
        # _,src = self.translator.beam_search(trg, train=False, beam_size=12, max_ratio=1.1,rnk=6)
        # print("EXCEPTION SOLVE: ",src[0])
        self.epoch = self.corpus.epoch
        return src, trg


class MganCorpusReader:
    def __init__(self, *corpuses):
        self.corpuses = corpuses
        self.epoch = corpuses[0].epoch

    def next_batch(self, size,noop=False):
        simcorpus = self.corpuses[0]
        comcorpus = self.corpuses[1]
        simsrc,_ = simcorpus.next_batch(size//4)
        comsrc,_ = comcorpus.next_batch(size//4)
        self.epoch = simcorpus.epoch
        return (simsrc,simsrc),(comsrc,comsrc)

class MulBackCorpusReader:
    def __init__(self, *corpuses,control_num=3):
        self.corpuses = corpuses
        self.epoch = corpuses[0].epoch
        self.control_num = control_num

    def next_batch(self, size,noop=False):
        src = []
        trg = []
        rem = size%self.control_num
        for idx,corpus in enumerate(self.corpuses):
            p,q= corpus.next_batch((size//self.control_num)+(rem if idx ==0 else 0),noop=noop)
            src+=p
            trg+=q
            # print(size,self.control_num,len(src),p)
        self.epoch = self.corpuses[0].epoch
        return src,trg

class MulCorpusReader:
    def __init__(self, *corpuses,control_num=3):
        self.corpuses = corpuses
        self.epoch = corpuses[0].epoch
        self.control_num = control_num

    def next_batch(self, size,noop=False):
        src = []
        rem = size%self.control_num
        for idx,corpus in enumerate(self.corpuses):
            p= corpus.next_batch((size//self.control_num)+(rem if idx ==0 else 0),noop=noop)
            src+=p[0]
            # print(size,self.control_num,len(src),p)
        self.epoch = self.corpuses[0].epoch
        return src,src


def read_embeddings(file, threshold=0, vocabulary=None,vocabonly=False):
    header = file.readline().split(' ')
    count = int(header[0]) if threshold <= 0 else min(threshold, int(header[0]))
    dim = int(header[1])
    words = []
    if not vocabonly:
        matrix = np.empty((count+1, dim)) if vocabulary is None else [np.zeros(dim)]
    for i in range(count):
        word, vec = file.readline().split(' ', 1)
        if vocabulary is None:
            words.append(word)
            if not vocabonly:
                matrix[i+1] = np.fromstring(vec, sep=' ')
        elif word in vocabulary:
            words.append(word)
            if not vocabonly:
                matrix.append(np.fromstring(vec, sep=' '))
    if vocabulary is not None:
        if not vocabonly:
            matrix = np.array(matrix)
    if not vocabonly:
        embeddings = nn.Embedding(matrix.shape[0], dim, padding_idx=0)
        embeddings.weight.data.copy_(torch.from_numpy(matrix))
        return embeddings, Dictionary(words)
    else:
        return Dictionary(words)


def random_embeddings(vocabulary_size, embedding_size):
    return nn.Embedding(vocabulary_size + 1, embedding_size)


def tokenize(sentence):
    return sentence.strip().split()