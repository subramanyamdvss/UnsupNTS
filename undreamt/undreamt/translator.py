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

from undreamt import data, devices

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from undreamt.trainset import repeatnoise, dropnoise, wordordernoise, numberfiltering
import random

random.seed(7)
torch.manual_seed(7)
# torch.cuda.manual_seed_all(7)

class Translator:
    def __init__(self, encoder_embeddings, decoder_embeddings, generator, src_dictionary, trg_dictionary, encoder,
                 decoder, denoising=True, device=devices.default,repeatnoise=None,psencoder_embeddings=None):
        self.encoder_embeddings = encoder_embeddings
        self.decoder_embeddings = decoder_embeddings
        self.generator = generator
        self.src_dictionary = src_dictionary
        self.trg_dictionary = trg_dictionary
        self.encoder = encoder
        self.decoder = decoder
        self.denoising = denoising
        self.device = device
        weight = device(torch.ones(generator.output_classes()))
        weight[data.PAD] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False,reduce=False)
        self.criterionred = nn.NLLLoss(weight, size_average=False,reduce=True)
        self.repeatnoise=repeatnoise
        self.psencoder_embeddings=psencoder_embeddings
    def _train(self, mode):
        self.encoder_embeddings.train(mode)
        self.decoder_embeddings.train(mode)
        self.generator.train(mode)
        self.encoder.train(mode)
        self.decoder.train(mode)
        self.criterion.train(mode)

    def encode(self, sentences, train=False, backbool=False, verbose = False, noiseratio=0.5, pass_embedds=False, no_noise=False,word_embeddings=None,testing=False):
        self._train(train)
        if noiseratio==0.0:
            no_noise=True
        # print("SOURCE: {}".format(sentences[0])) if verbose  else print("",end='')
        if self.denoising and not no_noise and self.repeatnoise is not None:
            sentences = repeatnoise(sentences) if self.repeatnoise else dropnoise(sentences)

        if self.denoising and not no_noise:  # Add order noise
            sentences = wordordernoise(sentences,noiseratio)
        
        passsents = sentences.copy()
            
        print("SOURCEnoised: {}".format(sentences[0])) if verbose  else print("",end='')
        ids, lengths = self.src_dictionary.sentences2ids(sentences, sos=False, eos=True,testing=testing)
        # print(self.src_dictionary.ids2sentences(ids))
        
            
        varids = self.device(Variable(torch.LongTensor(ids), requires_grad=False))
        hidden = self.device(self.encoder.initial_hidden(len(sentences)))
        if not pass_embedds:
            hidden, context = self.encoder(ids=varids, lengths=lengths, word_embeddings=self.encoder_embeddings if word_embeddings is  None else word_embeddings, hidden=hidden)
            return hidden, context, lengths, passsents
        else:
            hidden, context, passembeddings = self.encoder(ids=varids, lengths=lengths, word_embeddings=self.encoder_embeddings if word_embeddings is  None else word_embeddings, hidden=hidden, pass_embedds=True)
            return hidden, context, lengths, passembeddings, passsents

    def mask(self, lengths):
        batch_size = len(lengths)
        max_length = max(lengths)
        if max_length == min(lengths):
            return None
        mask = torch.ByteTensor(batch_size, max_length).fill_(0)
        for i in range(batch_size):
            for j in range(lengths[i], max_length):
                mask[i, j] = 1
        return self.device(mask)

    def greedy(self, sentences, max_ratio=2, train=False,pass_att=False,no_noise=False,encodings=None,pass_context=False\
        ,detach_encoder=False,ncontrol=None):
        self._train(train)
        input_lengths = [len(data.tokenize(sentence)) for sentence in sentences]
        if encodings is not None:
            (hidden,context,context_lengths,sentences) = encodings
        else:
            hidden, context, context_lengths, sentences = self.encode(sentences, train,no_noise=no_noise)
        context_mask = self.mask(context_lengths)
        translations = [[] for sentence in sentences]
        translations_att = [[] for sentence in sentences]
        prev_words = len(sentences)*[data.SOS]
        pending = set(range(len(sentences)))
        output = self.device(self.decoder.initial_output(len(sentences)))
        context_list = []
        # print("SENTENCES GIVEN TO SCORE: {}".format(sentences[0]))

        while len(pending) > 0:
            # print(pending)
            var = self.device(Variable(torch.LongTensor([prev_words]), requires_grad=False))
            logprobs, hidden, output, att_scores,att_contexts = self.decoder(var, len(sentences)*[1], self.decoder_embeddings, hidden, context, context_mask, output, self.generator\
                , pass_att=True, pass_context=True,detach_encoder=detach_encoder,ncontrol=ncontrol)
            postmask = torch.ByteTensor([0 if i in pending else 1 for i in range(var.data.size()[0])]).unsqueeze(0).unsqueeze(2)
            att_contexts.masked_fill_(self.device(Variable(postmask,requires_grad=False)),0)
            context_list.append(att_contexts)
            if logprobs.size()[1]==1:
                prev_words = [logprobs.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()]
            else:
                prev_words = logprobs.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            # prev_words = 
            # print('att_scores {}'.format(att_scores.size()))
            prev_words_att = att_scores.topk(dim=2,k=2)[1].squeeze().data.cpu().numpy().tolist()
            # print("att_scores IN GREEDY FUNCTION  {} {}".format(att_scores,prev_words_att))
            for i in pending.copy():
                if prev_words[i] == data.EOS:
                    pending.discard(i)
                else:
                    translations[i].append(prev_words[i])
                    translations_att[i].append(prev_words_att[i])
                    if len(translations[i]) >= max_ratio*input_lengths[i]:
                        pending.discard(i)
        if not pass_context:
            return self.trg_dictionary.ids2sentences(translations,translations_att=translations_att,sentences=sentences,pass_att=pass_att)
        else:
            # print(translations)
            # print("simpreds",max([len(x) for x in translations]))
            return self.trg_dictionary.ids2sentences(translations,translations_att=translations_att,sentences=sentences,pass_att=pass_att), torch.cat(context_list)
    def score(self, src, trg, train=False,backbool=False,reduce=True,verbose=False,find_cosine=False,find_preds=False,pass_att=False,word_embeddings=None\
        , pssrc =None,pass_context=False,pass_encodings=False,no_noise=False,ncontrol=None,encodeonly=False,inp_encodings=None):
        self._train(train)

        # Check batch sizes
        if len(src) != len(trg):
            raise Exception('Sentence and hypothesis lengths do not match')

        # Encode
        if inp_encodings is None:
            if not find_cosine:
                hiddensrc, context, context_lengths, src = self.encode(src, train, backbool=backbool, verbose=verbose, no_noise=no_noise)
                if pass_encodings:
                    encodings = (hiddensrc, context, context_lengths, src)
                    if encodeonly:
                        return encodings
                if word_embeddings is not None or pssrc is not None:
                    hiddenpssrc, pscontext, pscontext_lengths, pssrc = self.encode(src if pssrc is None else pssrc,train,backbool=backbool,verbose=verbose\
                        , word_embeddings=word_embeddings,no_noise=True)
                else:
                    hiddenpssrc = None
                passembeddings=None
            else:
                hiddensrc, context, context_lengths, passembeddings, src = self.encode(src, train, backbool=backbool, verbose=verbose,pass_embedds=True)
                if word_embeddings is not None or pssrc is not None:
                    hiddenpssrc, pscontext, pscontext_lengths,passembeddings, pssrc = self.encode(src if pssrc is None else pssrc,train,backbool=backbool,verbose=verbose\
                        , word_embeddings=word_embeddings,no_noise=True)
                else:
                    hiddenpssrc = None
        else:
            hiddensrc, context, context_lengths, src = inp_encodings
            hiddenpssrc = None
            passembeddings = None
            encodings = inp_encodings
        # hiddentrg, contexttrg, contexttrg_lengths = self.encode(trg, train, backbool=backbool, verbose=verbose) if backbool else (None,None,None)#2xbatchsizex600
        context_mask = self.mask(context_lengths)
        # print("hiddensrc, context, context_lengths: {} {} {}".format(hiddensrc.size(),context.size(),len(context_lengths)))
        # Decode
        initial_output = self.device(self.decoder.initial_output(len(src)))
        input_ids, lengths = self.trg_dictionary.sentences2ids(trg, eos=False, sos=True)
        input_ids_var = self.device(Variable(torch.LongTensor(input_ids), requires_grad=False))
        if pass_context:
            if find_cosine:
                logprobs, hiddendec, _,cosineloss,att_contexts = self.decoder(input_ids_var, lengths, self.decoder_embeddings, hiddensrc, context, context_mask,\
                 initial_output, self.generator,att_embeddings=passembeddings,pass_context=pass_context,ncontrol=ncontrol)
            else:
                logprobs, hiddendec, _,att_contexts = self.decoder(input_ids_var, lengths, self.decoder_embeddings, hiddensrc, context, context_mask,\
                 initial_output, self.generator,att_embeddings=passembeddings,pass_context=pass_context,ncontrol=ncontrol)
                # print('att_contexts true size',att_contexts.size())
            if context_mask is not None:
                # print('you are right')
                att_contexts.masked_fill_(Variable(context_mask.transpose(0,1).unsqueeze(2),requires_grad=False),0)
        else:
            if not pass_att:
                if find_cosine:
                    logprobs, hiddendec, _,cosineloss = self.decoder(input_ids_var, lengths, self.decoder_embeddings, hiddensrc, context, context_mask,\
                     initial_output, self.generator,att_embeddings=passembeddings,ncontrol=ncontrol)
                else:
                    logprobs, hiddendec, _ = self.decoder(input_ids_var, lengths, self.decoder_embeddings, hiddensrc, context, context_mask,\
                     initial_output, self.generator,att_embeddings=passembeddings,ncontrol=ncontrol)
            else:

                if find_cosine:
                    logprobs, hiddendec, _,cosineloss,att_scores = self.decoder(input_ids_var, lengths, self.decoder_embeddings, hiddensrc, context, context_mask,\
                     initial_output, self.generator,att_embeddings=passembeddings,pass_att=pass_att,ncontrol=ncontrol)
                else:
                    logprobs, hiddendec, _,att_scores = self.decoder(input_ids_var, lengths, self.decoder_embeddings, hiddensrc, context, context_mask,\
                     initial_output, self.generator,att_embeddings=passembeddings,pass_att=pass_att,ncontrol=ncontrol)


        # Compute loss
        output_ids, lengths = self.trg_dictionary.sentences2ids(trg, eos=True, sos=False)
        output_ids_var = self.device(Variable(torch.LongTensor(output_ids), requires_grad=False))
        #dimension of logprobs is (sentencelen,batchsize,vocabsize)
        if reduce:
            #sum the losses.
            loss = self.criterionred(logprobs.view(-1, logprobs.size()[-1]), output_ids_var.view(-1))
        else:
            #the loss that will be returned will be (batchsize,sentencelen)
            loss = self.criterion(logprobs.view(logprobs.size()[1],logprobs.size()[2],logprobs.size()[0]),\
             output_ids_var.view(output_ids_var.size()[1],output_ids_var.size()[0]))
        if pass_context:
            if not pass_encodings:
                encodings=None
            if find_preds:
                return (loss,(self.greedy(src[0:2],pass_att=False,ncontrol=ncontrol),),hiddensrc,hiddenpssrc,att_contexts,encodings) 
            else:
                return (loss,hiddensrc,hiddenpssrc,att_contexts,encodings) 
        if pssrc is None:
            if find_preds:
                if not find_cosine:
                    return (loss, (self.greedy(src[0:2],pass_att=False,ncontrol=ncontrol)),hiddensrc,hiddenpssrc) if train else loss
                else:
                    return (loss, cosineloss, (self.greedy(src[0:2],pass_att=False,ncontrol=ncontrol)),hiddensrc,hiddenpssrc) if train else loss
            else:
                if not find_cosine:
                    return (loss,hiddensrc,hiddenpssrc) if train else loss
                else:
                    return (loss, cosineloss,hiddensrc,hiddenpssrc) if train else loss
        else:
            if find_preds:
                if not find_cosine:
                    return (loss, (self.greedy(src[0:2],pass_att=False,ncontrol=ncontrol)),context,pscontext) if train else loss
                else:
                    return (loss, cosineloss, (self.greedy(src[0:2],pass_att=False,ncontrol=ncontrol)),context,pscontext) if train else loss
            else:
                if not find_cosine:
                    return (loss,context,pscontext) if train else loss
                else:
                    return (loss, cosineloss,context,pscontext) if train else loss


    def beam_search(self, sentences, beam_size=12, max_ratio=2, train=False,rnk=2,noiseratio=0.5,pass_att=False,ncontrol=0):
        self._train(train)
        batch_size = len(sentences)
        input_lengths = [len(data.tokenize(sentence)) for sentence in sentences]
        hidden, context, context_lengths, sentences = self.encode(sentences, train,noiseratio=noiseratio,testing=True)
        translations = [[] for sentence in sentences]
        pending = set(range(batch_size))

        hidden = hidden.repeat(1, beam_size, 1)
        context = context.repeat(1, beam_size, 1)
        context_lengths *= beam_size
        context_mask = self.mask(context_lengths)
        ones = beam_size*batch_size*[1]
        prev_words = beam_size*batch_size*[data.SOS]
        output = self.device(self.decoder.initial_output(beam_size*batch_size))

        translation_scores = batch_size*[-float('inf')]
        hypotheses = batch_size*[(0.0, [])] + (beam_size-1)*batch_size*[(-float('inf'), [])]  # (score, translation)

        while len(pending) > 0:
            # Each iteration should update: prev_words, hidden, output
            var = self.device(Variable(torch.LongTensor([prev_words]), requires_grad=False))
            logprobs, hidden, output, att_scores = self.decoder(var, ones, self.decoder_embeddings, hidden, context, context_mask, output, self.generator,pass_att=True,ncontrol=ncontrol)
            prev_words = logprobs.max(dim=2)[1].squeeze().data.cpu().numpy().tolist()
            prev_words_att = att_scores.topk(dim=2,k=2)[1].squeeze().data.cpu().numpy().tolist()
            word_scores, words = logprobs.topk(k=beam_size+1, dim=2, sorted=False)
            word_scores = word_scores.squeeze(0).data.cpu().numpy().tolist()  # (beam_size*batch_size) * (beam_size+1)
            words = words.squeeze(0).data.cpu().numpy().tolist()

            for sentence_index in pending.copy():
                #consider a particular source for which beamsize best half translations have been extracted.
                #now next best beamsize translations should be found.
                #candidates which aren't finished will be found.
                candidates = []  # (score, index, word)
                for rank in range(beam_size):
                    index = sentence_index + rank*batch_size
                    for i in range(beam_size + 1):
                        word = words[index][i]
                        word_att = prev_words_att[index]
                        score = hypotheses[index][0] + word_scores[index][i]
                        if word != data.EOS:
                            candidates.append((score, index, word, word_att))
                        elif score > translation_scores[sentence_index]:
                            translations[sentence_index] = hypotheses[index][1] + [(word,word_att)]
                            translation_scores[sentence_index] = score
                best = []  # score, word, translation, hidden, output
                #beamsize best translations are inserted into best.
                for score, current_index, word, word_att in sorted(candidates, reverse=True)[:beam_size]:
                    translation = hypotheses[current_index][1] + [(word,word_att)]
                    best.append((score, word, word_att, translation, hidden[:, current_index, :].data, output[current_index].data))
                #update hypotheses based on best array
                for rank, (score, word, word_att, translation, h, o) in enumerate(best):
                    next_index = sentence_index + rank*batch_size
                    hypotheses[next_index] = (score, translation)
                    prev_words[next_index] = word
                    hidden[:, next_index, :] = h
                    output[next_index, :] = o
                if len(hypotheses[sentence_index][1]) >= max_ratio*input_lengths[sentence_index] or translation_scores[sentence_index] > hypotheses[sentence_index][0]:
                    pending.discard(sentence_index)
                    if len(translations[sentence_index]) == 0:
                        translations[sentence_index] = hypotheses[sentence_index][1]
                        translation_scores[sentence_index] = hypotheses[sentence_index][0]
        translations_att = [[translations[i][j][1] for j in range(len(translations[i])) ] for i in range(len(translations))]
        translations = [[translations[i][j][0] for j in range(len(translations[i])) ] for i in range(len(translations))]
        return self.trg_dictionary.ids2sentences(translations,translations_att=translations_att,sentences=sentences,pass_att=pass_att,testing=True)
