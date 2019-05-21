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

from undreamt import devices
from undreamt.encoder import RNNEncoder
from undreamt.decoder import RNNAttentionDecoder
from undreamt.generator import *
from undreamt.translator import Translator
from undreamt.similarity_scorer import sentence_stats
from undreamt.discriminator import CNN
from torch.autograd import Variable
from undreamt import wordvecs 
import random

import argparse
import numpy as np
import sys
import time

random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)

def main_train():
    # Build argument parser
    parser = argparse.ArgumentParser(description='Train a neural machine translation model')

    # Training corpus
    corpora_group = parser.add_argument_group('training corpora', 'Corpora related arguments; specify either monolingual or parallel training corpora (or both)')
    corpora_group.add_argument('--src', default=None,help='the source language monolingual corpus')
    corpora_group.add_argument('--trg', default=None, help='the target language monolingual corpus')
    corpora_group.add_argument('--src2trg', default=None, metavar=('SRC', 'TRG'), nargs=2, help='the source-to-target parallel corpus')
    corpora_group.add_argument('--trg2src', default=None, metavar=('TRG', 'SRC'), nargs=2, help='the target-to-source parallel corpus')
    corpora_group.add_argument('--max_sentence_length', type=int, default=50, help='the maximum sentence length for training (defaults to 50)')
    corpora_group.add_argument('--cache', type=int, default=1000000, help='the cache size (in sentences) for corpus reading (defaults to 1000000)')
    corpora_group.add_argument('--cache_parallel', type=int, default=None, help='the cache size (in sentences) for parallel corpus reading')
    corpora_group.add_argument('--unsup',action='store_true',default=False,help='if true then supervised data is not used')
    # Embeddings/vocabulary
    embedding_group = parser.add_argument_group('embeddings', 'Embedding related arguments; either give pre-trained cross-lingual embeddings, or a vocabulary and embedding dimensionality to randomly initialize them')
    embedding_group.add_argument('--src_embeddings', help='the source language word embeddings')
    embedding_group.add_argument('--trg_embeddings', help='the target language word embeddings')
    embedding_group.add_argument('--com_embeddings', help='embeddings trained on the whole corpus')
    embedding_group.add_argument('--use_catembedds',action='store_true',default=False, help='use (common wordvecs,specific wordvecs) in embeddings')
    # embedding_group.add_argument('--use_comembedds',action='store_true',default=False, help='use common embeddings only')
    embedding_group.add_argument('--src_vocabulary', help='the source language vocabulary')
    embedding_group.add_argument('--trg_vocabulary', help='the target language vocabulary')
    embedding_group.add_argument('--embedding_size', type=int, default=0, help='the word embedding size')
    embedding_group.add_argument('--cutoff', type=int, default=0, help='cutoff vocabulary to the given size')
    embedding_group.add_argument('--learn_encoder_embeddings', action='store_true', help='learn the encoder embeddings instead of using the pre-trained ones')
    embedding_group.add_argument('--fixed_decoder_embeddings', action='store_true', help='use fixed embeddings in the decoder instead of learning them from scratch')
    embedding_group.add_argument('--fixed_generator', action='store_true', help='use fixed embeddings in the output softmax instead of learning it from scratch')
    embedding_group.add_argument('--learn_genembedds', action='store_true',default=False, help='learn the embeddings used at softmax when fixed_gen flag is True')
    embedding_group.add_argument('--learn_dec_scratch', action='store_true',default=False, help='do not initialize dec embedds before learning')
    embedding_group.add_argument('--cat_embedds', help='the source,target language word embeddings and vocabulary')
    embedding_group.add_argument('--finetune_encembedds', action='store_true',default=False, help='learn encoder embeddings after initializing')
    # Architecture
    architecture_group = parser.add_argument_group('architecture', 'Architecture related arguments')
    architecture_group.add_argument('--layers', type=int, default=2, help='the number of encoder/decoder layers (defaults to 2)')
    architecture_group.add_argument('--hidden', type=int, default=600, help='the number of dimensions for the hidden layer (defaults to 600)')
    architecture_group.add_argument('--disable_bidirectional', action='store_true', help='use a single direction encoder')
    architecture_group.add_argument('--disable_denoising', action='store_true', help='disable random swaps')
    architecture_group.add_argument('--disable_backtranslation', action='store_true', help='disable backtranslation')
    architecture_group.add_argument('--denoising_steps', type=int, default=20000,help='no of steps of denoising before backtranslation starts')
    architecture_group.add_argument('--backtranslation_steps',type=int,default=5000,help='no.of steps of backtranslation right after denoising')
    architecture_group.add_argument('--immediate_consecutive',action='store_true',default=False,help='after den+back steps immediately do the denoising and back consecutively')
    architecture_group.add_argument('--max_cosine',action='store_true',default=False,help='maximizes cosine between attended embedding and predicted embedding')
    architecture_group.add_argument('--addn_noise',action='store_true',default=False,help='repeat and drop noise added to almost every translator')
    architecture_group.add_argument('--denoi_enc_loss',action='store_true',default=False,help='while denoising encoder representations of same sentence should be same')
    architecture_group.add_argument('--enable_cross_alignment',action='store_true',default=False,help='use 2 discriminators for cross alignment')
    architecture_group.add_argument('--enable_enc_alignment',action='store_true',default=False,help='use a discriminator at encoder site')
    architecture_group.add_argument('--disable_autoencoder',action='store_true',default=False,help='disable use of autoenc Trainer')
    architecture_group.add_argument('--enable_mgan',action='store_true',default=False,help='enable training the mgan ')
    architecture_group.add_argument('--startfrom',type=int,default=1,help='what step to start from')
    # Optimization
    optimization_group = parser.add_argument_group('optimization', 'Optimization related arguments')
    optimization_group.add_argument('--batch', type=int, default=50, help='the batch size (defaults to 50)')
    optimization_group.add_argument('--learning_rate', type=float, default=0.0002, help='the global learning rate (defaults to 0.0002)')
    optimization_group.add_argument('--dropout', metavar='PROB', type=float, default=0.3, help='dropout probability for the encoder/decoder (defaults to 0.3)')
    optimization_group.add_argument('--param_init', metavar='RANGE', type=float, default=0.1, help='uniform initialization in the specified range (defaults to 0.1,  0 for module specific default initialization)')
    optimization_group.add_argument('--iterations', type=int, default=300000, help='the number of training iterations (defaults to 300000)')
    optimization_group.add_argument('--penalty_tuning', nargs=3, type=float, default=(0.5,0.5,0), help='penalty = w1*(1-docsimil)+w2*(1-treesimil)+w3')
    optimization_group.add_argument('--cosinealpha',type=float,default=1.0,help='scaling for cosine_similarity')
    # Model saving
    saving_group = parser.add_argument_group('model saving', 'Arguments for saving the trained model')
    saving_group.add_argument('--save', metavar='PREFIX', help='save models with the given prefix')
    saving_group.add_argument('--save_interval', type=int, default=0, help='save intermediate models at this interval')
    saving_group.add_argument('--load_model',default=None,help='prefix to load the model')
    # Logging/validation
    logging_group = parser.add_argument_group('logging', 'Logging and validation arguments')
    logging_group.add_argument('--log_interval', type=int, default=1000, help='log at this interval (defaults to 1000)')
    logging_group.add_argument('--validation', nargs='+', default=(), help='use parallel corpora for validation')
    logging_group.add_argument('--validation_directions', nargs='+', default=['src2src', 'trg2trg', 'src2trg', 'trg2src'], help='validation directions')
    logging_group.add_argument('--validation_output', metavar='PREFIX', help='output validation translations with the given prefix')
    logging_group.add_argument('--validation_beam_size', type=int, default=0, help='use beam search for validation')

    classifier_group = parser.add_argument_group('discriminators', 'arguments required for initialization and training of discriminators')
    classifier_group.add_argument('--embedd_dim',type=int,default=600,help='hidden embeddings will be passed to discriminator')
    classifier_group.add_argument('--class_num',type=int,default=2,help='no. of classes for the classifier')
    classifier_group.add_argument('--kernel_num',type=int,default=128, help='no. of filters used')
    classifier_group.add_argument('--kernel_sizes',type=int,nargs='+',default=(1,2,3,4,5),help='filter sizes used by classifier')
    classifier_group.add_argument('--rho',type=float,default=1,help='loss += rho*lossadv')
    classifier_group.add_argument('--disclr',type=float,default=0.0005,help='learning rate for discriminator')
    classifier_group.add_argument('--startadv',type=int,default=10000,help='start adversarial loss on discriminator after startadv steps')
    classifier_group.add_argument('--periodic',action='store_true',default=False,help='turns adversarial losses on and off periodically after every args.startadv steps')
    classifier_group.add_argument('--noclassf',action='store_true',default=False,help='if true then classifier will not be used.')
    classifier_group.add_argument('--phase',action='store_true',default=False,help='if true then training starts with adversarial loss')
    classifier_group.add_argument('--detach_encoder',action='store_true',default=False,help='if true contexts will be detached from encoder before going through adversarial loss')
    classifier_group.add_argument('--noclasssim',action='store_true',default=False,help='if true then classifier loss will be masked for simplf pipeline')
    classifier_group.add_argument('--advdenoi',action='store_true',default=False,help='use denoising even in the adversarial and classifier losses')
    classifier_group.add_argument('--singleclassf',action='store_true',default=False,help='stronger penalty as a classifier loss')
    classifier_group.add_argument('--oldclassf',action='store_true',default=False,help='use old classifier')
    # Other
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')

    # Parse arguments
    args = parser.parse_args()
    #PRINTED ARGS
    print(args)
    if args.unsup and args.load_model is None:
        args.src2trg=None
        args.trg2src=None
    # Validate arguments
    if args.src_embeddings is None and args.src_vocabulary is None or args.trg_embeddings is None and args.trg_vocabulary is None:
        print('Either an embedding or a vocabulary file must be provided')
        sys.exit(-1)
    if (args.src_embeddings is None or args.trg_embeddings is None) and (not args.learn_encoder_embeddings or args.fixed_decoder_embeddings or args.fixed_generator):
        print('Either provide pre-trained word embeddings or set to learn the encoder/decoder embeddings and generator')
        sys.exit(-1)
    if args.src_embeddings is None and args.trg_embeddings is None and args.embedding_size == 0:
        print('Either provide pre-trained word embeddings or the embedding size')
        sys.exit(-1)
    if len(args.validation) % 2 != 0:
        print('--validation should have an even number of arguments (one pair for each validation set)')
        sys.exit(-1)

    # Select device
    device = devices.gpu if args.cuda else devices.cpu

    # Create optimizer lists
    src2src_optimizers = []
    trg2trg_optimizers = []
    src2trg_optimizers = []
    trg2src_optimizers = []
    disc_optimizers = []
    enc_optimizers = []
    simdec_optimizers = []
    comdec_optimizers = []
    # Method to create a module optimizer and add it to the given lists
    def add_optimizer(module, directions=(),no_init=False,lr = None):
        if args.param_init != 0.0 and not no_init:
            for param in module.parameters():
                param.data.uniform_(-args.param_init, args.param_init)
        optimizer = torch.optim.Adam(module.parameters(), lr=args.learning_rate if lr is None else lr)
        for direction in directions:
            direction.append(optimizer)
        return optimizer

    # Load word embeddings
    src_words = trg_words = src_embeddings = trg_embeddings = src_dictionary = trg_dictionary = None
    embedding_size = args.embedding_size
    if args.src_vocabulary is not None and not args.load_model:
        f = open(args.src_vocabulary, encoding=args.encoding, errors='surrogateescape')
        src_words = [line.strip() for line in f.readlines()]
        if args.cutoff > 0:
            src_words = src_words[:args.cutoff]
        src_dictionary = data.Dictionary(src_words)
    if args.trg_vocabulary is not None and not args.load_model:
        f = open(args.trg_vocabulary, encoding=args.encoding, errors='surrogateescape')
        trg_words = [line.strip() for line in f.readlines()]
        if args.cutoff > 0:
            trg_words = trg_words[:args.cutoff]
        trg_dictionary = data.Dictionary(trg_words)

    if not args.use_catembedds:
        if args.src_embeddings is not None and not args.load_model:
            f = open(args.src_embeddings, encoding=args.encoding, errors='surrogateescape')
            src_embeddings, src_dictionary = data.read_embeddings(f, args.cutoff, src_words)
            src_embeddings = device(src_embeddings)
            #printed srcembeddings
            # print("srcembeddings {}".format(src_embeddings.size()))
            src_embeddings.weight.requires_grad = False
            if embedding_size == 0:
                embedding_size = src_embeddings.weight.data.size()[1]
            if embedding_size != src_embeddings.weight.data.size()[1]:
                print('Embedding sizes do not match')
                sys.exit(-1)
        if args.trg_embeddings is not None and not args.load_model:
            # print('you are doing nice')
            trg_file = open(args.trg_embeddings, encoding=args.encoding, errors='surrogateescape')
            trg_embeddings, trg_dictionary = data.read_embeddings(trg_file, args.cutoff, trg_words)
            trg_embeddings = device(trg_embeddings)
            # print(type(trg_embeddings))
            #printed trg_embeddings
            # print("trg_embeddings {}".format(trg_embeddings.size()))
            trg_embeddings.weight.requires_grad = False
            if embedding_size == 0:
                embedding_size = trg_embeddings.weight.data.size()[1]
            if embedding_size != trg_embeddings.weight.data.size()[1]:
                print('Embedding sizes do not match')
                sys.exit(-1)
    else:
        sys.stdout.flush()
        (src_embeddings, src_dictionary),(trg_embeddings, trg_dictionary) = torch.load(args.cat_embedds)
        src_embeddings = device(src_embeddings)
        trg_embeddings = device(trg_embeddings)
        src_embeddings.weight.requires_grad = False
        if embedding_size == 0:
            embedding_size = src_embeddings.weight.data.size()[1]
        if embedding_size != src_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)
        trg_embeddings.weight.requires_grad = False
        if embedding_size == 0:
            embedding_size = trg_embeddings.weight.data.size()[1]
        if embedding_size != trg_embeddings.weight.data.size()[1]:
            print('Embedding sizes do not match')
            sys.exit(-1)

    if args.learn_encoder_embeddings and not args.load_model:
        src_encoder_embeddings = device(data.random_embeddings(src_dictionary.size(), embedding_size))
        trg_encoder_embeddings = device(data.random_embeddings(trg_dictionary.size(), embedding_size))
        if args.finetune_encembedds:
            src_encoder_embeddings.weight.data = src_embeddings.weight.data.clone()
            trg_encoder_embeddings.weight.data = trg_embeddings.weight.data.clone()
        add_optimizer(src_encoder_embeddings, (src2src_optimizers, src2trg_optimizers,enc_optimizers))
        add_optimizer(trg_encoder_embeddings, (trg2trg_optimizers, trg2src_optimizers,enc_optimizers))
    elif not args.load_model:
        src_encoder_embeddings = src_embeddings
        trg_encoder_embeddings = trg_embeddings
    if args.fixed_decoder_embeddings and not args.load_model:
        src_decoder_embeddings = src_embeddings
        trg_decoder_embeddings = trg_embeddings
        src_decoder_embeddings.weight.requires_grad=False
        trg_decoder_embeddings.weight.requires_grad=False
    elif not args.load_model:
        src_decoder_embeddings = device(data.random_embeddings(src_dictionary.size(), embedding_size))
        trg_decoder_embeddings = device(data.random_embeddings(trg_dictionary.size(), embedding_size))
        if not args.learn_dec_scratch:
            src_decoder_embeddings.weight.data = src_embeddings.weight.data.clone()
            trg_decoder_embeddings.weight.data = trg_embeddings.weight.data.clone()
        src_decoder_embeddings.weight.requires_grad=True
        trg_decoder_embeddings.weight.requires_grad=True
        add_optimizer(src_decoder_embeddings, (src2src_optimizers, trg2src_optimizers, comdec_optimizers))
        add_optimizer(trg_decoder_embeddings, (trg2trg_optimizers, src2trg_optimizers, simdec_optimizers))
    if args.fixed_generator and not args.load_model:
        src_embedding_generator = device(EmbeddingGenerator(hidden_size=args.hidden, embedding_size=embedding_size))
        trg_embedding_generator = device(EmbeddingGenerator(hidden_size=args.hidden, embedding_size=embedding_size))
        src_genembedding = device(data.random_embeddings(src_dictionary.size(), embedding_size))
        trg_genembedding = device(data.random_embeddings(trg_dictionary.size(), embedding_size))
        src_genembedding.weight.data = src_embeddings.weight.data.clone()
        trg_genembedding.weight.data = trg_embeddings.weight.data.clone()
        src_genembedding.weight.requires_grad=args.learn_genembedds
        trg_genembedding.weight.requires_grad=args.learn_genembedds        
        add_optimizer(src_embedding_generator, (src2src_optimizers, trg2src_optimizers, comdec_optimizers))
        add_optimizer(trg_embedding_generator, (trg2trg_optimizers, src2trg_optimizers, simdec_optimizers))
        src_generator = device(WrappedEmbeddingGenerator(src_embedding_generator, src_genembedding))
        trg_generator = device(WrappedEmbeddingGenerator(trg_embedding_generator, trg_genembedding))
        if args.learn_genembedds:
            add_optimizer(src_genembedding, (src2src_optimizers, trg2src_optimizers, comdec_optimizers))
            add_optimizer(trg_genembedding, (trg2trg_optimizers, src2trg_optimizers, simdec_optimizers))
    elif not args.load_model:
        src_generator = device(LinearGenerator(args.hidden, src_dictionary.size()))
        trg_generator = device(LinearGenerator(args.hidden, trg_dictionary.size()))
        add_optimizer(src_generator, (src2src_optimizers, trg2src_optimizers, comdec_optimizers))
        add_optimizer(trg_generator, (trg2trg_optimizers, src2trg_optimizers, simdec_optimizers))

    # Build encoder
    

    # Build translators
    if args.load_model is not None:
        print(args.src,args.src2trg)
        if args.src is not None:
            t = torch.load(args.load_model+'.src2src.pth')
            # Translate sentences
            src2src_denoising=not args.disable_denoising
            src2src_device=device
            src2src_encoder=device(t.encoder)
            src2src_decoder=device(t.decoder)
            src2src_encoder_embeddings=device(t.encoder_embeddings)
            src2src_decoder_embeddings=device(t.decoder_embeddings)
            src2src_generator=device(t.generator)
            src2src_src_dictionary=t.src_dictionary
            src2src_trg_dictionary=t.trg_dictionary
            src2src_translator = Translator(src2src_encoder_embeddings,src2src_decoder_embeddings,src2src_generator,src2src_src_dictionary,\
            src2src_trg_dictionary,src2src_encoder,src2src_decoder,src2src_denoising,src2src_device,repeatnoise=True if args.addn_noise else None)
        else:
            src2src_translator = None
            
        if args.src2trg is not None:
            print(args.src2trg)
            t = torch.load(args.load_model+'.src2trg.pth')
            src2trg_denoising=not args.disable_denoising
            src2trg_device=device
            src2trg_encoder=device(t.encoder)
            src2trg_decoder=device(t.decoder)
            src2trg_encoder_embeddings=device(t.encoder_embeddings)
            src2trg_decoder_embeddings=device(t.decoder_embeddings)
            src2trg_generator=device(t.generator)
            src2trg_src_dictionary=t.src_dictionary
            src2trg_trg_dictionary=t.trg_dictionary
            # Translate sentences
            src2trg_translator = Translator(src2src_encoder_embeddings,src2trg_decoder_embeddings,src2trg_generator,src2src_src_dictionary,\
            src2trg_trg_dictionary,src2src_encoder,src2trg_decoder,src2trg_denoising,src2trg_device,repeatnoise=True if args.addn_noise else None)
            add_optimizer(src2trg_decoder, (src2trg_optimizers,trg2trg_optimizers,simdec_optimizers),no_init=True)
            add_optimizer(src2trg_decoder_embeddings, (src2trg_optimizers,trg2trg_optimizers,simdec_optimizers),no_init=True)
            add_optimizer(src2trg_generator, (src2trg_optimizers,trg2trg_optimizers,simdec_optimizers),no_init=True)
            # if args.learn_encoder_embeddings:
            #     t.encoder_embeddings.requires_grad=True
            #     add_optimizer(src_encoder_embeddings, (src2src_optimizers, src2trg_optimizers,enc_optimizers),no_init=True)
        else:
            src2trg_translator = None

        if args.trg is not None:
            # t = torch.load(args.load_model+'.trg2trg.pth')
            # t.denoising=not args.disable_denoising
            # t.device=device
            # t.encoder=device(t.encoder)
            # t.decoder=device(t.decoder)
            # t.encoder_embeddings=device(t.encoder_embeddings)
            # t.decoder_embeddings=device(t.decoder_embeddings)
            # t.generator=device(t.generator)
            # Translate sentences
            trg2trg_translator = Translator(src2src_encoder_embeddings,src2trg_decoder_embeddings,src2trg_generator,src2src_src_dictionary,\
            src2trg_trg_dictionary,src2src_encoder,src2trg_decoder,src2trg_denoising,src2trg_device,repeatnoise=True if args.addn_noise else None)

            
        else:
            trg2trg_translator = None

        if args.trg2src is not None:
            # t = torch.load(args.load_model+'.trg2src.pth')
            # t.denoising=not args.disable_denoising
            # t.device=device
            # t.encoder=device(t.encoder)
            # t.decoder=device(t.decoder)
            # t.encoder_embeddings=device(t.encoder_embeddings)
            # t.decoder_embeddings=device(t.decoder_embeddings)
            # t.generator=device(t.generator)
            # Translate sentences
            trg2src_translator = Translator(src2src_encoder_embeddings,src2src_decoder_embeddings,src2src_generator,src2src_src_dictionary,\
            src2src_trg_dictionary,src2src_encoder,src2src_decoder,src2src_denoising,src2src_device,repeatnoise=True if args.addn_noise else None)
            add_optimizer(src2src_decoder, (src2src_optimizers, trg2src_optimizers,comdec_optimizers),no_init=True)
            add_optimizer(src2src_decoder_embeddings, (src2src_optimizers,trg2src_optimizers,comdec_optimizers),no_init=True)
            add_optimizer(src2src_generator, (src2src_optimizers,trg2src_optimizers,comdec_optimizers),no_init=True)
            add_optimizer(src2src_encoder, (src2src_optimizers,src2trg_optimizers,trg2trg_optimizers,trg2src_optimizers,enc_optimizers),no_init=True)
            # if args.learn_encoder_embeddings:
            #     t.encoder_embeddings.requires_grad=True
            #     add_optimizer(t.encoder_embeddings, (trg2trg_optimizers, trg2src_optimizers,enc_optimizers),no_init=True)

        else:
            trg2src_translator =None

        if args.enable_mgan:
            mgan_disc = torch.load(args.load_model+'.mgandiscriminator.pth')
            mgan_disc = device(mgan_disc)
            enc_discriminator = None
            add_optimizer(mgan_disc, (disc_optimizers,),lr=args.disclr,no_init=True)
        else:
            mgan_disc = None
        enc_discriminator=None
        # print(src2src_translator,src2trg_translator,trg2trg_translator,trg2src_translator,mgan_disc,enc_discriminator)
        
    else:
        encoder = device(RNNEncoder(embedding_size=embedding_size, hidden_size=args.hidden,
                                bidirectional=not args.disable_bidirectional, layers=args.layers, dropout=args.dropout))
        add_optimizer(encoder, (src2src_optimizers, trg2trg_optimizers, src2trg_optimizers, trg2src_optimizers,enc_optimizers))

        # Build decoders
        src_decoder = device(RNNAttentionDecoder(embedding_size=embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
        trg_decoder = device(RNNAttentionDecoder(embedding_size=embedding_size, hidden_size=args.hidden, layers=args.layers, dropout=args.dropout))
        add_optimizer(src_decoder, (src2src_optimizers, trg2src_optimizers,comdec_optimizers))
        add_optimizer(trg_decoder, (trg2trg_optimizers, src2trg_optimizers,simdec_optimizers))

        src2src_translator = Translator(encoder_embeddings=src_encoder_embeddings,
                                        decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                        src_dictionary=src_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                        decoder=src_decoder, denoising=not args.disable_denoising, device=device,repeatnoise=False\
                                         if args.addn_noise else None, psencoder_embeddings=trg_encoder_embeddings)
        src2trg_translator = Translator(encoder_embeddings=src_encoder_embeddings,
                                        decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                        src_dictionary=src_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                        decoder=trg_decoder, denoising=not args.disable_denoising, device=device,repeatnoise=True\
                                         if args.addn_noise else None, psencoder_embeddings=trg_encoder_embeddings)
        trg2trg_translator = Translator(encoder_embeddings=trg_encoder_embeddings,
                                        decoder_embeddings=trg_decoder_embeddings, generator=trg_generator,
                                        src_dictionary=trg_dictionary, trg_dictionary=trg_dictionary, encoder=encoder,
                                        decoder=trg_decoder, denoising=not args.disable_denoising, device=device,repeatnoise=True\
                                         if args.addn_noise else None, psencoder_embeddings=src_encoder_embeddings)
        trg2src_translator = Translator(encoder_embeddings=trg_encoder_embeddings,
                                        decoder_embeddings=src_decoder_embeddings, generator=src_generator,
                                        src_dictionary=trg_dictionary, trg_dictionary=src_dictionary, encoder=encoder,
                                        decoder=src_decoder, denoising=not args.disable_denoising, device=device,repeatnoise=False\
                                         if args.addn_noise else None, psencoder_embeddings=src_encoder_embeddings)
        if args.enable_cross_alignment:
            src_discriminator = CNN(args)
            trg_discriminator = CNN(args)
        
        if args.enable_enc_alignment:
            enc_discriminator = CNN(args)
            enc_discriminator = device(enc_discriminator)
            add_optimizer(enc_discriminator, (disc_optimizers,),lr=args.disclr)
        else:
            enc_discriminator=None
        if args.enable_mgan:
            mgan_disc = CNN(args)
            mgan_disc = device(mgan_disc)
            add_optimizer(mgan_disc, (disc_optimizers,),lr=args.disclr)

    if args.unsup:
        args.src2trg=None
        args.trg2src=None
    # Build trainers
    trainers = []
    src2src_trainer = trg2trg_trainer = src2trg_trainer = trg2src_trainer = None
    srcback2trg_trainer = trgback2src_trainer = None
    if args.src is not None:
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        if not args.disable_autoencoder:
            src2src_trainer = Trainer(translator=src2src_translator, optimizers=src2src_optimizers, corpus=corpus, batch_size=args.batch)
            trainers.append(src2src_trainer)
        if not args.disable_backtranslation:
            trgback2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus, translator=src2trg_translator),\
                                           batch_size=args.batch, backbool=True, penalty_tuning=args.penalty_tuning\
                                           ,cosinealpha=args.cosinealpha)
            trainers.append(trgback2src_trainer)

    if args.trg is not None:
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        if not args.disable_autoencoder:
            trg2trg_trainer = Trainer(translator=trg2trg_translator, optimizers=trg2trg_optimizers, corpus=corpus, batch_size=args.batch)
            trainers.append(trg2trg_trainer)
        if not args.disable_backtranslation:
            srcback2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers,
                                          corpus=data.BacktranslatorCorpusReader(corpus=corpus, translator=trg2src_translator)\
                                          , batch_size=args.batch,\
                                           backbool=True, penalty_tuning=args.penalty_tuning, cosinealpha=args.cosinealpha)
            trainers.append(srcback2trg_trainer)
    if args.src2trg is not None:
        f1 = open(args.src2trg[0], encoding=args.encoding, errors='surrogateescape')
        f2 = open(args.src2trg[1], encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f1, f2, max_sentence_length=args.max_sentence_length, cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
        src2trg_trainer = Trainer(translator=src2trg_translator, optimizers=src2trg_optimizers, corpus=corpus, batch_size=args.batch)
        trainers.append(src2trg_trainer)
    if args.trg2src is not None:
        f1 = open(args.trg2src[0], encoding=args.encoding, errors='surrogateescape')
        f2 = open(args.trg2src[1], encoding=args.encoding, errors='surrogateescape')
        corpus = data.CorpusReader(f1, f2, max_sentence_length=args.max_sentence_length, cache_size=args.cache if args.cache_parallel is None else args.cache_parallel)
        trg2src_trainer = Trainer(translator=trg2src_translator, optimizers=trg2src_optimizers, corpus=corpus, batch_size=args.batch)
        trainers.append(trg2src_trainer)
    if args.src is not None and args.trg is not None and args.enable_mgan:

        #instantiate Three corpuses and pass them to combined wrapper.
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpustrg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpussrc = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        corpus = data.MganCorpusReader(corpustrg,corpussrc)
        #instantiate an Mgan Trainer
        disc_trainer = MganTrainer(src2src_translator=src2src_translator,src2trg_translator=src2trg_translator, trg2src_translator=trg2src_translator,\
         trg2trg_translator=trg2trg_translator, discriminator=mgan_disc\
        , optimizers=(disc_optimizers,enc_optimizers,simdec_optimizers,comdec_optimizers), corpus=corpus, gen_train=False)

        #instantiate Three corpuses and pass them to combined wrapper.
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpustrg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpussrc = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        corpus = data.MganCorpusReader(corpustrg,corpussrc)
        mgan_trainer = MganTrainer(src2src_translator=src2src_translator,src2trg_translator=src2trg_translator, trg2src_translator=trg2src_translator,\
         trg2trg_translator=trg2trg_translator, discriminator=mgan_disc\
        , optimizers=(disc_optimizers,enc_optimizers,simdec_optimizers,comdec_optimizers), corpus=corpus, gen_train=True)
        trainers.append(disc_trainer)
        trainers.append(mgan_trainer)
    # if args.enable_cross_alignment and args.src is not None and args.trg is not None:
    #     #initializing src Cross Trainer
    #     f = open(args.src, encoding=args.encoding, errors='surrogateescape')
    #     corpus_src = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
    #     f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
    #     corpus_trg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
    #     src_cross_trainer = CrossTrainer(rec_translator=src2src_translator,cross_translator=trg2src_translator,discriminator=src_discriminator\
    #         ,corpus_rec=corpus_src,corpus_cross=corpus_trg)
    #     #initializing trg Cross Trainer
    #     f = open(args.src, encoding=args.encoding, errors='surrogateescape')
    #     corpus_src = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
    #     f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
    #     corpus_trg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
    #     src_cross_trainer = CrossTrainer(rec_translator=trg2trg_translator,cross_translator=src2trg_translator,discriminator=trg_discriminator\
    #         ,corpus_rec=corpus_trg,corpus_cross=corpus_src)

    if args.enable_enc_alignment:
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpus_src = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpus_trg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        src_align_trainer = EncTrainer(rec_translator=src2src_translator,discriminator=enc_discriminator\
            ,corpus_rec=corpus_src,corpus_cross=corpus_trg, optimizers=(src2src_optimizers,disc_optimizers),srcbool=True)
        trainers.append(src_align_trainer)
        f = open(args.src, encoding=args.encoding, errors='surrogateescape')
        corpus_src = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        f = open(args.trg, encoding=args.encoding, errors='surrogateescape')
        corpus_trg = data.CorpusReader(f, max_sentence_length=args.max_sentence_length, cache_size=args.cache)
        trg_align_trainer = EncTrainer(rec_translator=trg2trg_translator,discriminator=enc_discriminator\
            ,corpus_rec=corpus_trg,corpus_cross=corpus_src, optimizers=(trg2trg_optimizers,disc_optimizers),srcbool=False)
        trainers.append(trg_align_trainer)




    # Build validators
    src2src_validators = []
    trg2trg_validators = []
    src2trg_validators = []
    trg2src_validators = []
    for i in range(0, len(args.validation), 2):
        src_validation = open(args.validation[i],   encoding=args.encoding, errors='surrogateescape').readlines()
        trg_validation = open(args.validation[i+1], encoding=args.encoding, errors='surrogateescape').readlines()
        if len(src_validation) != len(trg_validation):
            print('Validation sizes do not match')
            sys.exit(-1)
        map(lambda x: x.strip(), src_validation)
        map(lambda x: x.strip(), trg_validation)
        if 'src2src' in args.validation_directions:
            src2src_validators.append(Validator(src2src_translator, src_validation, src_validation, args.batch, args.validation_beam_size))
        if 'trg2trg' in args.validation_directions:
            trg2trg_validators.append(Validator(trg2trg_translator, trg_validation, trg_validation, args.batch, args.validation_beam_size))
        if 'src2trg' in args.validation_directions:
            src2trg_validators.append(Validator(src2trg_translator, src_validation, trg_validation, args.batch, args.validation_beam_size))
        if 'trg2src' in args.validation_directions:
            trg2src_validators.append(Validator(trg2src_translator, trg_validation, src_validation, args.batch, args.validation_beam_size))

    # Build loggers
    loggers = []
    src2src_output = trg2trg_output = src2trg_output = trg2src_output = None
    if args.validation_output is not None:
        src2src_output = '{0}.src2src'.format(args.validation_output)
        trg2trg_output = '{0}.trg2trg'.format(args.validation_output)
        src2trg_output = '{0}.src2trg'.format(args.validation_output)
        trg2src_output = '{0}.trg2src'.format(args.validation_output)
    loggers.append(Logger('Source to target (backtranslation)', srcback2trg_trainer, [], None, args.encoding))
    loggers.append(Logger('Target to source (backtranslation)', trgback2src_trainer, [], None, args.encoding))
    loggers.append(Logger('Source to source', src2src_trainer, src2src_validators, src2src_output, args.encoding))
    loggers.append(Logger('Target to target', trg2trg_trainer, trg2trg_validators, trg2trg_output, args.encoding))
    loggers.append(Logger('Source to target', src2trg_trainer, src2trg_validators, src2trg_output, args.encoding))
    loggers.append(Logger('Target to source', trg2src_trainer, trg2src_validators, trg2src_output, args.encoding))

    # Method to save models
    def save_models(name):
        torch.save(src2src_translator, '{0}.{1}.src2src.pth'.format(args.save, name))
        torch.save(trg2trg_translator, '{0}.{1}.trg2trg.pth'.format(args.save, name))
        torch.save(src2trg_translator, '{0}.{1}.src2trg.pth'.format(args.save, name))
        torch.save(trg2src_translator, '{0}.{1}.trg2src.pth'.format(args.save, name))
        torch.save(enc_discriminator,'{0}.{1}.encdiscriminator.pth'.format(args.save, name))
        torch.save(mgan_disc,'{0}.{1}.mgandiscriminator.pth'.format(args.save, name))
    # Training
    total = args.denoising_steps+args.backtranslation_steps
    numdenbac = args.iterations//total
    rem = args.iterations%total
    den =False
    back=False
    for i in range(numdenbac+1):
        if i==numdenbac:
            nsteps = rem
        else:
            nsteps = total
        offset = (i)*total

        for step in range(1+offset, nsteps+offset + 1):
            verbose = (step % args.log_interval == 0)
            if nsteps==rem:
                back=True
                den=True
            if step-(offset+1)<args.denoising_steps and nsteps!=rem:
                den=True
                back=False
            if step-(offset+1)>=args.denoising_steps and nsteps!=rem:
                den=False
                back=True

            if i >=1 and args.immediate_consecutive:
                den=True
                back=True

            for idx,trainer in enumerate(trainers):
                # src2src, trg2src, trg2trg, src2trg, src2trgsup, trg2srcsup
                if not args.disable_backtranslation:
                    if den and idx%2 and not back and not idx>3:
                        continue
                    if back and idx%2==0 and not den and not idx>3:
                        continue
                    if idx%2==0 and idx<=3 and verbose:
                        print('denoising STEP')
                    if idx%2 and idx<=3 and verbose:
                        print('backtranslation STEP')
                    if idx==4 and back and not den:
                        continue
                    if idx==5 and back and not den:
                        continue
                    if idx==4 and verbose:
                        print('src2trg STEP')
                    if idx==5 and verbose:
                        print('trg2src STEP')
                try:
                    trainer.step(step,args.log_interval,device,args=args)
                except Exception:
                    print('EXCEPTION OCCURED BRO !!')
                    continue
            if args.save is not None and args.save_interval > 0 and step % args.save_interval == 0 and step>args.startfrom:
                save_models('it{0}'.format(step))

            if step % args.log_interval == 0:
                print()
                print('STEP {0} x {1} no of loggers {2}'.format(step, args.batch,len(loggers)))
            #     for logger in loggers:
            #         print(logger.name,logger.trainer,logger.validators,logger.output_prefix,logger.encoding)
            #         sys.stdout.flush()
            #         logger.log(step)

            # step += 1

    save_models('final')


class Trainer:
    def __init__(self, corpus, optimizers, translator, batch_size=2,backbool=False,penalty_tuning=None,cosinealpha=None):
        self.corpus = corpus
        self.translator = translator
        self.optimizers = optimizers
        self.batch_size = batch_size
        self.backbool = backbool
        self.penalty_tuning = penalty_tuning
        self.cosinealpha = cosinealpha
        self.reset_stats()

    def step(self,nstep,log_interval,device,args=None):
        verbose = (nstep % log_interval == 0)
        backbool = self.backbool
        # Reset gradients
        for optimizer in self.optimizers:
            optimizer.zero_grad()

        # Read input sentences
        t = time.time()
        if nstep>args.startfrom:
            src, trg = self.corpus.next_batch(self.batch_size,noop=False)        
        
        if nstep<=args.startfrom:
            src, trg = self.corpus.next_batch(self.batch_size,noop=True)
            return
        print("BACKTRANSLATION TRAINING PAIR") if self.backbool and verbose else print("",end='')
        print("DENOISING TRAINING PAIR") if (not self.backbool) and verbose else print("",end='')
        print("SOURCE: {}".format(src[0])) if verbose else print("",end='')
        sys.stdout.flush()
        batchtrg_word_count = sum([len(data.tokenize(sentence)) + 1 for sentence in trg])
        self.src_word_count += sum([len(data.tokenize(sentence)) + 1 for sentence in src])  # TODO Depends on special symbols EOS/SOS
        self.trg_word_count += batchtrg_word_count  # TODO Depends on special symbols EOS/SOS
        self.io_time += time.time() - t

        # Compute loss
        t = time.time()
        #here translator should also return predictions
        if backbool:
            if not verbose:
                if not args.max_cosine:
                    loss,hiddensrc,hiddenpssrc = self.translator.score(src, trg, train=True,backbool=backbool,verbose=verbose,find_cosine=args.max_cosine)
                else:
                    loss,cosineloss,hiddensrc,hiddenpssrc = self.translator.score(src, trg, train=True,backbool=backbool,verbose=verbose,find_cosine=args.max_cosine)
            else:
                if not args.max_cosine:
                    loss,preds,hiddensrc,hiddenpssrc = self.translator.score(src, trg, train=True,backbool=backbool,verbose=verbose,find_cosine=args.max_cosine,find_preds=True)
                else:
                    loss,cosineloss,preds,hiddensrc,hiddenpssrc = self.translator.score(src, trg, train=True,backbool=backbool,verbose=verbose,find_cosine=args.max_cosine,find_preds=True)
            print("TARGET: {}".format(trg[0])) if verbose else print("",end='')
            print("PREDICTIONS FOR BACKTRANSLATION PAIR") if verbose else print("",end='')
            print(preds[0]) if verbose else print("",end='')
        else:
            psencoder_embeddings = None if not args.denoi_enc_loss else self.translator.psencoder_embeddings
            if not verbose:
                if not args.max_cosine:
                    loss,hiddensrc,hiddenpssrc = self.translator.score(src, trg, train=True,backbool=backbool,verbose=verbose,find_cosine=args.max_cosine,word_embeddings=psencoder_embeddings)
                else:
                    loss,cosineloss,hiddensrc,hiddenpssrc = self.translator.score(src, trg, train=True,backbool=backbool,verbose=verbose,find_cosine=args.max_cosine,word_embeddings=psencoder_embeddings)
            else:
                if not args.max_cosine:
                    loss,preds,hiddensrc,hiddenpssrc = self.translator.score(src, trg, train=True,backbool=backbool,verbose=verbose,find_cosine=args.max_cosine,find_preds=True,word_embeddings=psencoder_embeddings)
                else:
                    loss,cosineloss,preds,hiddensrc,hiddenpssrc = self.translator.score(src, trg, train=True,backbool=backbool,verbose=verbose,find_cosine=args.max_cosine,find_preds=True,word_embeddings=psencoder_embeddings)
            print("TARGET: {}".format(trg[0])) if verbose else print("",end='')
            print("PREDICTIONS FOR DENOISING PAIR") if verbose else print("",end='')
            print(preds[0]) if verbose else print("",end='')
        sys.stdout.flush()
        if args.max_cosine:
            print("loss,cosineloss",loss,cosineloss) if verbose else print("",end='')
            loss.add(cosineloss.mul(args.cosinealpha))
        self.forward_time += time.time() - t
        self.nsteps +=1
        # Backpropagate error + optimize
        t = time.time()
        if backbool or not args.denoi_enc_loss:
            cebatchloss = loss.data
            loss.div(self.batch_size).backward()
        else:
            
            cebatchloss = loss.data
            #now calculate batch embedding loss
            hiddensrc = torch.transpose(hiddensrc,0,1).contiguous().view(hiddensrc.size()[1],-1) #(batchsize,2*600)
            hiddenpssrc = torch.transpose(hiddenpssrc,0,1).contiguous().view(hiddenpssrc.size()[1],-1)
            # embedd loss = -(1+cos(theta))
            embeddloss = torch.nn.functional.cosine_similarity(hiddensrc,hiddenpssrc).add(1).mul(-args.cosinealpha)
            print("INSTANCE EMBEDDLOSS : {}".format(embeddloss.data[0])) if verbose else print("",end='')
            embbatchloss = embeddloss.data.sum()
            embeddloss = embeddloss.sum().div(self.batch_size)
            print("BATCH EMBEDDLOSS : {}".format(embeddloss.data)) if verbose else print("",end='')
            loss.div(self.batch_size).backward()
        sys.stdout.flush()
        self.celoss+=cebatchloss

        for optimizer in self.optimizers:
            optimizer.step()
        self.backward_time += time.time() - t

    def reset_stats(self):
        self.src_word_count = 0
        self.trg_word_count = 0
        self.io_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.celoss = 0
        self.sentstats = np.zeros(3)
        # self.embloss = 0
        self.nsteps = 0
        self.nbacksteps = 0

    def perplexity_per_word(self):
        return np.exp(self.celoss/max(self.trg_word_count,1e-8))

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / max(self.total_time(),1e-8),  self.trg_word_count / max(self.total_time(),1e-8)

    def sent_stats(self):
        return self.sentstats/(self.nbacksteps)


class EncTrainer:
    def __init__(self,rec_translator=None,discriminator=None,corpus_rec=None,corpus_cross=None,optimizers=None,srcbool=None):
        self.rec_translator = rec_translator
        self.discriminator = discriminator
        self.corpus_rec = corpus_rec
        self.corpus_cross = corpus_cross
        self.rec_optimizer,self.disc_optimizer = optimizers[0],optimizers[1]
        self.srcbool = srcbool
        self.reset_stats()

    def step(self,nstep,log_interval,device,args=None):
        verbose = (nstep % log_interval == 0)
        # print(nstep)

        # Reset gradients
        for optimizer in self.rec_optimizer:
            optimizer.zero_grad()
        for optimizer in self.disc_optimizer:
            optimizer.zero_grad()

        # Read input sentences
        t = time.time()
        src, trg = self.corpus_rec.next_batch(args.batch)
        pssrc,_ = self.corpus_cross.next_batch(args.batch)
        print("SRC RECONSTRUCTION TRAINING PAIR") if self.srcbool and verbose else print("",end='')
        print("TRG RECONSTRUCTION TRAINING PAIR") if (not self.srcbool) and verbose else print("",end='')
        print("SOURCE: {}".format(src[0])) if verbose else print("",end='')
        sys.stdout.flush()
        batchtrg_word_count = sum([len(data.tokenize(sentence)) + 1 for sentence in trg])
        self.src_word_count += sum([len(data.tokenize(sentence)) + 1 for sentence in src])  # TODO Depends on special symbols EOS/SOS
        self.trg_word_count += batchtrg_word_count  # TODO Depends on special symbols EOS/SOS
        self.io_time += time.time() - t

        # Compute loss
        t = time.time()
        #here translator should also return predictions
        #find loss, predictions and print target
        if not verbose:
            loss,contextsrc,contextpssrc = self.rec_translator.score(src, trg, train=True,backbool=False,verbose=verbose,find_cosine=args.max_cosine,pssrc = pssrc)
        else:
            loss,preds,contextsrc,contextpssrc = self.rec_translator.score(src, trg, train=True,backbool=False,verbose=verbose,find_cosine=args.max_cosine,find_preds=True,pssrc = pssrc)

        contextsrc = contextsrc.transpose(0,1)
        contextpssrc = contextpssrc.transpose(0,1)
        print("TARGET: {}".format(trg[0])) if verbose else print("",end='')
        print("PREDICTIONS") if verbose else print("",end='')
        print(preds[0]) if verbose else print("",end='')
        #find lossadv_gen
        cebatchloss = loss.data
        real = Variable(device(torch.LongTensor([int(self.srcbool) for i in range(args.batch)])))
        fake = Variable(device(torch.LongTensor([int(not self.srcbool) for i in range(args.batch)])))
        print('BATCH celoss, ppl per word: {0:4f},{1:4f}'.format(float(cebatchloss),float(self.perplexity_per_word()))) if verbose else print("",end='')

        # if float(cebatchloss) <args.batch*0.5:
        if True:
            lossadv_gen = self.discriminator(contextpssrc,real)
            print('BATCH lossadv_gen: {0:4f}'.format(float(lossadv_gen.data))) if verbose else print("",end='')
            sys.stdout.flush()
            loss+=lossadv_gen
        self.forward_time += time.time() - t
        self.nsteps +=1
        # Backpropagate error + optimize generator
        t = time.time()
        loss.div(args.batch).backward()
        self.celoss+=cebatchloss
        for optimizer in self.rec_optimizer:
            optimizer.step()

        # if float(cebatchloss) <args.batch*0.5:
        if True:
            # find lossadv_disc
            lossadv_disc = self.discriminator(contextsrc.detach(),real) + self.discriminator(contextpssrc.detach(),fake)
            print('BATCH lossadv_disc: {0:4f}'.format(float(lossadv_disc.data))) if verbose else print("",end='')
            # backpropagate error and optimize discriminator
            lossadv_disc.div(args.batch).backward()
            for optimizer in self.disc_optimizer:
                optimizer.step()
        self.backward_time += time.time() - t

    def reset_stats(self):
        self.src_word_count = 0
        self.trg_word_count = 0
        self.io_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.celoss = 0
        self.sentstats = np.zeros(3)
        # self.embloss = 0
        self.nsteps = 0
        self.nbacksteps = 0

    def perplexity_per_word(self):
        return np.exp(self.celoss/max(self.trg_word_count,1e-8))

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / max(self.total_time(),1e-8),  self.trg_word_count / max(self.total_time(),1e-8)

    def sent_stats(self):
        return self.sentstats/(self.nbacksteps)
 

class MganTrainer:
    def __init__(self, src2src_translator=None,src2trg_translator=None, trg2src_translator=None, trg2trg_translator=None, discriminator=None\
        , optimizers=None, corpus=None, gen_train=False):
        self.src2src_translator=src2src_translator
        self.disc = discriminator
        self.src2trg_translator=src2trg_translator
        self.trg2trg_translator=trg2trg_translator
        self.trg2src_translator=trg2src_translator
        self.corpus = corpus
        self.gen_train=gen_train
        self.listoptimizers = optimizers
        (self.disc_optimizers,self.enc_optimizers,self.simdec_optimizers,self.comdec_optimizers)= optimizers
        self.advbool=False
        self.reset_stats()

    def step(self,nstep,log_interval,device,args=None):
        verbose = (nstep % log_interval == 0)
        if nstep==1:
            self.advbool = args.phase
        # verbose = True
        # print(nstep)
        gen_train = self.gen_train
        # Reset gradients
        for optimizers in self.listoptimizers:
            for optimizer in optimizers:
                optimizer.zero_grad()

        # Read input sentences
        t = time.time()
        (simsrc,simtrg),(comsrc,comtrg) = self.corpus.next_batch(args.batch)
        if nstep<args.startfrom:
            return
        if gen_train:
            print("TRAINING THE GENERATOR") if verbose else print("",end='')
        else:
            print("TRAINING THE DISCRIMINATOR") if verbose else print("",end='')
        sys.stdout.flush()
        batchtrg_word_count = sum([len(data.tokenize(sentence)) + 1 for sentence in (simtrg+comtrg)])
        # self.src_word_count += sum([len(data.tokenize(sentence)) + 1 for sentence in src])  # TODO Depends on special symbols EOS/SOSfind_preds=True
        self.trg_word_count += batchtrg_word_count  # TODO Depends on special symbols EOS/SOS
        self.io_time += time.time() - t
        # Compute cross entropy loss
        t = time.time()
        if not verbose:
            celosssim,hiddensrc,hiddenpssrc,recsimwcontexts,simencodings = self.trg2trg_translator.score(simsrc, simtrg, train=gen_train,backbool=False,no_noise=not args.advdenoi,verbose=verbose,find_cosine=args.max_cosine,pass_context=True,pass_encodings=True)
            celosscom,hiddensrc,hiddenpssrc,reccomwcontexts,comencodings = self.src2src_translator.score(comsrc, comtrg, train=gen_train,backbool=False,no_noise=not args.advdenoi,verbose=verbose,find_cosine=args.max_cosine,pass_context=True,pass_encodings=True)
        else:
            celosssim,simpreds,hiddensrc,hiddenpssrc,recsimwcontexts,simencodings = self.trg2trg_translator.score(simsrc, simtrg, train=gen_train,backbool=False,no_noise=not args.advdenoi,verbose=verbose,find_cosine=args.max_cosine,find_preds=True,pass_context=True,pass_encodings=True)
            # print(celosssim,simpreds,hiddensrc,hiddenpssrc,recsimwcontexts.size(),simencodings)
            celosscom,compreds,hiddensrc,hiddenpssrc,reccomwcontexts,comencodings = self.src2src_translator.score(comsrc, comtrg, train=gen_train,backbool=False,no_noise=not args.advdenoi,verbose=verbose,find_cosine=args.max_cosine,find_preds=True,pass_context=True,pass_encodings=True)
        recsimwcontexts = recsimwcontexts.transpose(0,1)
        reccomwcontexts = reccomwcontexts.transpose(0,1)
        print("SIMPLE RECONSTRUCTION TRAINING PAIR:") if verbose else print("",end='')
        print("SOURCE: {}".format(simsrc[0])) if verbose else print("",end='')
        print("TARGET: {}".format(simtrg[0])) if verbose else print("",end='')
        print("SIMPLE PREDICTIONS") if verbose else print("",end='')
        print(simpreds[0]) if verbose else print("",end='')
        print("COMPLEX RECONSTRUCTION TRAINING PAIR:") if verbose else print("",end='')
        print("SOURCE: {}".format(comsrc[0])) if verbose else print("",end='')
        print("TARGET: {}".format(comtrg[0])) if verbose else print("",end='')
        print("COMPLEX PREDICTIONS") if verbose else print("",end='')
        print(compreds[0]) if verbose else print("",end='')
        cebatchloss = celosssim.data+celosscom.data
        #find encodings and contexts
        print("SIMPLIFYING and COMPLICATING TRAINING PAIR:") if verbose else print("",end='')
        print("SIMPLE SOURCE: {}".format(simsrc[0])) if verbose else print("",end='')
        print("COMPLEX SOURCE: {}".format(comsrc[0])) if verbose else print("",end='')
        #encodings are obtained in inference mode to avoid propagating harmful gradients from discriminator
        # encodings = self.src2src_translator.encode(norsrc, train=False,no_noise=True)
        compreds,norcomwcontexts = self.trg2src_translator.greedy(simsrc, max_ratio=2, train=gen_train,pass_att=True,no_noise=not args.advdenoi,encodings=simencodings,pass_context=True,detach_encoder=args.detach_encoder)
        simpreds, norsimwcontexts = self.src2trg_translator.greedy(comsrc,max_ratio=2,train=gen_train,pass_att=True,no_noise=not args.advdenoi,encodings=comencodings,pass_context=True,detach_encoder=args.detach_encoder)
        norcomwcontexts = norcomwcontexts.transpose(0,1)
        norsimwcontexts = norsimwcontexts.transpose(0,1)
        # print("simpreds",max([len(x[0].strip().split()) for x in simpreds]))
        # print("norsimwcontexts",norsimwcontexts.size())
        # print("simtrg",max([len(x.strip().split()) for x in simtrg]))
        # print("recsimwcontexts",recsimwcontexts.size())
        print("SIMPLIFYING PREDICTIONS") if verbose else print("",end='')
        print(simpreds[0]) if verbose else print("",end='')
        print("COMPLICATING PREDICTIONS") if verbose else print("",end='')
        print(compreds[0]) if verbose else print("",end='')
        print('BATCH celoss, ppl per word: {0:4f},{1:4f}'.format(float(cebatchloss),float(self.perplexity_per_word()))) if verbose else print("",end='')

        #find L_D and L_C
        sim = Variable(device(torch.LongTensor([0 for i in range(args.batch//4)])))
        com = Variable(device(torch.LongTensor([1 for i in range(args.batch//4)]))) 
        # print(recsimwcontexts)
        L_Drecsim = self.disc(recsimwcontexts.detach(),sim,type="discsim",train=not gen_train)
        L_Dnorsim = self.disc(norsimwcontexts.detach(),com,type="discsim",train=not gen_train)
        L_Dreccom = self.disc(reccomwcontexts.detach(),com,type="disccom",train=not gen_train)
        L_Dnorcom = self.disc(norcomwcontexts.detach(),sim,type="disccom",train=not gen_train)
        print("BATCH losses of discriminator in increasing order of complication (recsim,norsim,norcom,reccom): {0:4f} {1:4f} {2:4f} {3:4f}".format(float(L_Drecsim.data),float(L_Dnorsim.data)\
            ,float(L_Dnorcom.data),float(L_Dreccom.data))) if verbose else print("",end='')
        L_D = L_Drecsim.div(args.batch//2)+L_Dnorsim.div(args.batch//2)+L_Dreccom.div(args.batch//2)+L_Dnorcom.div(args.batch//2)
        if not args.singleclassf:
            L_Crecsim = self.disc(recsimwcontexts.detach(),sim,type="classsim",train=not gen_train)
            L_Cnorsim = self.disc(norsimwcontexts.detach(),sim,type="classcom",train=not gen_train)
            L_Creccom = self.disc(reccomwcontexts.detach(),com,type="classcom",train=not gen_train)
            L_Cnorcom = self.disc(norcomwcontexts.detach(),com,type="classsim",train=not gen_train)
            print("BATCH losses of classifiers in increasing order of complication (recsim,norsim,norcom,reccom): {0:4f} {1:4f} {2:4f} {3:4f}".format(float(L_Crecsim.data),float(L_Cnorsim.data)\
                ,float(L_Cnorcom.data),float(L_Creccom.data))) if verbose else print("",end='')
            L_C = L_Crecsim.div(args.batch//2)+L_Cnorsim.div(args.batch//2)+L_Creccom.div(args.batch//2)+L_Cnorcom.div(args.batch//2)
        else:
            print("training one classifier") if verbose else print('',end='')
            L_Crecsim = self.disc(recsimwcontexts.detach(),sim,type="classsim",train=not gen_train)
            L_Creccom = self.disc(reccomwcontexts.detach(),com,type="classsim",train=not gen_train)
            print("BATCH losses of classifiers in increasing order of complication (recsim,reccom): {0:4f} {1:4f}".format(float(L_Crecsim.data)\
                ,float(L_Creccom.data))) if verbose else print("",end='')
            L_C = L_Crecsim.div(args.batch//2)+L_Creccom.div(args.batch//2)
        if not gen_train:
            L = L_C+L_D
            L.backward()
            for optimizer in self.disc_optimizers:
                optimizer.step()
        else:
            #find L_G
            L_Gdiscnorsim = self.disc(norsimwcontexts,sim,type="discsim",train = gen_train)
            L_Gdiscnorcom = self.disc(norcomwcontexts,com,type="disccom",train = gen_train)
            if not args.singleclassf:
                if not args.oldclassf:
                    L_Gclassnorsim = self.disc(norsimwcontexts,sim,type="classcom",train= gen_train)
                    L_Gclassnorcom = self.disc(norcomwcontexts,com,type="classsim",train= gen_train)
                else:
                    L_Gclassnorsim = self.disc(norsimwcontexts,sim,type="classsim",train= gen_train)
                    L_Gclassnorcom = self.disc(norcomwcontexts,com,type="classcom",train= gen_train)
            else:
                print("training using one classifier") if verbose else print('',end='')
                L_Gclassnorsim = self.disc(norsimwcontexts,sim,type="classsim",train=gen_train)
                L_Gclassnorcom = self.disc(norcomwcontexts,com,type="classsim",train=gen_train)

            print("BATCH losses of generator (discnorsim,discnorcom,classnorsim,classnorcom): {0:4f} {1:4f} {2:4f} {3:4f}".format(float(L_Gdiscnorsim.data),float(L_Gdiscnorcom.data)\
            ,float(L_Gclassnorsim.data),float(L_Gclassnorcom.data))) if verbose else print("",end='')
            if args.noclassf:
                classrho=0.0
            else:
                classrho=1.0
            if args.noclasssim:
                classsimrho=0.0
            else:
                classsimrho=1.0
            if not args.periodic:
                if nstep >= args.startadv:
                    print("using adversarial loss") if verbose else print("",end='') 
                    rho = args.rho
                else:
                    print("using no adversarial loss") if verbose else print("",end='')
                    rho = 0.0
            else:
                if nstep%10000 ==0:
                    self.advbool = not self.advbool
                if not self.advbool:
                    print("using no adversarial loss") if verbose else print("",end='')
                    rho = 0
                else:
                    print("using adversarial loss") if verbose else print("",end='') 
                    rho = args.rho

            L = (L_Gdiscnorsim.div(args.batch//4)+L_Gdiscnorcom.div(args.batch//4)+(L_Gclassnorsim.div(args.batch//4).mul(classsimrho)+\
            L_Gclassnorcom.div(args.batch//4)).mul(classrho)).mul(rho)+celosssim.div(args.batch//4)+celosscom.div(args.batch//4)
            L.backward()
            for optimizer in self.enc_optimizers:
                optimizer.step()
            for optimizer in self.simdec_optimizers:
                optimizer.step()
            for optimizer in self.comdec_optimizers:
                optimizer.step()
        self.forward_time += time.time() - t
        self.nsteps +=1
        # Backpropagate error + optimize generator
        t = time.time()
        self.celoss+=cebatchloss
        
        self.backward_time += time.time() - t

    def reset_stats(self):
        self.src_word_count = 0
        self.trg_word_count = 0
        self.io_time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.celoss = 0
        self.sentstats = np.zeros(3)
        # self.embloss = 0
        self.nsteps = 0
        self.nbacksteps = 0

    def perplexity_per_word(self):
        return np.exp(self.celoss/max(self.trg_word_count,1e-8))

    def total_time(self):
        return self.io_time + self.forward_time + self.backward_time

    def words_per_second(self):
        return self.src_word_count / max(self.total_time(),1e-8),  self.trg_word_count / max(self.total_time(),1e-8)

    def sent_stats(self):
        return self.sentstats/(self.nbacksteps)
 



class Validator:
    def __init__(self, translator, source, reference, batch_size=3, beam_size=0):
        self.translator = translator
        self.source = source
        self.reference = reference
        self.sentence_count = len(source)
        self.reference_word_count = sum([len(data.tokenize(sentence)) + 1 for sentence in self.reference])  # TODO Depends on special symbols EOS/SOS
        self.batch_size = batch_size
        self.beam_size = beam_size

        # Sorting
        lengths = [len(data.tokenize(sentence)) for sentence in self.source]
        self.true2sorted = sorted(range(self.sentence_count), key=lambda x: -lengths[x])
        self.sorted2true = sorted(range(self.sentence_count), key=lambda x: self.true2sorted[x])
        self.sorted_source = [self.source[i] for i in self.true2sorted]
        self.sorted_reference = [self.reference[i] for i in self.true2sorted]

    def perplexity(self):
        loss = 0
        for i in range(0, self.sentence_count, self.batch_size):
            # print("VALIDATING BATCH {}".format(i))
            j = min(i + self.batch_size, self.sentence_count)
            loss += self.translator.score(self.sorted_source[i:j], self.sorted_reference[i:j], train=False).data
        return np.exp(loss/self.reference_word_count)

    def translate(self):
        translations = []
        for i in range(0, self.sentence_count, self.batch_size):
            j = min(i + self.batch_size, self.sentence_count)
            batch = self.sorted_source[i:j]
            if self.beam_size <= 0:
                translations += self.translator.greedy(batch, train=False)
            else:
                translations += self.translator.beam_search(batch, train=False, beam_size=self.beam_size)
        return [translations[i] for i in self.sorted2true]


class Logger:
    def __init__(self, name, trainer, validators=(), output_prefix=None, encoding='utf-8'):
        self.name = name
        self.trainer = trainer
        self.validators = validators
        self.output_prefix = output_prefix
        self.encoding = encoding

    def log(self, step=0):
        if self.trainer is not None or len(self.validators) > 0:
            print('{0}'.format(self.name))
        if self.trainer is not None:
            print('  - Training:   {0:10.2f}   ({1:.2f}s: {2:.2f}tok/s src, {3:.2f}tok/s trg; epoch {4} '
                .format(float(self.trainer.perplexity_per_word()), self.trainer.total_time(),
                self.trainer.words_per_second()[0], self.trainer.words_per_second()[1], self.trainer.corpus.epoch))
            print("sentstats(ds,ts,overlap): {0} )".format(self.trainer.sent_stats()))\
            if self.trainer.backbool else print("",end="") 
            self.trainer.reset_stats()
        sys.stdout.flush()
        for id, validator in enumerate(self.validators):
            t = time.time()
            perplexity = validator.perplexity()
            print('  - Validation: {0:10.2f}   ({1:.2f}s)'.format(float(perplexity), time.time() - t))
            if self.output_prefix is not None:
                f = open('{0}.{1}.{2}.txt'.format(self.output_prefix, id, step), mode='w',
                         encoding=self.encoding, errors='surrogateescape')
                for line in validator.translate():
                    print(line, file=f)
                f.close()
        sys.stdout.flush()