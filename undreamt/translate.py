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

import argparse
import sys
import torch
import undreamt
from undreamt.translator import Translator
from undreamt import data
from undreamt.devices import gpu
import random
random.seed(7)
torch.manual_seed(7)
# torch.cuda.manual_seed_all(7)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Translate using a pre-trained model')
    parser.add_argument('model', help='a model previously trained with train.py')
    parser.add_argument('--batch_size', type=int, default=50, help='the batch size (defaults to 50)')
    parser.add_argument('--beam_size', type=int, default=12, help='the beam size (defaults to 12, 0 for greedy search)')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    parser.add_argument('-o', '--output', default=sys.stdout.fileno(), help='the output file (defaults to stdout)')
    parser.add_argument('--noise',type=float,default=0.5)
    parser.add_argument('--pass_att',action='store_true',default=False)
    parser.add_argument('--src_embeddings',default=None,help='common intersection source embeddings')
    parser.add_argument('--cutoff', type=int, default=None, help='cutoff for source embeddings above')
    parser.add_argument('--cat_embedds',help='use torch.load to load src and trg ')
    parser.add_argument('--ncontrol',type=int,default=0,help='control number given while using the decoder')
    args = parser.parse_args()

    try:
        t = torch.load(args.model)
    except Exception:
        t = torch.load(args.model,map_location={'cuda:1':'cuda:0'})

    # Translate sentences
    end = False
    fin = open(args.input, encoding=args.encoding, errors='surrogateescape')
    fout = open(args.output, mode='w', encoding=args.encoding, errors='surrogateescape')
    if args.src_embeddings is not None:
        encoder_embeddings,src_dictionary = data.read_embeddings(open(args.src_embeddings,'r'),threshold=args.cutoff) 
        encoder_embeddings = gpu(encoder_embeddings)
        t.decoder_embeddings=gpu(t.decoder_embeddings)
        t.generator=gpu(t.generator)
        t.encoder=gpu(t.encoder)
        t.decoder=gpu(t.decoder)

        translator_new = Translator(encoder_embeddings,t.decoder_embeddings,t.generator,src_dictionary,\
        t.trg_dictionary,t.encoder,t.decoder,t.denoising,t.device)
    else:
        t.device=gpu
        t.encoder=gpu(t.encoder)
        t.decoder=gpu(t.decoder)
        t.encoder_embeddings=gpu(t.encoder_embeddings)
        t.decoder_embeddings=gpu(t.decoder_embeddings)
        t.generator=gpu(t.generator)
        t.src_dictionary = data.Dictionary(t.src_dictionary.id2word[1:])
        t.trg_dictionary = data.Dictionary(t.trg_dictionary.id2word[1:])
        translator_new = Translator(t.encoder_embeddings,t.decoder_embeddings,t.generator,t.src_dictionary,\
        t.trg_dictionary,t.encoder,t.decoder,t.denoising,t.device)
    # print (translator_new.denoising)
    # exit(0)
    while not end:
        batch = []
        while len(batch) < args.batch_size and not end:
            line = fin.readline()
            if not line:
                end = True
            else:
                batch.append(line)
        if args.beam_size <= 0 and len(batch) > 0:
            for translation in translator_new.greedy(batch, train=False):
                print(translation, file=fout)
        elif len(batch) > 0:
            translations = translator_new.beam_search(batch, train=False, beam_size=12, max_ratio=2,rnk=6,noiseratio=args.noise,pass_att=args.pass_att,ncontrol=args.ncontrol if args.ncontrol!=0 else None)
            print(translations)
            if args.pass_att:
                for translation1,trans2 in translations:
                    print(translation1,trans2, file=fout)
            else:
                for translation in translations:
                    print(translation, file=fout)
        fout.flush()
    fin.close()
    fout.close()


if __name__ == '__main__':
    main()
