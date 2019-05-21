import sys
import os
import codecs
import logging
from itertools import izip
from SARI import SARIsent
from nltk.translate.bleu_score import *
import numpy as np
smooth = SmoothingFunction()
from nltk import word_tokenize
from textstat.textstat import textstat

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)

def files_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def folders_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isdir(os.path.join(mypath,f)) ]

def files_in_folder_only(mypath):
    return [ f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]

def remove_features(sent):
    tokens = sent.split(" ")
    return " ".join([token.split("|")[0] for token in tokens])

def remove_underscores(sent):
    return sent.replace("_", " ")

def replace_parant(sent):
    sent = sent.replace("-lrb-", "(").replace("-rrb-", ")")
    return sent.replace("(", "-lrb-").replace(")", "-rrb-")

def lowstrip(sent):
    return sent.lower().strip()

def normalize(sent):
    return replace_parant(lowstrip(sent))

def as_is(sent):
    return sent

def get_hypothesis(filename):
    hypothesis = '-'
    if "_h1" in filename:
        hypothesis = '1'
    elif "_h2" in filename:
        hypothesis = '2'
    elif "_h3" in filename:
        hypothesis = '3'
    elif "_h4" in filename: 
        hypothesis = '4'
    return hypothesis

def mean(numbers):
    return float(sum(numbers)) / max(len(numbers), 1)

def print_scores(pairs, whichone = ''):
    # replace filenames by hypothesis name for csv pretty print
    for k,v in pairs:
        hypothesis = get_hypothesis(k)
        print "\t".join( [whichone, "{:10.2f}".format(v), k, hypothesis] )

def SARI_file(source, preds, refs, preprocess,pass_indiv=False):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds, refs]]
    scores = []
    for src, pred, ref in izip(*files):
        references = [preprocess(r) for r in ref.split('\t')]
        scores.append(SARIsent(preprocess(src), preprocess(pred), references))
    for fis in files:
        fis.close()
    if not pass_indiv :
        return mean(scores)
    else:
        return mean(scores),scores


# BLEU doesn't need the source
def BLEU_file(source, preds, refs, preprocess=as_is,pass_indiv=False):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [preds, refs]]
    scores = []
    references = []
    hypothese = []
    for pred, ref in izip(*files):
        references.append([word_tokenize(preprocess(r)) for r in ref.split('\t')])
        hypothese.append(word_tokenize(preprocess(pred)))
    for fis in files:
        fis.close()
    # Smoothing method 3: NIST geometric sequence smoothing
    if not pass_indiv:
        return corpus_bleu(references, hypothese, smoothing_function=smooth.method3)
    else:
        return corpus_bleu(references, hypothese, smoothing_function=smooth.method3),[corpus_bleu([ref],[hypo], smoothing_function=smooth.method3) for ref,hypo in zip(references,hypothese)]

def iBLEU_file(source, preds, refs, preprocess=as_is,pass_indiv=False):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds, refs]]
    scores = []
    references = []
    hypothese = []
    ibleu = 0
    n = 0
    for src, pred, ref in izip(*files):
        n+=1
        references = [word_tokenize(preprocess(r)) for r in ref.split('\t')]
        hypothese = word_tokenize(preprocess(pred))
        source = word_tokenize(preprocess(src))
        ibleu+=0.9*corpus_bleu([references],[hypothese], smoothing_function=smooth.method3)-0.1*corpus_bleu([source],[hypothese], smoothing_function=smooth.method3)
        
    ibleu/=n
    for fis in files:
        fis.close()
    # Smoothing method 3: NIST geometric sequence smoothing
    return ibleu


def fkBLEU_file(source, preds, refs, preprocess=as_is,pass_indiv=False):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds, refs]]
    scores = []
    references = []
    hypothese = []
    fkbleu = 0
    n = 0
    for src, pred, ref in izip(*files):
        references = [word_tokenize(preprocess(r)) for r in ref.split('\t')]
        hypothese = word_tokenize(preprocess(pred))
        source = word_tokenize(preprocess(src))
        ibleu=0.9*corpus_bleu([references],[hypothese], smoothing_function=smooth.method3)-0.1*corpus_bleu([source],[hypothese], smoothing_function=smooth.method3)
        try:
            fkdiff = textstat.flesch_reading_ease(' '.join(hypothese))-textstat.flesch_reading_ease(' '.join(source))
            n+=1
            fkdiff= 1/(1+np.exp(-fkdiff))
            fkbleu+=fkdiff*ibleu
        except Exception:
            continue
    fkbleu/=n
    for fis in files:
        fis.close()
    # Smoothing method 3: NIST geometric sequence smoothing
    return ibleu


def worddiff_file(source, preds, refs, preprocess=as_is,pass_indiv=False):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds]]
    scores = []
    references = []
    hypothese = []
    worddiff = 0
    n = 0
    for src, pred in izip(*files):
        source = word_tokenize(preprocess(src))
        hypothese = word_tokenize(preprocess(pred))
        n+=1
        worddiff+=len(source)-len(hypothese)
    # print(worddiff)
    # print(n)
    worddiff/=float(n)
    for fis in files:
        fis.close()
    # Smoothing method 3: NIST geometric sequence smoothing
    return worddiff/100.0

def score(source, refs, fold, METRIC_file, preprocess=as_is,pass_indiv=False):
    new_files = files_in_folder(fold)
    data = []
    for fis in new_files:
        # ignore log files
        if ".log" in os.path.basename(fis):
            continue
        logging.info("Processing "+os.path.basename(fis))
        if not pass_indiv:
            score = METRIC_file(source, fis, refs, preprocess,pass_indiv=False)
            val = 100*score
        else:
            score,scorearr = METRIC_file(source, fis, refs, preprocess,pass_indiv=True)
            val = 100*score
            valarr = [100*scoreelem for scoreelem in scorearr]
        logging.info("Done "+str(val))
        data.append((os.path.basename(fis), val))
    data.sort(key=lambda tup: tup[1])
    data.reverse()
    if not pass_indiv:
        return data
    else:
        return data,valarr

if __name__ == '__main__':
    try:
        revbleu = None
        source = sys.argv[1]
        logging.info("Source: " + source)
        refs = sys.argv[2]
        logging.info("References in tsv format: " + refs)
        fold = sys.argv[3]
        logging.info("Directory of predictions: " + fold)
        if(len(sys.argv)==5):
            revbleu = True
    except:
        logging.error("Input parameters must be: " + sys.argv[0] 
            + "    SOURCE_FILE    REFS_TSV (paste -d \"\t\" * > reference.tsv)    DIRECTORY_OF_PREDICTIONS")
        sys.exit(1)

    '''
        SARI can become very unstable to small changes in the data.
        The newsela turk references have all the parantheses replaced
        with -lrb- and -rrb-. Our output, however, contains the actual
        parantheses '(', ')', thus we prefer to apply a preprocessing
        step to normalize the text.
    '''

    sari_test,sariarr = score(source, refs, fold, SARI_file, normalize,pass_indiv=True) if not revbleu else None
    bleu_test,bleuarr = score(source, refs, fold, BLEU_file, lowstrip,pass_indiv=True) 
    #find IBLEU = 0.9*BLEU(fold,refs)-0.1*BLEU(fold,source)
#     ibleu = score(source,refs,fold,iBLEU_file,lowstrip,pass_indiv=False)
#     fkbleu = score(source,refs,fold,fkBLEU_file,lowstrip,pass_indiv=False)
    worddiff =score(source,refs,fold,worddiff_file,lowstrip,pass_indiv=False)
    whichone = os.path.basename(os.path.abspath(os.path.join(fold, '..'))) + \
                    '\t' + \
                    os.path.basename(refs).replace('.ref', '').replace("test_0_", "")
    print_scores(sari_test, "SARI\t" + whichone) 
    print_scores(bleu_test, "BLEU\t" + whichone)
#     print_scores(ibleu, "iBLEU\t" + whichone)
#     print_scores(fkbleu,"fkBLEU\t"+whichone)
    print_scores(worddiff,"worddiff\t"+whichone)
    # print('SARI individual scores')
    # print('\n'.join([str(score) for score in sariarr]))
    # print('BLEU individual scores')
    # print('\n'.join([str(score) for score in bleuarr]))