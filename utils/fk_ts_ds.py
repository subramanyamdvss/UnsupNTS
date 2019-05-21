import argparse
import sys
import torch
from fuzzywuzzy import fuzz
from textstat.textstat import textstat

import spacy
from collections import Counter
import numpy as np

nlp = spacy.load('en')


def inorder(node):
    tags = str(node.dep_) +" "
    #print tags, node.text
    if node.lefts:
        for n in node.lefts:
            tags+= inorder(n)
    if node.rights:
        for n in node.rights:
            tags+= inorder(n)
    return tags

def doc_sim(doc1,doc2):
    sim = doc1.similarity(doc2)
    return sim

def tree_sim(doc1,doc2):
    ino1 = ""
    ino2 = ""
    for tok in doc1:
        if tok.dep_=="ROOT":
            ino1 = inorder(tok)
            break
    for tok in doc2:
        if tok.dep_=="ROOT":
            ino2 = inorder(tok)
            break
    ts = fuzz.ratio(ino1,ino2)
    return ts

def sentence_stats(s1,s2):
    #s2 should be predictions and s1 should be source
    try:
        fkdiff = textstat.flesch_reading_ease(s2)-textstat.flesch_reading_ease(s1)
    except Exception:
        fkdiff = 0.0
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    ts = tree_sim(doc1,doc2)/100
    ds = doc_sim(doc1,doc2)
    return (torch.FloatTensor([fkdiff,ts,ds]))

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='edit distance')
    parser.add_argument('-i', '--input', default=sys.stdin.fileno(), help='the input file (defaults to stdin)')
    parser.add_argument('-src','--source',default="",help='source file with which edit distance should be calculated.')
    args = parser.parse_args()
    infile = open(args.input,'r')
    srcfile = open(args.source,'r')
    inplines = [ln.strip() for ln in infile]
    srclines = [ln.strip() for ln in srcfile]
    stats = torch.FloatTensor(3).fill_(0.0)
    for i in range(len(inplines)):
    	stats+=sentence_stats(srclines[i],inplines[i])
    stats=stats.div(len(inplines))
    print("(fkdiff,ts,ds) between {} and {} are: {:.4f} {:.4f} {:.4f}".format(args.input.split("/")[-1],args.source.split("/")[-1]\
        ,stats[0],stats[1],stats[2]))

if __name__ == '__main__':
    main()