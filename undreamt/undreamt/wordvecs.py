import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import argparse
from undreamt import data
import torch
import torch.nn as nn

def saveembedds(embedds,comdict,filename):
	fl = open(filename,'w')
	dim = embedds.weight.data.size()[1]
	count = embedds.weight.data.size()[0]-1
	header = '{} {}\n'.format(count,dim)
	fl.write(header)
	for i in range(1,count+1):
		word = comdict.id2word[i]
		vec = ' '.join(str(x) for x in embedds.weight.data[i])
		pair = '{} {}\n'.format(word,vec)
		fl.write(pair)


def getembeddings(srcpath,trgpath,compath,cutoff=50000):
	ts='/home/15CS10013/important-sai/ts12'
	tsdata=ts+'/tsdata'
	compath = tsdata+'/fk.lower.vec'
	srcpath = tsdata+'/fkdifficpart.lower.vec.id'
	trgpath = tsdata+'/fkeasypart.lower.vec.id'
	vocabcom = data.read_embeddings(open(compath),vocabonly=True)
	vocabsrc = data.read_embeddings(open(srcpath),vocabonly=True)
	vocabtrg = data.read_embeddings(open(trgpath),vocabonly=True)
	vocabcom = set(vocabcom.id2word[1:])
	vocabsrc = set(vocabsrc.id2word[1:])
	vocabtrg = set(vocabtrg.id2word[1:])
	vocabinter = vocabcom & vocabsrc & vocabtrg
	embeddcom,vocabcom = data.read_embeddings(open(compath),vocabulary=vocabinter)
	embeddsrc,vocabsrc = data.read_embeddings(open(srcpath),vocabulary=vocabinter)
	embeddtrg,vocabtrg = data.read_embeddings(open(trgpath),vocabulary=vocabinter)
	saveembedds(embeddsrc,vocabsrc,tsdata+'/fkeasypart.lower.vec.id.com')
	saveembedds(embeddtrg,vocabtrg,tsdata+'/fkdifficpart.lower.vec.id.com')
	saveembedds(embeddcom,vocabcom,tsdata+'/fk.lower.vec.com')

	# embeddsrccom = nn.Embedding(embeddsrc.weight.data.size(0),embeddsrc.weight.data.size(1)+embeddcom.weight.data.size(1))
	# embeddsrccom.weight.data = torch.cat([embeddsrc.weight.data,embeddcom.weight.data],dim=1)
	# embeddtrgcom = nn.Embedding(embeddtrg.weight.data.size(0),embeddtrg.weight.data.size(1)+embeddcom.weight.data.size(1))
	# embeddtrgcom.weight.data = torch.cat([embeddtrg.weight.data,embeddcom.weight.data],dim=1)
	return (embeddsrccom,vocabsrc),(embeddtrgcom,vocabtrg)

if __name__ == '__main__':
	getembeddings(None,None,None)