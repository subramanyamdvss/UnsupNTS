import sys
import random
from random import shuffle
random.seed(7)

def getrepl(word):
	if word =='-LRB-':
		return '('
	if word=='-RRB-':
		return ')'
	return word

def replace_RB(pair):
	lst = []
	for sent in pair:
		lst.append(' '.join([getrepl(w) for w in sent.strip().split()])+'\n')
	return tuple(lst)

def getsplit(nsplit,nwiki,fsplit,fen,fsen,outpathen,outpathsen):
	ns = 0
	fouten = open(outpathen,'w')
	foutsen = open(outpathsen,'w')
	lst = []
	for pair in yieldsplit(fsplit):
		pr = replace_RB(pair)
		lst.append(pr)
		ns+=1
		if ns==nsplit:
			break
	ns=0
	for senten,sentsen in yieldwiki(fen,fsen):
		lst.append(replace_RB((senten,sentsen)))
		ns+=1
		if ns==nwiki:
			break
	shuffle(lst)
	for pair in lst:
		fouten.write(pair[0])
		foutsen.write(pair[1])
	return

def yieldsplit(fsplit):
	prevsrc = None
	prevtrglst = []
	while True:
		ln = fsplit.readline()
		ln = ln.strip()
		if ln == "":
			continue
		if len(ln.split(':'))==1:
			#src area
			if prevsrc is not None:
				assert len(prevtrglst) >=1
				yield (prevsrc,prevtrglst[random.randint(0,len(prevtrglst)-1)])
			prevsrc = fsplit.readline().strip()+'\n'
			prevtrglst = []
		else:
			#trg area, get all the sentences 
			assert prevsrc is not None
			splits = []
			while(True):
				lln = fsplit.readline().strip()
				if lln =="":
					break
				if lln.split('=')[0]=='category':
					continue
				else:
					splits.append(lln)
			#isolating 1-1 split type.
			prevtrglst = [ln for ln in prevtrglst if len([w for w in ln.split() if w=='.'])==2]
			prevtrglst.append(' '.join(splits)+'\n')

def yieldwiki(fen,fsen):
	while(True):
		senten = fen.readline().strip()
		sentsen = fsen.readline().strip()
		if " " in [senten,sentsen] or senten==sentsen:
			continue
		yield senten+'\n',sentsen+'\n'



def main():
	fsplit = open('tsdata/benchmark-v1.0/final-complexsimple-meanpreserve-intreeorder-full.txt')
	fen = open('tsdata/train.en')
	fsen = open('tsdata/train.sen')
	outpathen = sys.argv[1]
	outpathsen = sys.argv[2]
	# getsplit(15000,5000,fsplit,fen,fsen,outpathen,outpathsen) #the first 20k version.
	getsplit(16000,4000,fsplit,fen,fsen,outpathen,outpathsen)
if __name__ == '__main__':
	main()