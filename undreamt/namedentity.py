
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import argparse
import nltk
import undreamt
from undreamt import data

def extract_entity_names(t):
    entity_names = []

    if hasattr(t, 'label') and t.label:
        if t.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in t]))
        else:
            for child in t:
                entity_names.extend(extract_entity_names(child))

    return entity_names




# def main():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--input',default='gendirprod/gen_lower.sen.src2trg.learndec.back3.newvocab.fk4.semilater.20.test',\
# 		help='sequence of sentences for which named entities are to be identified and replaced with <NAMED> token')
# 	parser.add_argument('--output',default='gendirprod/gen_lower.sen.src2trg.learndec.back3.newvocab.fk4.semilater.20.test.named')
# 	args=parser.parse_args()
# 	nlp = spacy.load('en_core_web_sm')
# 	doc = nlp('San Francisco considers banning sidewalk delivery robots')
# 	outfl = open(args.output,'w')
# 	for ln in open(args.input,'r'):
# 		line = []
# 		doc = nlp(ln.strip())
# 		for word in doc:
# 			if word.ent_type_ != "":
# 				line.append("<NAMED>")
# 			else:
# 				line.append(word.text)
# 		outfl.write(' '.join(line)+'\n')


# def main():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--input',default='gendirprod/gen_lower.sen.src2trg.learndec.back3.newvocab.fk4.semilater.20.test',\
# 		help='sequence of sentences for which named entities are to be identified and replaced with <NAMED> token')
# 	parser.add_argument('--output',default='gendirprod/gen_lower.sen.src2trg.learndec.back3.newvocab.fk4.semilater.20.test.named')
# 	args=parser.parse_args()
# 	outfl = open(args.output,'w')
# 	sentences = [ln.strip() for ln in open(args.input,'r')]
# 	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# 	tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
# 	chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
# 	entity_names = []
# 	for tree in chunked_sentences:
# 	    # Print results per sentence
# 	    # print extract_entity_names(tree)

# 	    entity_names.extend(extract_entity_names(tree))

# 	# Print all entity names
# 	#print entity_names
# 	print(entity_names)
# 	# Print unique entity names
# 	print(set(entity_names))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--inputsrc',default='/home/15CS10013/important-sai/ts12/tsdata/test.en.lower',\
		help='sequence of sentences for which named entities are to be identified and replaced with <NAMED> token')
	parser.add_argument('--inputgen',default='/home/15CS10013/important-sai/ts12/gendirprod/gen_lower.sen.src2trg.learndec.back3.newvocab.fk4.semilater.20.test',\
		help='sequence of sentences for which named entities are to be identified and replaced with <NAMED> token')
	parser.add_argument('--output',default='/home/15CS10013/important-sai/ts12/gendirprod/gen_lower.sen.src2trg.learndec.back3.newvocab.fk4.semilater.20.test')
	args=parser.parse_args()
	outposgen = open(args.output+'.pos','w')
	outoovsrc = open(args.inputsrc+'.oov','w')
	outoovgen = open(args.inputgen+'.oov','w')
	sentencesgen = [ln.strip() for ln in open(args.inputgen,'r')]
	tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentencesgen]
	tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
	# chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)
	# entity_names = []
	# for tree in chunked_sentences:
	#     # Print results per sentence
	#     # print extract_entity_names(tree)

	#     entity_names.extend(extract_entity_names(tree))

	# # Print all entity names
	# #print entity_names
	# print(entity_names)
	# Print unique entity names
	# print(set(entity_names))
	ts='/home/15CS10013/important-sai/ts12'
	tsdata=ts+'/tsdata'
	vocabgen = data.read_embeddings(open(tsdata+'/fkeasypart.lower.vec'),vocabonly=True,threshold=50000)
	vocabsrc = data.read_embeddings(open(tsdata+'/fkdifficpart.lower.vec.id.com'),vocabonly=True,threshold=50000)
	sentencessrc = [ln.strip() for ln in open(args.inputsrc,'r')]
	for sentence in tagged_sentences:
		print(sentence,file=outposgen)
	for sentence in sentencessrc:
		print([(word1,word2) for word1,word2 in zip(sentence.split(),vocabsrc.ids2sentence(vocabsrc.sentence2ids(sentence)).split())] ,file=outoovsrc)
	for sentence in sentencessrc:
		print( [(word1,word2) for word1,word2 in zip(sentence.split(),vocabgen.ids2sentence(vocabgen.sentence2ids(sentence)).split())],file=outoovgen)


# def saveembedds(embedds,comdict,filename):
# 	fl = open(filename,'w')
# 	dim = embedds.weight.data.size()[1]
# 	count = embedds.weight.data.size()[0]-1
# 	header = '{} {}\n'.format(count,dim)
# 	fl.write(header)
# 	for i in range(1,count+1):
# 		word = comdict.id2word[i]
# 		vec = ' '.join(str(x) for x in embedds.weight.data[i])
# 		pair = '{} {}\n'.format(word,vec)
# 		fl.write(pair)

# def main():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--inputsrc',default='/home/15CS10013/important-sai/ts12/tsdata/test.en.lower',\
# 		help='sequence of sentences for which named entities are to be identified and replaced with <NAMED> token')
# 	parser.add_argument('--inputgen',default='/home/15CS10013/important-sai/ts12/gendirprod/gen_lower.sen.src2trg.learndec.back3.newvocab.fk4.semilater.20.test',\
# 		help='sequence of sentences for which named entities are to be identified and replaced with <NAMED> token')
# 	parser.add_argument('--output',default='/home/15CS10013/important-sai/ts12/gendirprod/gen_lower.sen.src2trg.learndec.back3.newvocab.fk4.semilater.20.test')
# 	args=parser.parse_args()
# 	ts='/home/15CS10013/important-sai/ts12'
# 	tsdata=ts+'/tsdata'
# 	vocabtrg = data.read_embeddings(open(tsdata+'/fkeasypart.lower.vec.id','r'),vocabonly=True,threshold=50000)
# 	vocabsrc = data.read_embeddings(open(tsdata+'/fkdifficpart.lower.vec.id','r'),vocabonly=True,threshold=50000)
# 	commonvocab = [x for x in list(set(vocabtrg.id2word)&set(vocabsrc.id2word)) if x is not None]
# 	embeddstrgcom,trgcomdict = data.read_embeddings(open(tsdata+'/fkeasypart.lower.vec.id','r'),vocabulary=commonvocab,threshold=50000)
# 	embeddssrccom,srccomdict = data.read_embeddings(open(tsdata+'/fkdifficpart.lower.vec.id','r'),vocabulary=commonvocab,threshold=50000)
# 	print(len(trgcomdict.id2word),len(srccomdict.id2word))
# 	saveembedds(embeddstrgcom,trgcomdict,tsdata+'/fkeasypart.lower.vec.id.com')
# 	saveembedds(embeddssrccom,srccomdict,tsdata+'/fkdifficpart.lower.vec.id.com')

if __name__ == '__main__':
	main()
