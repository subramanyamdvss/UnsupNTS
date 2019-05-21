# Compare semantic (word-embedding based) and syntactic similarity between two sentencces

#Insall spacy via `pip install spacy' if you get import error. Then run `python -m spacy download en' 
#Install fuzzywuzzy using `pip install fuzzywuzzy'
# Install python-Levenshtein using `pip install python-Levenshtein'
# import future        # pip install future
# import builtins      # pip install future
# import past          # pip install future
# import six           # pip install six
import spacy
from fuzzywuzzy import fuzz
from collections import Counter
import numpy as np

nlp = spacy.load('en')

def find_overlap(sent,original):
	# interseclen/reflen
	lsent = sent.split()
	lorig = original.split()
	w1 = Counter(lsent)
	w2 = Counter(lorig)
	w3 = w1 & w2
	overlap = float(sum(w3.values()))/max(1e-8,len(lsent))
	return overlap if overlap<=1 else 1
	
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
	#CALL this function .
	#Returns a list: [document_similarity, tree_similarity]
	overlap = find_overlap(s1,s2)
	doc1 = nlp(s1)
	doc2 = nlp(s2)
	ts = tree_sim(doc1,doc2)
	# if overlap >0.9 and ts==100:
	# 	#even changing a very minor portion would result in a high sentence similarity, which we do not want.
	# 	ds = 0.001
	# 	ts = 0.1
	# 	return [ds,ts]
	
	ds = doc_sim(doc1,doc2)
	return (np.asarray((ds,ts,overlap)))




# def similarity(s1,s2,lexical=True):
# 	#we want the model to repeat the input but using different words
# 	#document similarity: words either verbs or nouns should be synonyms in two sentences.
# 	#tree similarity: if syntactic structure of the two sentences should be same 
# 	#to achieve lexical paraphrasing -> the document similarity should be high but the overlap should be low and tree similarity should be high
# 	#to achieve syntactic paraphrasing -> the tree similarity should be low and document similarity should be high.
# 	#this function returns reward which shall improve the lexical paraphrasing ability of the model.
	
# 	#docsimil-intersection
# 	#stage-I: doesn't return anything relevant, hence overlap will be low. hence safe to use docsimil as reward
# 	#stage-II: returns outputs with repeating words. Hence intersection between input and output is a littile considerable.  overlap measure
# 	#compromised, 
# 	#stage-III: complete autoencoding, returns whatver input is given. here we have high tree similarity and document similarity.




if __name__ =="__main__":
	s1 = "I love blue"
	s2= "I like blue" 
	print(rewardfunc(s1,s2))
