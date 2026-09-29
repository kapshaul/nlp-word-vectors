import os
from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE

import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
	pass


###########################
## TASK 2.2              ##
###########################

def compute_cooccurrence_matrix(corpus, vocab):
	"""
	    
	    compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N count matrix as described in the handout. It is up to the student to define the context of a word

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus between token i and j in the vocabulary

	    """ 

	if os.path.exists('C.npy'):
		C = np.load('C.npy')
	else:
		N = len(vocab.word2idx)
		C = np.zeros((N, N), dtype=int)
		# Count the number of contexts
		for context in corpus:
			# For each text string, implement text2idx
			text2idx = vocab.text2idx(context)
			# Increment counts in the co-occurrence matrix
			for word1, index1 in enumerate(text2idx):
				for word2, index2 in enumerate(text2idx):
					C[index1, index2] += 1
		np.save('C.npy', C)

	return C

	

###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
	"""
	    
	    compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function. 

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

	    """

	# Obtain co-occurrence matrix
	C = compute_cooccurrence_matrix(corpus, vocab)
	# Calculate the sum of all co-occurrences
	N = np.sum(C)

	###########################
	# Type 1
	# Compute for PPMI
	#vocab_size = len(C)
	#PMI = np.zeros((vocab_size, vocab_size), dtype=float)
	#for i in range(vocab_size):
	#	for j in range(vocab_size):
	#		PMI[i, j] = (C[i, j] * N) / (C[i, i] * C[j, j])
	# Calculate PMI
	#PMI = np.log(PMI + 1e-8)
	# Replace negative PMI values with 0 for PPMI
	#PPMI = np.maximum(0, PMI)

	###########################
	# Type 2
	# Compute the probability of each word
	p_x = np.sum(C, axis=1) / N
	p_y = np.sum(C, axis=0) / N
	# Compute the joint probability matrix for each word pair
	p_xy = C / N
	# Compute PMI, using broadcasting for p(x)*p(y)
	PMI = np.log((p_xy / (p_x * p_y)) + 1e-8)
	# Replace negative PMI values with 0 for PPMI
	PPMI = np.maximum(PMI, 0)

	return PPMI

	

################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():

	logging.info("Loading dataset")
	dataset = load_dataset("ag_news")
	dataset_text =  [r['text'] for r in dataset['train']]
	dataset_labels = [r['label'] for r in dataset['train']]


	logging.info("Building vocabulary")
	vocab = Vocabulary(dataset_text)
	vocab.make_vocab_charts()
	plt.close()
	plt.pause(0.01)


	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix( [doc['text'] for doc in dataset['train']], vocab)


	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)


	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab)


def dim_reduce(PPMI, k=16):
	U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
	SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

	U = U*SqrtSigma
	V = VT.T*SqrtSigma

	word_vectors = np.concatenate( (U, V), axis=1) 
	word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab):
	coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)

	plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=0.5, markersize=3)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=5)
	plt.show()


if __name__ == "__main__":
    main_freq()

