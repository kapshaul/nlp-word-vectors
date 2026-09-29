from collections import Counter 
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np

import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):

		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]


	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """

		# Initialize the lemmatizer
		lemmatizer = WordNetLemmatizer()
		# Convert to lowercase and remove punctuation
		text = text.lower()
		text = text.translate(str.maketrans('', '', string.punctuation))
		# Tokenize the text
		tokens = word_tokenize(text)
		# Lemmatize each word
		lemmatized_tokens = [lemmatizer.lemmatize(word, 'v') for word in tokens]

		return lemmatized_tokens



	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self, corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """

		# Flattening all the texts in the training corpus
		flattened_text = ' '.join(corpus)
		# Mapping the training corpus to tokens
		tokens = self.tokenize(flattened_text)
		# Counting the occurrence of each word
		count_words = Counter(tokens)
		# Use the .most_common() method to get the elements sorted by count
		sorted_count_words = count_words.most_common()
		# Thresh holding heuristics 96%, of usage frequency
		self.entire_freq = {}
		for word, frequency in sorted_count_words:
			self.entire_freq[word] = frequency
		value = list(self.entire_freq.values())
		cumulative_value = np.cumsum(value) / np.sum(value)
		for i in range(len(cumulative_value)):
			if cumulative_value[i] >= 0.96:
				self.thresh_hold = value[i]
				break
		# Build word2idx, idx2word, and freq dictionaries
		word2idx, idx2word, freq = {}, {}, {}
		index = 0
		unk_freq = 0
		for word, frequency in sorted_count_words:
			if frequency >= self.thresh_hold:
				word2idx[word] = index
				idx2word[index] = word
				freq[word] = frequency
				index += 1
			else:
				word2idx['UNK'] = index
				idx2word[index] = 'UNK'
				unk_freq += frequency
				freq['UNK'] = unk_freq

		return word2idx, idx2word, freq



	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    
	    """

		# Summarize plotting data
		value = list(self.entire_freq.values())
		cumulative_value = np.cumsum(value) / np.sum(value)
		index = max(self.word2idx.values())

		# Creating a figure and a set of subplots
		fig, axs = plt.subplots(2)
		# Plotting on the first subplot
		axs[0].plot(value)
		axs[0].set_title('Token Frequency Distribution')
		axs[0].set_xlabel('Token ID (sorted by frequency)')
		axs[0].set_ylabel('Frequency')
		axs[0].set_yscale('log')
		axs[0].axhline(y=self.thresh_hold, color='r', linestyle='-')
		axs[0].text(0.8*len(value), 80, f'freq={self.thresh_hold}', color='r', ha='left', va='center')
		# Plotting on the second subplot
		axs[1].plot(cumulative_value)
		axs[1].set_title('Cumulative Fraction Covered')
		axs[1].set_xlabel('Token ID (sorted by frequency)')
		axs[1].set_ylabel('Fraction of Token occurrences Covered')
		axs[1].axvline(x=index, color='r', linestyle='-')
		axs[1].text(index, 0.96, '0.96', color='r', ha='right', va='center')

		plt.tight_layout()  # Adjust subplots to fit into the figure area.
		plt.show()

