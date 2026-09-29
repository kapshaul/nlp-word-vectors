# Word Vectors in Natural Language Processing

## Overview

This project delves into the foundational aspects of natural language processing, focusing on the creation and analysis of word vectors, distributed representations of words, and the exploration of inherent biases in these representations. The AG News Benchmark dataset is used for implementing tokenization, vocabulary building, and investigating various techniques for generating and analyzing word vectors.

### 1. Tokenization and Vocabulary Building

The project begins by transforming raw text into tokenized forms, with experimentation on different tokenization methods, including lemmatization. A vocabulary is then built based on the frequency of tokens, using heuristics to optimize the vocabulary size for computational efficiency.

<div align="center">

<img src="Figures/Figure_1.png" alt="Cumulative Regret of UCB" width="600">

**Figure 1**: Token frequency distribution (top) and cumulative fraction covered (bottom)

</div>

<br>

Figure 1 shows the effect of applying a cutoff heuristic where tokens with a frequency of 12 or higher are retained, capturing 96\% of the tokens in the dataset. This threshold was chosen for computational feasibility, as it allows the co-occurrence matrix $C$ to remain approximately 1GB in size. Expanding the vocabulary beyond this point would significantly increase memory requirements, potentially exceeding available resources. The figure illustrates how this cutoff effectively balances the coverage of the dataset with the constraints of computational capacity.

### 2. Frequency-Based Word Vectors

Frequency-based word vectors are explored using *Pointwise Mutual Information (PPMI)*. This involves constructing a co-occurrence matrix from the corpus, computing PPMI values, and then reducing the dimensionality of the word vectors through techniques like Truncated SVD. Visualization of these word vectors is performed using *t-SNE* to better understand the captured semantic relationships.

<div align="center">

<img src="Figures/Figure_2.png" alt="t-SNE Visualization" width="600">

**Figure 2**: t-SNE Visualization

<br>

<img src="Figures/Figure_3.png" alt="t-SNE Visualization" width="300">
<img src="Figures/Figure_4.png" alt="t-SNE Visualization" width="300">
<img src="Figures/Figure_5.png" alt="t-SNE Visualization" width="300">

**Figure 3**: t-SNE clusters — War (left), Technology (middle), and Politics (right)

</div>

<br>

### 3. Learning-Based Word Vectors with GloVe

The GloVe algorithm is implemented to generate word vectors by modeling word co-occurrences as a weighted log-bilinear regression problem. The process includes deriving gradients, optimizing the objective via stochastic gradient descent, and visualizing the resulting word vectors. The behavior of the loss during training is monitored to ensure proper convergence.
The GloVe objective can be written as a sum of weighted squared error terms for each word-pair in a vocabulary,

$$
J = \overbrace{\sum_{i,j  \in V}}^{\mbox{{sum over\\ word pairs}}} \underbrace{f(C_{ij})}_ {\mbox{weight}} ~~~( \overbrace{w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij}}^{\mbox{error term}})^2
$$

where each word $i$ is associated with word vector $w_i$, context vector $\tilde{w}_ i$, and word/context biases $b_i$ and $\tilde{b}_ i$.
The $f(C_{ij})$ term is a weighting to avoid frequent co-occurrences from dominating the objective and is defined as,

$$
f(X_{ij}) = min(1, C_{ij}/100)^{0.75}
$$

The derivation of the gradient for the objective $J$ is expressed as follows,

$\nabla_{w_i}J=\nabla_{w_i}\sum_{i,j  \in V}f(C_{ij})(w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij})^2$

$\hspace{0.75cm}=2{\tilde{w}_ j}f(C_{ij})(w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij})$

$\nabla_{\tilde{w}_ j}J=\nabla_{\tilde{w}_ j}\sum_{i,j  \in V}f(C_{ij})(w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij})^2$

$\hspace{0.75cm}=2w_if(C_{ij})(w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij})$

$\nabla_{b_i}J=\nabla_{b_i}\sum_{i,j  \in V}f(C_{ij})(w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij})^2$

$\hspace{0.75cm}=2f(C_{ij})(w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij})$

$\nabla_{\tilde{b}_ j}J=\nabla_{\tilde{b}_ j}\sum_{i,j  \in V}f(C_{ij})(w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij})^2$

$\hspace{0.75cm}=2f(C_{ij})(w_i^T\tilde{w}_ j + b_i + \tilde{b}_ j - \log C_{ij})$

<br>

Training GloVe vectors involved monitoring the loss function throughout the process. The behavior of the loss during training is detailed below,

```python
2024-04-17 04:09:49 INFO     Iter 14400 / 15227: avg. loss over last 100 batches = 0.046686563985831216
2024-04-17 04:09:49 INFO     Iter 14500 / 15227: avg. loss over last 100 batches = 0.04769956457112328
2024-04-17 04:09:49 INFO     Iter 14600 / 15227: avg. loss over last 100 batches = 0.04687950216720886
2024-04-17 04:09:49 INFO     Iter 14700 / 15227: avg. loss over last 100 batches = 0.04827717854832922
2024-04-17 04:09:49 INFO     Iter 14800 / 15227: avg. loss over last 100 batches = 0.047144581882744535
2024-04-17 04:09:49 INFO     Iter 14900 / 15227: avg. loss over last 100 batches = 0.047903630422071866
2024-04-17 04:09:49 INFO     Iter 15000 / 15227: avg. loss over last 100 batches = 0.04676183418646468
2024-04-17 04:09:49 INFO     Iter 15100 / 15227: avg. loss over last 100 batches = 0.048071157216658514
2024-04-17 04:09:49 INFO     Iter 15200 / 15227: avg. loss over last 100 batches = 0.04732485846561704
```

### 4. Exploring Bias in Word Vectors

A significant focus of this project is the exploration of biases that can be inherent in word vectors. Relationships learned by word2vec are analyzed, revealing how these vectors can reinforce gender, racial, or other societal biases. This highlights the importance of understanding and addressing these biases, particularly in the deployment of NLP models in real-world applications.

The following examples illustrate how word2vec reinforces gender stereotypes in medicine,

```python
>>> analogy('man', 'doctor', 'woman')
    man : doctor :: woman : ?
    [('gynecologist', 0.709), ('nurse', 0.648), ('doctors', 0.647), ('physician', 0.644), ('pediatrician', 0.625), ('nurse_practitioner', 0.622), ('obstetrician', 0.607), ('ob_gyn', 0.599), ('midwife', 0.593), ('dermatologist', 0.574)]

>>> analogy('woman', 'doctor', 'man')
    woman : doctor :: man : ?
    [('physician', 0.646), ('doctors', 0.586), ('surgeon', 0.572), ('dentist', 0.552), ('cardiologist', 0.541), ('neurologist', 0.527), ('neurosurgeon', 0.525), ('urologist', 0.525), ('Doctor', 0.524), ('internist', 0.518)]
```

These results show that word2vec tends to associate female doctors with roles in nursing or specializations focused on women’s or children’s health, thus reinforcing gender stereotypes in the medical field.

---

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/kapshaul/nlp-word-vectors.git
cd nlp-word-vectors
pip install -r requirements.txt
```

## Implementation

1. To implement *Tokenization and Vocabulary Building*, run `build_freq_vectors.py`.
2. To implement *Frequency-Based Word Vectors* and *Learning-Based Word Vectors with GloVe*, run `build_glove_vectors.py`.
3. To implement *Exploring Bias in Word Vectors*, run `Exploring_learned_biases.py`.
