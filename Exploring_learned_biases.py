import gensim.downloader
w2v = gensim.downloader.load('word2vec-google-news-300')

def analogy(a, b, c):
    print(a+" : "+b+" :: "+c+" : ?")
    print([(w, round(c, 3)) for w, c in w2v.most_similar(positive=[c, b], negative=[a])])
    print()

##################
# Example analogy
##################
analogy('man', 'king', 'woman')

#####################################
# Task 4.1, Find analogies that work
#####################################
analogy('Italy', 'Rome', 'France')
analogy('man', 'actor', 'woman')
analogy('happy', 'joy', 'sad')

############################################
# Task 4.2, Find analogies that do not work
############################################
analogy('fighter', 'fight', 'doctor')
analogy('sun', 'hot', 'snow')
analogy('bird', 'fly', 'fish')

###############
# Example bias
###############
analogy('man', 'doctor', 'woman')
analogy('woman', 'doctor', 'man')
analogy('man', 'victim', 'woman')
analogy('woman', 'victim', 'man')

#############################################################################################
# Task 4.3, Find case of bias based on gender, politics, religion, ethnicity, or nationality.
#############################################################################################
analogy('man', 'marine', 'woman')
analogy('woman', 'marine', 'man')
analogy('man', 'delicate', 'woman')
analogy('woman', 'delicate', 'man')
