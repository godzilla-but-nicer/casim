import numpy as np
from scipy.stats import entropy

def word_entropy(state_vector, word_size):
    # stack the array on itself to get the word vectors
    word_vecs = state_vector.copy()
    for si in range(1, word_size):
        word_vecs = np.vstack((word_vecs, np.roll(state_vector, si)))

    # quick way to encode the words as numbers
    encoding = 2**(np.arange(word_size, 0, -1) - 1)
    words = encoding.dot(word_vecs)
    _, word_counts = np.unique(words, return_counts=True)

    # if words are missing unique won't catch them    
    # if word_counts.shape[0] < 2**word_size:
    #     all_words = np.zeros(2**word_size)
    #     all_words[:word_counts.shape[0]] = word_counts
    #     word_counts = all_words

    word_freqs = word_counts / np.sum(word_counts)

    return entropy(word_counts, base=2)
