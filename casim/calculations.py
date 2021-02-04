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

    return entropy(word_counts, base=2)
