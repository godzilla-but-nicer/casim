import numpy as np
from casim.calculations import word_entropy

def test_word_entropy():
    test_arr = np.array([1, 0, 0, 1, 1, 0, 1, 0])

    assert np.round(word_entropy(test_arr, 3), decimals=1) == 2.5