import numpy as np
from src import time_domain_fncs as fcns


def test_COSINEWINDOW_K1():
    L = 3
    known_win = np.array([0,1,0])
    known_hamm = np.array([4/46,1,4/46])
    test_COSWIN = fcns.COSINEWINDOW_K1(3,0.5)
    test_hamm = fcns.HAMMING(3)
    test_WELCH = fcns.WELCH_WINDOW(3)
    test_TRIANGULAR = fcns.BARTLETT(3)
    test_FLAT_TOP = fcns.FLAT_TOP_WINDOW(3)
    assert np.all(known_win==test_COSWIN)
    assert np.all(known_win==test_WELCH)
    assert np.all(known_win==test_TRIANGULAR)
    assert np.allclose(known_hamm,test_hamm) # since it's not a zero window, it's not exactly equal, but for all these windows, it equals [0,1,0] if length is 3


