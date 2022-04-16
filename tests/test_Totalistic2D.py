import numpy as np
import casim.utils
import casim.Totalistic2D
from .Totalistic2D_knowns import gol_glider, gol_glider_next, gol_series
from .Totalistic2D_knowns import dl_glider, dl_glider_next, dl_series

gol = casim.Totalistic2D.GameOfLife(seed=123)
dl = casim.Totalistic2D.DormantLife(seed=123)


# test generic step
def test_gen_step():
    thresh_sets = [[set([0, 1, 2, 4, 5, 6, 7, 8]), set([3])],
                   [set((0, 1, 4, 5, 6, 7, 8)),    set((2, 3))]]
    t1 = casim.Totalistic2D.Totalistic2D(2, thresholds=thresh_sets)
    assert np.array_equal(t1.step(gol_glider), gol_glider_next)


# game of life tests
def test_gol_init():
    assert gol.reproduce == 3


def test_gol_step():
    assert np.array_equal(gol.step(gol_glider), gol_glider_next)


def test_gol_simulate():
    hist = gol.simulate(gol_glider, 1)
    assert np.array_equal(hist[-1], gol_glider_next)


def test_gol_simulate_transients():
    hist, trans = gol.simulate_transients(gol_series[0], 10)
    assert trans == 4


def test_gol_noise():
    gol = casim.Totalistic2D.GameOfLife(noise=1.0,
                                        transitions=np.array([[0.0, 1.0],
                                                              [1.0, 0.0]]),
                                        seed=123)
    filter, states = gol._resolve_noise(gol_glider)
    assert np.sum(states > 0) == 20


# dormant life tests
def test_dl_init():
    assert dl.reproduce == 3


def test_dl_step():
    assert np.array_equal(dl.step(dl_glider), dl_glider_next)


def test_dl_simulate():
    hist = dl.simulate(dl_glider, 1)
    assert np.array_equal(hist[-1], dl_glider_next)


def test_gl_simulate_transients():
    hist, trans = dl.simulate_transients(dl_series[0], 10)

    all_match = True
    for hi, grid in enumerate(hist):
        if not np.array_equal(dl_series[hi], grid):
            all_match = False

    assert trans == 1 and all_match


def test_dl_noise():
    dl = casim.Totalistic2D.DormantLife(noise=1.0,
                                        transitions=np.array([[0., 1.0, 0.0],
                                                              [1., 0.0, 0.0],
                                                              [1., 0.0, 0.0]]),
                                        seed=123)
    filter, states = dl._resolve_noise(dl_glider)
    assert np.sum(states > 0) == 20
