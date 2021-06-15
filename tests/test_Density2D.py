import numpy as np
import casim.utils
import casim.Density2D

gol = casim.Density2D.GameOfLife(123)
glider = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]])
glider_next = np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0],
                        [0, 0, 1, 1, 0],
                        [0, 0, 1, 0, 0]])


def test_gol_init():
    assert gol.reproduce == 3


def test_gol_step():
    assert np.array_equal(gol.step(glider), glider_next)


def test_gol_simulate():
    hist = gol.simulate(glider, 1)
    assert np.array_equal(hist[-1], glider_next)


dl = casim.Density2D.DormantLife(123)
dl_glider = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0],
                      [0, 0, 0, 2, 0],
                      [0, 2, 2, 2, 0],
                      [0, 0, 0, 0, 0]])
dl_glider_next = np.array([[0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 2, 0],
                           [0, 1, 2, 2, 0],
                           [0, 0, 2, 0, 0]])


def test_dl_init():
    assert dl.reproduce == 3


def test_dl_step():
    assert np.array_equal(dl.step(dl_glider), dl_glider_next)


def test_dl_simulate():
    hist = dl.simulate(dl_glider, 1)
    assert np.array_equal(hist[-1], dl_glider_next)
