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


def test_init():
    assert gol.reproduce == 3


def test_step():
    assert np.array_equal(gol.step(glider), glider_next)


def test_simulate():
    hist = gol.simulate(glider, 1)
    assert np.array_equal(hist[-1], glider_next)
