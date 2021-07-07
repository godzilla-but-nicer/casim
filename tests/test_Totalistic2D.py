import numpy as np
import casim.utils
import casim.Totalistic2D
from .Totalistic2D_knowns import gol_glider, gol_glider_next, gol_series
from .Totalistic2D_knowns import dl_glider, dl_glider_next, dl_series

gol = casim.Totalistic2D.GameOfLife(123)
dl = casim.Totalistic2D.DormantLife(123)


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


def test_dl_init():
    assert dl.reproduce == 3


def test_dl_step():
    assert np.array_equal(dl.step(dl_glider), dl_glider_next)


def test_dl_simulate():
    hist = dl.simulate(dl_glider, 1)
    assert np.array_equal(hist[-1], dl_glider_next)


def test_gl_simulate_transients():
    hist, trans = dl.simulate_transients(dl_series[0], 10)
    assert trans == 5
