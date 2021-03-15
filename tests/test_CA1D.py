import numpy as np
import casim.utils
import casim.CA1D

eca = casim.CA1D.CA1D(3, 54)


def test_init():
    assert eca.k == 3


def test_step():
    next_step = eca.step(np.array([1, 0, 1]))
    assert np.array_equal(next_step, np.array([0, 1, 0]))


def test_initialize_state():
    x = eca.initialize_state(3, p=1.0)
    assert np.sum(x) == 3


def test_set_state():
    eca.set_state(np.array([1, 0, 1]))
    assert np.array_equal(eca.state, np.array([1, 0, 1]))

def test_set_rule():
    eca2 = casim.CA1D.CA1D(3, 54)
    eca2.set_rule(110)
    assert eca2.rule == 110

def test_lambda_rule():
    assert np.sum(casim.utils.to_binary(eca.lambda_rule(2))) == 6


def test_get_state_transition_graph():
    G = eca.get_state_transition_graph(3)
    assert ((5, 2) in G.edges() and (2, 7) in G.edges() and (7, 0) in G.edges()
            and (0, 0) in G.edges())


def test_simulate_time_series():
    known = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1],
                      [0, 0, 0], [0, 0, 0]])
    eca.set_state(np.array([1, 0, 1]))
    eca.simulate_time_series(3, 6)
    assert np.array_equal(eca.history, known)


def test_simulate_entropy_series():
    known = np.array([1.58, 1.58, 0, 0, 0, 0])
    eca.set_state(np.array([1, 0, 1]))
    eca.simulate_entropy_series(3, 6)
    assert np.array_equal(np.round(eca.entropies, 2), known)


def test_find_exact_attractors_fancy():
    period, transient = eca.find_exact_attractor(
        8, 8, np.array([1, 0, 1, 1, 0, 1, 0, 1]))
    assert period == 4 and transient == 2


def test_find_approx_attractors_fancy():
    period, transient = eca.find_approx_attractor(
        8, 8, np.array([1, 0, 1, 1, 0, 1, 0, 1]))
    assert period == 2 and transient == 2


def test_find_exact_attractors_cutoff():
    period, transient = eca.find_exact_attractor(
        12, 6, np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0]))
    assert np.isnan(period) and np.isnan(transient)


def test_find_approx_attractors_cutoff():
    period, transient = eca.find_approx_attractor(
        12, 6, np.array([0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0]))
    assert np.isnan(period) and np.isnan(transient)


def test_find_exact_attractors_immediate():
    period, transient = eca.find_exact_attractor(
        8, 5, np.array([0, 1, 1, 0, 0, 1, 1, 0]))
    assert period == 2 and transient == 0


def test_find_approx_attractors_immediate():
    period, transient = eca.find_approx_attractor(
        8, 5, np.array([0, 1, 1, 0, 0, 1, 1, 0]))
    assert period == 1 and transient == 0


def test_find_exact_attractors_hard():
    eca = casim.CA1D.CA1D(3, 110)
    period, transient = eca.find_exact_attractor(
        16, 200, np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]))
    assert period == 16 and transient == 6


def test_find_approx_attractors_hard():
    eca = casim.CA1D.CA1D(3, 110)
    period, transient = eca.find_approx_attractor(
        16, 200, np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0]))
    assert period == 4 and transient == 6


def test_find_exact_attractor_tricky():
    eca = casim.CA1D.CA1D(3, 110)
    period, transient = eca.find_exact_attractor(
        16, 200, np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]))
    assert period == 16 and transient == 0


def test_find_approx_attractor_tricky():
    eca = casim.CA1D.CA1D(3, 110)
    period, transient = eca.find_approx_attractor(
        16, 200, np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]))
    assert period == 4 and transient == 0
