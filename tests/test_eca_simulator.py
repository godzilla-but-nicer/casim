import numpy as np
import casim.eca

eca = casim.eca.eca_sim(54)


def test_init():
    assert eca.k == 3


def test_step():
    next_step = eca.step(np.array([1, 0, 1]))
    assert np.array_equal(next_step, np.array([0, 1, 0]))


def test_initialize_state():
    eca.initialize_state(3, p=1.0)
    assert np.sum(eca.state) == 3


def test_set_state():
    eca.set_state(np.array([1, 0, 1]))
    assert np.array_equal(eca.state, np.array([1, 0, 1]))


def test_get_state_transition_graph():
    G = eca.get_state_transition_graph(3)
    assert ((5, 2) in G.edges() and (2, 7) in G.edges() and (7, 0) in G.edges()
            and (0, 0) in G.edges())


def test_simulate_time_series():
    known = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1],
                      [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    eca.set_state(np.array([1, 0, 1]))
    eca.simulate_time_series(3, 6)
    assert np.array_equal(eca.history, known)


def test_simulate_entropy_series():
    known = np.array([1.58, 1.58, 0, 0, 0, 0])
    eca.set_state(np.array([1, 0, 1]))
    eca.simulate_entropy_series(3, 6)
    assert np.array_equal(np.round(eca.entropies, 2), known)


def test_find_exact_attractors():
    period, transient = eca.find_exact_attractor(8, 8, np.array([1, 0, 1, 1, 0, 1, 0, 1]))
    assert period == 4 and transient == 2


def test_find_approx_attractors():
    period, transient = eca.find_approx_attractor(8, 8, np.array([1, 0, 1, 1, 0, 1, 0, 1]))
    assert period == 2 and transient == 2
