import numpy as np
import networkx as nx
from .utils import to_binary, to_decimal
from .calculations import word_entropy


class eca_sim:
    def __init__(self, rule, random_seed=None, float_precision=16):
        """
        This class has methods for simulating elementary cellular automata

        Parameters
        ----------

        rule : int
            specifies the logic for the update rule in Wolfram's notation
        """
        # k is the neighborhood, for eca it is 3
        self.k = 3

        self.rule = rule
        self.update = to_binary(self.rule)

        self.round_digits = float_precision

        if not random_seed:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(random_seed)

        # will get assigned later
        self.state = None

    def step(self, state):
        """ Takes a complete state vector of the system at a time point
        and provides the new value for Centers """
        # calculate the encoding vector based on the rule binary
        # the sum of the element-wise product of this vector and the inputs
        # is the encoded state
        input_size = self.k
        expos = np.arange(input_size, 0, -1) - 1
        enc = 2**expos

        # we need to get new 3-vectors (Left, Center, Right) for each position
        # can use roll to "shift" the whole array left and right on a circle
        # and vstack to set them up as a list of 3-vectors
        vec = np.vstack((np.roll(state, 1), state,
                         np.roll(state, -1))).astype(np.int8)
        encoded = vec.T.dot(enc).astype(np.int8)

        # lookup the index corresponding to the transition
        return self.update[sum(enc) - encoded]

    def initialize_state(self, N, p=0.5):
        """ Initialize state of the system by setting the paobability
        that each cell is in the 1 state """
        self.state = self.rng.choice([0, 1], size=N, p=[1-p, p])

        return True

    def set_state(self, state):
        """ set the system state to a particular value, useful in testing """
        self.state = state

        return True

    def get_state_transition_graph(self, N):
        """ Calculates the complete state transition graph for a specified
        number of cells. Its not going to work
        for even medium numbers of cells maybe like 16?

        Parameters
        ----------
        N : int
            number of cells

        Returns
        -------
        G : nx.DiGraph
            networkx directed graph with nodes as states
        """
        # initialize empty networkx DiGraph
        self.G = nx.DiGraph()

        # for each possible state we decode the integer into a state vector
        # perform the update and encode the next state as an integer label
        for state in range(2**N):
            state_vec = to_binary(state, N)
            next_state = self.step(state_vec)
            dec_next = to_decimal(next_state, N)

            self.G.add_edge(state, dec_next)

        return self.G

    def simulate_time_series(self, N, steps):
        """
        Generate a time series for a specified number of cells and steps. This
        function retains complete state history for summarization elsewhere.

        Parameters
        ----------
        N : int
            number of cells
        steps : int
            number of updates

        Returns
        -------
        history : np.array (steps, N)
            complete state history of the system
        """

        if self.state is None:
            self.initialize_state()

        self.history = np.zeros((steps, N))

        for step in range(steps):
            self.history[step] = self.state
            self.state = self.step(self.state)

        return self.history

    def simulate_entropy_series(self, N, steps, block_size=3):
        """
        Generate a time series for a specified number of cells and steps. This
        function only tracks entropy over time as an approximation for faster
        attractor finding. Default calculates the entropy for blocks of 3 cells

        Parameters
        ----------
        N : int
            number of cells
        steps : int
            number of updates

        Returns
        -------
        history : np.array (steps, N)
            entropy history of the system
        """

        if self.state is None:
            self.initialize_state()

        self.entropies = np.zeros(steps)

        for step in range(steps):
            self.entropies[step] = word_entropy(self.state, block_size)
            self.state = self.step(self.state)

        return self.entropies

    def find_exact_attractor(self, N, steps, state):
        """ this uses the state histories to find the attractor by matching
        exact system states """

        self.set_state(state)
        self.simulate_time_series(N, steps)

        cycle = None  # use it as a flag before assignment
        for i, hist in enumerate(self.history[::-1]):
            if np.array_equal(hist, self.state) and not cycle:
                cycle = self.history[-(i+1):]
                break
        
        if cycle is not None:
            # find the first time each state in the cycle appears in the history
            cycle_hits = []
            for cycle_state in cycle:
                in_cycle = np.sum(cycle_state == self.history, axis=1) == N
                cycle_hits.append(np.argmax(in_cycle))

            self.exact_transient = np.min(cycle_hits)
            self.exact_period = cycle.shape[0]

        else:
            self.exact_period = np.nan
            self.exact_transient = np.nan
        return (self.exact_period, self.exact_transient)

    def find_approx_attractor(self, N, steps, state, block_size=3):
        """ Uses the entropy to find attractor by matching rounded entropy
        values """

        self.set_state(state)
        self.simulate_entropy_series(N, steps, block_size=block_size)

        # round everything so comparisons will work
        end_ent = np.round(word_entropy(self.state, block_size),
                           decimals=self.round_digits)
        entropies = np.round(self.entropies, decimals=self.round_digits)

        # returns idx of frst True
        cycle_match = entropies[::-1] == end_ent
        if np.sum(cycle_match) > 2:
            # cycle_len = np.argmax(cycle_match) + 1
            # (locally) maximize the number of states that repeat in sequence
            cycle_len = None
            for cli in range(1, int(self.entropies.shape[0] / 2)):
                test_cycle = entropies[-cli:]
                previous = entropies[-2*cli:-cli]
                more_previous = entropies[-3*cli:-2*cli]
                if np.array_equal(test_cycle, previous) and np.array_equal(test_cycle, more_previous):
                    cycle_len = cli
                    cycle = entropies[-cli:]
                elif cycle_len is not None:
                    break
                
            # the following line returns the first step not in the cycle
            transient = np.argmin(np.isin(entropies, cycle)[::-1])

            self.approx_period = cycle_len
            # my test for the transient end will return zero if the entire 
            # time series is in the attractor so we need to check for that
            if transient > 1:
                self.approx_transient = steps - transient
            else:
                self.approx_transient = 0

        else:
            self.approx_period = np.nan
            self.approx_transient = np.nan

        return (self.approx_period, self.approx_transient)
