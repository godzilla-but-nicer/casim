import numpy as np
from numpy.core.numeric import array_equal
from scipy import signal


class Totalistic2D:
    def __init__(self, n_states: int, thresholds, seed: int = None):
        """
        This class contains functions for 2D CA models that depend on the
        number of neighbors but not their specific arrangement """

        # set the possible states and thresholds for the states
        self.thresholds = np.zeros(n_states)
        self.states = np.arange(n_states)

        for i in range(self.states.shape[0]):
            self.thresholds[i] = thresholds[i]

        # rng
        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def simulate(self, init_grid, steps: int):
        """
        this method simulates the thing
        """
        if type(init_grid) == int:
            self.grid = self.rng.choice([0, 1], size=[init_grid, init_grid])
        else:
            self.grid = init_grid

        self.history = np.zeros(
            (steps+1, self.grid.shape[0], self.grid.shape[0]))

        for st in range(steps):
            self.history[st] = self.grid
            self.grid = self.step(self.grid)

        self.history[-1] = self.grid

        return self.history

    def simulate_transients(self, init_grid, max_steps: int):
        """
        this method simulates the thing until an attractor is found
        """
        all_history = self.simulate(init_grid, max_steps)

        # find the attractor by looking backwards
        found_attractor = False
        for z, past_state in enumerate(all_history[::-1]):
            if np.array_equal(past_state, all_history[-1]) and z > 0:
                idx = all_history.shape[0] - z
                found_attractor = True
                break
        
        # the way this in indexed should get JUST one of each state in the
        # attractor. It at least works with period 1 attractors. Might not
        # on longer attractors??
        # iterate from the beginning to see where attractor first appears
        if not found_attractor:
            return all_history, np.nan
        else:
            attractor = all_history[idx:]
            for i, past_state in enumerate(all_history):
                for attr_state in attractor:
                    if np.array_equal(past_state, attr_state):
                        return (all_history[:i], i-1)

    def simulate_transients_old(self, init_grid, max_steps):
        """
        this method simulates the thing until an attractor is found
        """
        if type(init_grid) == int:
            self.grid = self.rng.choice([0, 1], size=[init_grid, init_grid])
        else:
            self.grid = init_grid

        # flag to set NaN if attractor is not found
        found_attractor = False

        self.history = np.zeros(
            (max_steps+1, self.grid.shape[0], self.grid.shape[0]))

        for st in range(max_steps):
            self.history[st] = self.grid
            self.grid = self.step(self.grid)

            for hi, prev in enumerate(self.history[:st]):
                if array_equal(prev, self.grid):
                    last_idx_transient = hi - 1
                    found_attractor = True
                    break
            
            # exit loop if we're done with the attractor
            if found_attractor:
                end_step = st + 1
                break

        # if we dont find the attractor the transient index should be NaN
        if not found_attractor:
            end_step = max_steps
            last_idx_transient = np.nan

        return self.history[:end_step], last_idx_transient


class GameOfLife(Totalistic2D):
    def __init__(self, seed: int = None):
        """
        this class simulates conway's game of life
        """
        # set params
        self.survive = {'low': 2, 'high': 3}
        self.reproduce = 3

        # set conv filter
        self.filter = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def step(self, grid):
        """
        this does a single step. we're going to do convolution!!
        """
        neighbors = signal.convolve2d(grid, self.filter,
                                      mode='same', boundary='wrap')
        new_grid = grid.copy()

        # survival
        new_grid[(grid == 1) &
                 ((neighbors < self.survive['low']) |
                  (neighbors > self.survive['high']))] = 0

        # reproduction
        new_grid[(grid == 0) & (neighbors == self.reproduce)] = 1

        return new_grid


class DormantLife(Totalistic2D):
    def __init__(self, seed: int):
        """
        this class implements the three state game of life described in
        Javid 2007
        """
        # set params
        self.survive = {'low': 2, 'high': 3}
        self.reproduce = 3
        self.die = 4

        # three states {alive: 2, dormant: 1, dead: 0}

        # set conv filter
        self.filter = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]])

        if seed:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def step(self, grid):
        """
        single step of the dormancy CA using convolution
        """
        # copy the grid to make a new grid
        new_grid = grid.copy()

        # this mask ensures thats the checks on dormant cells occur correctly
        updated = np.zeros(new_grid.shape, dtype=bool)

        # need 2 arrays to convolve
        dormant = grid == 1
        alive = grid == 2

        # convolve!
        d_neighbors = signal.convolve2d(
            dormant, self.filter, mode='same', boundary='wrap')
        a_neighbors = signal.convolve2d(
            alive, self.filter, mode='same', boundary='wrap')

        # sporulation
        new_grid[(grid == 2) &
                 ((a_neighbors < self.survive['low']) |
                  (a_neighbors > self.survive['high']))] = 1

        # reproduction
        new_grid[(grid == 0) & (a_neighbors == self.reproduce)] = 2

        # dormant dying
        new_grid[(grid == 1) &
                 (d_neighbors + a_neighbors > self.die)] = 0
        updated[(grid == 1) &
                (d_neighbors + a_neighbors > self.die)] = True

        # dormant awakening
        new_grid[(updated == False) &  # flake8 whines but numpy-friendly
                 (grid == 1) &
                 ((d_neighbors >= self.survive['low']) &
                  (d_neighbors <= self.survive['high']))] = 2

        return new_grid
