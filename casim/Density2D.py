import numpy as np
from scipy import signal


class GameOfLife:
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


class Density2D:
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
