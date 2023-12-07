import numpy as np
from algo import ValueFunctionWithApproximation

class ValueFunctionWithTile(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width

        # number of tiles in each dimension should be: ceil[(high-low)/tile_width] + 1
        self.num_tiles = np.ceil((state_high - state_low) / tile_width).astype(int) + 1
        # each tiling should start from (low - tiling_index / # tilings * tile width) where the tiling index starts from 0
        self.offsets = [(state_low - (tiling_index / num_tilings) * tile_width) for tiling_index in range(num_tilings)]
        # Initialize weights for each tile in each tiling
        self.weights = np.zeros((num_tilings, *self.num_tiles))

    def get_tile_indices(self, s):
        indices = []
        # iterate over each tiling by index and offset
        for tiling_index, offset in enumerate(self.offsets):
            tile_index = np.floor((s - offset) / self.tile_width).astype(int)
            indices.append(tuple(tile_index.astype(int)))
        return indices

    def __call__(self,s):
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        tile_indices = self.get_tile_indices(s)
        # return the sum of the weights of the tiles that the state falls into
        return np.sum([self.weights[t][i] for t, i in enumerate(tile_indices)])

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        tile_indices = self.get_tile_indices(s_tau)
        value_estimate = self(s_tau)

        # Update weights for the tiles of the target state
        for t, i in enumerate(tile_indices):
            # w <- w + alpha [
            # Gradient \nabla\hat{v}(s_tau;w) is 1 for active tiles so we can ignore it
            self.weights[t][i] += alpha * (G - value_estimate)
