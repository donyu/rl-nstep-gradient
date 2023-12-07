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
        self.num_tiles = np.ceil((state_high - state_low) / tile_width) + 1
        # each tiling should start from (low - tiling_index / # tilings * tile width) where the tiling index starts from 0
        self.offsets = [(state_low - (tiling_index / num_tilings) * tile_width) for tiling_index in range(num_tilings)]

    def get_tile_indices(self, s):
        tile_indices = []
        for offset in self.offsets:
            indices = np.floor((s - offset) / self.tile_width)
            tile_indices.append(indices)
        return tile_indices

    def __call__(self,s):
        value = 0
        for tile_indices in self.get_tile_indices(s):
            index = int(np.dot(tile_indices, np.prod(self.num_tiles) ** np.arange(len(self.num_tiles))))
            value += self.weights[index]
        return value / self.num_tilings

    def update(self,alpha,G,s_tau):
        tile_indices_list = self.get_tile_indices(s_tau)
        value_estimate = self(s_tau)
        delta = G - value_estimate
        for tile_indices in tile_indices_list:
            index = int(np.dot(tile_indices, np.prod(self.num_tiles) ** np.arange(len(self.num_tiles))))
            self.weights[index] += alpha * delta / self.num_tilings
