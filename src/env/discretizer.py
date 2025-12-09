import numpy as np


class Discretizer:
    def __init__(self, bins=(6, 6, 12, 12)):
        self.bins = bins
        self.obs_space_low = np.array([-4.8, -5.0, -0.418, -5.0])
        self.obs_space_high = np.array([4.8, 5.0, 0.418, 5.0])
        self.bin_edges = [
            np.linspace(low, high, num=b + 1)[
                1:-1
            ]  # [1:-1] remove the first and the last
            for low, high, b in zip(self.obs_space_low, self.obs_space_high, bins)
        ]

    def discretize(self, obs):
        return tuple(np.digitize(val, self.bin_edges[i]) for i, val in enumerate(obs))
