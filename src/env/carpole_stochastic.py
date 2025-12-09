import gymnasium as gym
import numpy as np

from src.env.discretizer import Discretizer


class CartPoleStochastic(gym.Wrapper):
    def __init__(self, force_std=0.1, bins=(6, 6, 12, 12)):
        self.env = gym.make("CartPole-v1")
        super().__init__(self.env)
        self.base_env = self.env.unwrapped
        self.force_std = force_std
        self.np_random = np.random
        self.discretizer = Discretizer(bins)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.discretizer.discretize(obs), info

    def step(self, action):
        original_force_mag = self.base_env.force_mag
        # add noise on force
        force = original_force_mag if int(action) == 1 else -original_force_mag
        noise = self.np_random.normal(0, self.force_std)
        noisy_force = force + noise
        try:
            self.base_env.force_mag = abs(noisy_force)
            obs, reward, done, truncated, info = super().step(action)
        finally:
            # reset force_mag
            self.base_env.force_mag = original_force_mag

        return self.discretizer.discretize(obs), reward, done, truncated, info
