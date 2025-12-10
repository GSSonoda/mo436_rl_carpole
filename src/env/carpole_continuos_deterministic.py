import gymnasium as gym
import numpy as np


class CartPoleContinuosDeterministic(gym.Wrapper):
    def __init__(self, seed: int = 10):
        self.env = gym.make("CartPole-v1")
        super().__init__(self.env)

        # fix de Random Number Generation
        self.seed_value = seed
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, **kwargs):
        super().reset(seed=self.seed_value)

        # deterministic small-offset start
        # overrides Gym's random initialization.
        state = np.array(
            [
                0.01,  # position (x)
                0.0,  # speed (x')
                0.01,  # angle (theta)
                0.0,  # anglar speed (theta')
            ],
            dtype=np.float32,
        )

        self.env.unwrapped.state = state.copy()
        return state, {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, reward, done, truncated, info
