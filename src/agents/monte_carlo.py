from collections import defaultdict

import numpy as np

from src.agents.agent_base import AgentBase


class MonteCarloAgent(AgentBase):
    def __init__(self, n_actions, n0=100, gamma=0.99):
        self.n_actions = n_actions
        self.n0 = n0
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.returns_count = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        """Política ε-greedy"""
        state_key = tuple(state)
        Ns = self.returns_count[state_key].sum()
        epsilon = self.n0 / (self.n0 + Ns)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.Q[state_key]))

    def update(self, episode):
        """
        episode: lista de tuplas [(s0,a0,r0), (s1,a1,r1), ...]
        """
        G = 0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = self.gamma * G + r
            state_key = tuple(s)

            # primeira visita (first-visit MC)
            if not any((np.array_equal(s, x[0]) and a == x[1]) for x in episode[:t]):
                self.returns_count[state_key][a] += 1
                alpha = 1 / self.returns_count[state_key][a]
                self.Q[state_key][a] += alpha * (G - self.Q[state_key][a])
