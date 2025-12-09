import numpy as np

from src.agents.agent_base import AgentBase


class QLearningAgent(AgentBase):
    """
    Q-Learning tabular para ambientes discretizados
    """

    def __init__(self, n_states, n_actions, gamma=0.99, n0=100):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.n0 = n0

        # Q-table inicializada em zero
        self.Q = np.zeros(n_states + (n_actions,), dtype=np.float32)

        self.N_s = np.zeros(n_states, dtype=np.int32)  # visitas por estado
        self.N_sa = np.zeros(
            n_states + (n_actions,), dtype=np.int32
        )  # visitas estado-ação

    def select_action(self, state):
        """
        Política ε-greedy dependente de N(s):
        ε = N0 / (N0 + N(s))
        """
        state_idx = tuple(state)
        self.N_s[state_idx] += 1
        epsilon = self.n0 / (self.n0 + self.N_s[state_idx])

        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[state_idx])

    def update(self, state, action, reward, next_state, done):
        state_idx = tuple(state)
        next_state_idx = tuple(next_state)

        self.N_sa[state_idx + (action,)] += 1
        alpha = 1.0 / self.N_sa[state_idx + (action,)]  # step-size adaptativo

        # Q-learning update (off-policy)
        td_target = reward + self.gamma * np.max(self.Q[next_state_idx]) * (not done)
        td_error = td_target - self.Q[state_idx + (action,)]
        self.Q[state_idx + (action,)] += alpha * td_error
