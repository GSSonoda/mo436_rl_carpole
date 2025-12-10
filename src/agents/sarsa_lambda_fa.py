import numpy as np

from src.common.linear_approximator import LinearApproximator


class SarsaLambdaFA:
    def __init__(
        self, n_actions, n_features, alpha=0.05, gamma=0.99, lam=0.9, epsilon=0.1
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon

        self.n_actions = n_actions
        self.approximator = LinearApproximator(n_actions, n_features)

        # eligibility traces (uma entrada por ação)
        self.e = np.zeros((n_actions, n_features), dtype=np.float32)

    def select_action(self, state):
        """Epsilon-greedy sobre Q(s,a)."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        qvals = self.approximator.predict(state)
        return int(np.argmax(qvals))

    def reset_traces(self):
        self.e[...] = 0.0

    def update(self, state, action, reward, next_state, next_action, done):
        """
        Atualização SARSA(λ):
        TD = r + γ Q(s',a') - Q(s,a)
        e[a] = γλ e[a] + φ(s)
        w[a] += α TD e[a]
        """
        phi = self.approximator.features(state)
        next_q = 0 if done else self.approximator.q_value(next_state, next_action)
        current_q = self.approximator.q_value(state, action)

        td_error = reward + self.gamma * next_q - current_q

        # atualiza eligibility
        self.e *= self.gamma * self.lam
        self.e[action] += phi

        # gradiente por ação
        for a in range(self.n_actions):
            self.approximator.w[a] += self.alpha * td_error * self.e[a]
