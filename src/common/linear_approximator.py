import numpy as np


class LinearApproximator:
    """
    Q(s,a) = w_a · phi(s)
    Onde:
        - phi(s) são as features (raw = apenas o vetor de estado)
        - w é um vetor por ação
    """

    def __init__(self, n_actions: int, n_features: int):
        self.n_actions = n_actions
        self.n_features = n_features

        # pesos: uma matriz [n_actions, n_features]
        self.w = np.zeros((n_actions, n_features), dtype=np.float32)

    def features(self, state):
        """Raw features: devolve o estado como vetor 1D."""
        return np.array(state, dtype=np.float32)

    def predict(self, state):
        """Retorna Q(s,a) para todas as ações."""
        phi = self.features(state)
        return self.w @ phi

    def q_value(self, state, action):
        """Retorna Q(s,a) específico."""
        phi = self.features(state)
        return float(self.w[action].dot(phi))

    def update(self, action, td_error, eligibility):
        """
        Atualização: w[a] += alfa * TD * e[a]
        O SARSA(λ) fornece eligibility e TD error.
        """
        self.w[action] += td_error * eligibility[action]
