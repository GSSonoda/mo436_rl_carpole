from collections import defaultdict

import numpy as np

from src.agents.agent_base import AgentBase


class SarsaLambdaAgent(AgentBase):
    def __init__(self, n_actions, gamma=0.99, N0=100, lam=0.9):
        """
        n_actions: número de ações
        gamma: fator de desconto
        N0: constante para política ε-greedy dependente de N(s)
        lam: lambda para eligibility traces
        """
        self.n_actions = n_actions
        self.gamma = gamma
        self.N0 = N0
        self.lam = lam

        self.Q = defaultdict(lambda: np.zeros(n_actions))
        self.N_sa = defaultdict(
            lambda: np.zeros(n_actions)
        )  # contador de visitas a (s,a)

    def select_action(self, state):
        state_key = tuple(state)
        N_s = self.N_sa[state_key].sum()
        epsilon = self.N0 / (self.N0 + N_s)

        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(np.argmax(self.Q[state_key]))

    def update(self, episode):
        """
        episode: lista de tuplas [(s0,a0,r0), (s1,a1,r1), ...]
        Atualiza Q(s,a) usando SARSA(λ) com eligibility traces
        """
        # Inicializa eligibility traces
        E = defaultdict(lambda: np.zeros(self.n_actions))

        for t in range(len(episode)):
            s, a, r = episode[t]
            state_key = tuple(s)

            # Próximo estado e ação
            if t < len(episode) - 1:
                s_next, a_next, _ = episode[t + 1]
                td_target = r + self.gamma * self.Q[tuple(s_next)][a_next]
            else:
                td_target = r  # último passo do episódio

            td_error = td_target - self.Q[state_key][a]

            # Incrementa contador de (s,a) e step-size α
            self.N_sa[state_key][a] += 1
            alpha = 1 / self.N_sa[state_key][a]

            # Atualiza eligibility trace
            E[state_key][a] += 1

            # Atualiza todos os Q(s,a) com eligibility traces
            for s_k, q_vals in self.Q.items():
                for a_k in range(self.n_actions):
                    self.Q[s_k][a_k] += alpha * td_error * E[s_k][a_k]
                    E[s_k][a_k] *= self.gamma * self.lam
