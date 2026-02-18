import numpy as np


def _softmax(logits):
    """Softmax numerico stabile (senza dipendenza da scipy)."""
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / e.sum()


class QLearningAgent:
    def __init__(self, n_states, n_actions,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):

        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions))

    # ── action selection ───────────────────────────────────────

    def act(self, state, policy="eps_greedy", temperature=0.5):
        """Seleziona azione secondo la policy richiesta."""
        if policy == "greedy":
            return int(np.argmax(self.Q[state]))
        elif policy == "softmax":
            return self._act_softmax(state, temperature)
        else:  # eps_greedy (default)
            if np.random.rand() < self.epsilon:
                return np.random.randint(self.n_actions)
            return int(np.argmax(self.Q[state]))

    def _act_softmax(self, state, temperature=0.5):
        """Softmax stocastico: prob(a|s) = softmax(Q(s,·)/τ)."""
        temperature = max(temperature, 1e-8)
        logits = self.Q[state] / temperature
        probs = _softmax(logits)
        return int(np.random.choice(self.n_actions, p=probs))

    def act_eval(self, state, policy="eps_greedy",
                 epsilon_min=0.05, temperature=0.5):
        """Azione per valutazione (epsilon fissato a epsilon_min)."""
        if policy == "greedy":
            return int(np.argmax(self.Q[state]))
        elif policy == "softmax":
            return self._act_softmax(state, temperature)
        else:  # eps_greedy con epsilon = epsilon_min
            if np.random.rand() < epsilon_min:
                return np.random.randint(self.n_actions)
            return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        max_next = 0.0 if done else np.max(self.Q[s_next])

        # Versione equivalente, stabile:
        # Q <- (1-α)Q + α [ r + γ max_next ]
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * max_next)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)
