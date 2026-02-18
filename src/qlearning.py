import numpy as np

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

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])

    '''
    def update(self, s, a, r, s_next):
        max_next = np.max(self.Q[s_next])

        # Formula:
        # Q(s,a) ← Q(s,a)(1-α) + α [ r + γ max Q(s’,a’) - Q(s,a) ]

        self.Q[s, a] = (
            self.Q[s, a] * (1 - self.alpha)
            + self.alpha * (r + self.gamma * max_next - self.Q[s, a])
        )
    '''

    def update(self, s, a, r, s_next, done):
        max_next = 0.0 if done else np.max(self.Q[s_next])

        # Versione equivalente, stabile:
        # Q <- (1-α)Q + α [ r + γ max_next ]
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (r + self.gamma * max_next)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon * self.epsilon_decay)
