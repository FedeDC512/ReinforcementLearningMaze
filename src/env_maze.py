import numpy as np

ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}

# Mappa azione → azione opposta (per penalty backtracking)
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


class MazeEnv:
    def __init__(self, grid, start, goal, max_steps=200,
                 backtrack_penalty=0.0):
        self.grid = np.array(grid)
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
        self.backtrack_penalty = backtrack_penalty
        self.h, self.w = self.grid.shape
        self.reset()

    def state_id(self, pos):
        r, c = pos
        return r * self.w + c

    @property
    def n_states(self):
        return self.h * self.w

    @property
    def n_actions(self):
        return 4

    def reset(self):
        self.pos = self.start
        self.steps = 0
        self.last_action = None
        return self.state_id(self.pos)

    def step(self, action):
        self.steps += 1

        dr, dc = ACTIONS[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc

        reward = -0.01
        done = False

        # Penalty per backtracking (azione opposta alla precedente)
        if (self.backtrack_penalty != 0.0
                and self.last_action is not None
                and action == OPPOSITE[self.last_action]):
            reward += self.backtrack_penalty

        if (nr < 0 or nr >= self.h or
            nc < 0 or nc >= self.w or
            self.grid[nr, nc] == 1):
            reward -= 0.1
            nr, nc = r, c

        self.pos = (nr, nc)
        self.last_action = action

        if self.pos == self.goal:
            reward += 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self.state_id(self.pos), reward, done
