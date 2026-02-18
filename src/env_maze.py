import numpy as np

ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1),   # RIGHT
}

class MazeEnv:
    def __init__(self, grid, start, goal, max_steps=200):
        self.grid = np.array(grid)
        self.start = start
        self.goal = goal
        self.max_steps = max_steps
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
        return self.state_id(self.pos)

    def step(self, action):
        self.steps += 1

        dr, dc = ACTIONS[action]
        r, c = self.pos
        nr, nc = r + dr, c + dc

        reward = -0.01
        done = False

        if (nr < 0 or nr >= self.h or
            nc < 0 or nc >= self.w or
            self.grid[nr, nc] == 1):
            reward -= 0.1
            nr, nc = r, c

        self.pos = (nr, nc)

        if self.pos == self.goal:
            reward += 1.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        return self.state_id(self.pos), reward, done
