import numpy as np
import cv2
import pygame
from env_maze import MazeEnv
from maze_loader import load_maze
from config import MAZE_PATH

# colori semplici (pygame usa RGB)
WHITE = (240, 240, 240)
BLACK = (20, 20, 20)
GRAY  = (120, 120, 120)
GREEN = (80, 200, 120)
RED   = (220, 80, 80)
BLUE  = (80, 120, 220)

def policy_action(Q, s):
    return int(np.argmax(Q[s]))

def render_policy_run(Q_path, out_video, max_steps=200, cell=80, fps=10):
    name, grid, start, goal, max_steps_file = load_maze(MAZE_PATH)
    env = MazeEnv(grid, start, goal, max_steps=max_steps_file)

    Q = None if Q_path is None else np.load(Q_path)

    h, w = grid.shape
    W, H = w * cell, h * cell

    pygame.init()
    screen = pygame.Surface((W, H))  # surface offscreen (no finestra)
    frames = []

    s = env.reset()
    done = False

    for _ in range(max_steps):
        # draw
        screen.fill(WHITE)
        for r in range(h):
            for c in range(w):
                rect = pygame.Rect(c*cell, r*cell, cell, cell)
                pygame.draw.rect(screen, GRAY, rect, 1)
                if grid[r, c] == 1:
                    pygame.draw.rect(screen, BLACK, rect)

        # goal
        gr, gc = goal
        pygame.draw.rect(screen, GREEN, pygame.Rect(gc*cell, gr*cell, cell, cell))

        # agent
        ar, ac = env.pos
        pygame.draw.circle(screen, BLUE, (ac*cell + cell//2, ar*cell + cell//2), cell//3)

        # capture frame (pygame -> numpy)
        frame = pygame.surfarray.array3d(screen)  # (W,H,3)
        frame = np.transpose(frame, (1, 0, 2))    # (H,W,3)
        frames.append(frame.copy())

        if done:
            break
        if Q is None:
            a = np.random.randint(4)   # NOOB random
        else:
            a = policy_action(Q, s)

        s, _, done = env.step(a)

    pygame.quit()

    # salva mp4 con opencv (BGR)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()

if __name__ == "__main__":
    render_policy_run(None, "outputs/run_noob.mp4")
    render_policy_run("outputs/Q_mid.npy", "outputs/run_mid.mp4")
    render_policy_run("outputs/Q_pro.npy", "outputs/run_pro.mp4")
    print("Creati: outputs/run_noob.mp4, run_mid.mp4, run_pro.mp4")
