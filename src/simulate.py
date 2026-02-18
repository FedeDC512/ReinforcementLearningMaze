"""
Modulo condiviso per simulare episodi su MazeEnv con un QLearningAgent.
Usato da evaluate_checkpoints.py e render_run.py per evitare duplicazione.
"""

import numpy as np
from env_maze import MazeEnv
from qlearning import QLearningAgent


def run_episode(env: MazeEnv, agent: QLearningAgent,
                policy: str, epsilon_min: float,
                temperature: float, max_steps: int):
    """Esegue un episodio e restituisce (trajectory, total_reward, success).

    trajectory: lista di posizioni (r, c) visitate (inclusa la posizione
                iniziale).
    """
    s = env.reset()
    trajectory = [env.pos]
    total_reward = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        a = agent.act_eval(s, policy=policy,
                           epsilon_min=epsilon_min,
                           temperature=temperature)
        s, r, done = env.step(a)
        trajectory.append(env.pos)
        total_reward += r
        steps += 1
    success = (env.pos == env.goal)
    return trajectory, total_reward, success


def run_episodes(env: MazeEnv, agent: QLearningAgent,
                 n_runs: int, policy: str,
                 epsilon_min: float, temperature: float,
                 max_steps: int, seed: int | None = None):
    """Esegue n_runs episodi, restituisce (trajectories, rewards, successes)."""
    if seed is not None:
        np.random.seed(seed)

    trajectories = []
    rewards = []
    successes = []
    for _ in range(n_runs):
        traj, rew, ok = run_episode(env, agent, policy,
                                    epsilon_min, temperature, max_steps)
        trajectories.append(traj)
        rewards.append(rew)
        successes.append(ok)
    return trajectories, rewards, successes
