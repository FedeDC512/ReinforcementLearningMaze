import numpy as np
import matplotlib.pyplot as plt
from env_maze import MazeEnv
from qlearning import QLearningAgent
from pathlib import Path
import json
from maze_loader import load_maze
from config import MAZE_PATH

def export_policy_json(Q, path_json):
    policy = {}
    for s in range(Q.shape[0]):
        policy[str(s)] = int(np.argmax(Q[s]))
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

def train(episodes=3000):

    name, grid, start, goal, max_steps = load_maze(MAZE_PATH)
    env = MazeEnv(grid, start, goal, max_steps=max_steps)

    agent = QLearningAgent(
        env.n_states,
        env.n_actions
    )

    Path("outputs").mkdir(exist_ok=True)
    # NOOB: prima di allenare (Q iniziale)
    np.save("outputs/Q_noob.npy", agent.Q)
    export_policy_json(agent.Q, "outputs/policy_noob.json")
    print("Salvato checkpoint: noob")

    checkpoints = {
        2000: "mid",
        episodes: "pro"
    }

    rewards = []
    steps_list = []

    for ep in range(episodes):

        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        Path("outputs").mkdir(exist_ok=True)

        # Salvataggio checkpoint mid / pro
        if (ep + 1) in checkpoints:
            tag = checkpoints[ep + 1]
            np.save(f"outputs/Q_{tag}.npy", agent.Q)
            export_policy_json(agent.Q, f"outputs/policy_{tag}.json")
            print(f"Salvato checkpoint: {tag}")

        rewards.append(total_reward)
        steps_list.append(steps)

        if (ep+1) % 500 == 0:
            print(f"Episode {ep+1}/{episodes} - steps: {steps}")

    Path("outputs").mkdir(exist_ok=True)
    np.save("outputs/Q.npy", agent.Q)

    plt.plot(steps_list)
    plt.xlabel("Episode")
    plt.ylabel("Steps to goal")
    plt.savefig("outputs/steps_curve.png")
    plt.close()

    print("Training completato.")

if __name__ == "__main__":
    train()
