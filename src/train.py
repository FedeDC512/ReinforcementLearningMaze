import argparse
import numpy as np
import matplotlib.pyplot as plt
from env_maze import MazeEnv
from qlearning import QLearningAgent
from pathlib import Path
import json
from maze_loader import load_maze
import config


def export_policy_json(Q, path_json):
    policy = {}
    for s in range(Q.shape[0]):
        policy[str(s)] = int(np.argmax(Q[s]))
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)


def save_checkpoint(Q, tag, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(str(out / f"Q_{tag}.npy"), Q)
    export_policy_json(Q, str(out / f"Q_{tag}_policy.json"))
    print(f"  ✓ checkpoint salvato: {tag}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Q-Learning su labirinto")
    p.add_argument("--episodes",       type=int,   default=config.EPISODES)
    p.add_argument("--alpha",          type=float, default=config.ALPHA)
    p.add_argument("--gamma",          type=float, default=config.GAMMA)
    p.add_argument("--epsilon_start",  type=float, default=config.EPSILON_START)
    p.add_argument("--epsilon_min",    type=float, default=config.EPSILON_MIN)
    p.add_argument("--epsilon_decay",  type=float, default=config.EPSILON_DECAY)
    p.add_argument("--policy",         type=str,   default=config.POLICY_MODE,
                   choices=["greedy", "eps_greedy", "softmax"])
    p.add_argument("--temperature",    type=float, default=config.TEMPERATURE)
    p.add_argument("--checkpoint_every", type=int, default=config.CHECKPOINT_EVERY)
    p.add_argument("--maze",           type=str,   default=config.MAZE_PATH)
    p.add_argument("--backtrack_penalty", type=float, default=config.BACKTRACK_PENALTY)
    p.add_argument("--out_dir",        type=str,   default=str(config.CHECKPOINT_DIR))
    return p.parse_args()


def train(args):
    name, grid, start, goal, max_steps = load_maze(args.maze)
    env = MazeEnv(grid, start, goal, max_steps=max_steps,
                  backtrack_penalty=args.backtrack_penalty)

    agent = QLearningAgent(
        env.n_states,
        env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
    )

    ckpt_dir = Path(args.out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint iniziale (ep 0 = "noob")
    save_checkpoint(agent.Q, "ep0000", str(ckpt_dir))

    rewards = []
    steps_list = []
    best_avg = -np.inf

    for ep in range(args.episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done:
            action = agent.act(state, policy=args.policy,
                               temperature=args.temperature)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        rewards.append(total_reward)
        steps_list.append(steps)

        ep1 = ep + 1

        # ── checkpoint periodico ───────────────────────────────
        if ep1 % args.checkpoint_every == 0:
            tag = f"ep{ep1:04d}"
            save_checkpoint(agent.Q, tag, str(ckpt_dir))

        # ── checkpoint "best" (media mobile ultimi 100 ep) ─────
        if len(rewards) >= 100:
            avg = np.mean(rewards[-100:])
            if avg > best_avg:
                best_avg = avg
                save_checkpoint(agent.Q, "best", str(ckpt_dir))

        # ── log ────────────────────────────────────────────────
        if ep1 % 500 == 0:
            print(f"Episode {ep1}/{args.episodes}  steps={steps}  "
                  f"eps={agent.epsilon:.4f}  reward={total_reward:.2f}")

    # Checkpoint finale
    save_checkpoint(agent.Q, f"ep{args.episodes:04d}", str(ckpt_dir))
    np.save(str(ckpt_dir / "Q.npy"), agent.Q)

    # ── curva di training ──────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(steps_list, alpha=0.3, label="steps")
    window = min(100, len(steps_list))
    if window > 1:
        avg_steps = np.convolve(steps_list, np.ones(window)/window,
                                mode="valid")
        plt.plot(range(window-1, len(steps_list)), avg_steps,
                 label=f"media mobile ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Steps to goal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(log_dir / "steps_curve.png"), dpi=150)
    plt.close()

    print(f"\nTraining completato — {args.episodes} episodi, "
          f"policy={args.policy}, epsilon_min={args.epsilon_min}")


if __name__ == "__main__":
    train(parse_args())
