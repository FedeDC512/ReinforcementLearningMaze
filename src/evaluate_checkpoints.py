"""
Evaluate all saved checkpoints: run N episodes with stochastic policy,
collect trajectories, compute metrics, and generate visualisation PNGs.

Usage examples:
    python evaluate_checkpoints.py
    python evaluate_checkpoints.py --policy softmax --temperature 0.3 --eval_runs 30
    python evaluate_checkpoints.py --policy eps_greedy --epsilon_min 0.1
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

import config
from env_maze import MazeEnv
from maze_loader import load_maze
from qlearning import QLearningAgent
from simulate import run_episode, run_episodes
from logger import setup_logger


# ── helpers ────────────────────────────────────────────────────────────────

def discover_checkpoints(out_dir=None):
    """Return list of (tag, path) sorted by episode number."""
    out_dir = Path(out_dir) if out_dir else config.CHECKPOINT_DIR
    pattern = re.compile(r"^Q_(.+)\.npy$")
    found = []
    for p in sorted(Path(out_dir).glob("Q_*.npy")):
        m = pattern.match(p.name)
        if not m:
            continue
        tag = m.group(1)
        if tag.endswith("_policy"):
            continue
        # extract episode number for sorting (best/noob go last)
        ep_match = re.search(r"ep(\d+)", tag)
        sort_key = int(ep_match.group(1)) if ep_match else 999999
        found.append((sort_key, tag, str(p)))
    found.sort()
    return [(tag, path) for _, tag, path in found]


# ── visualisation ──────────────────────────────────────────────────────────

TRAJECTORY_COLORS = list(mcolors.TABLEAU_COLORS.values())


def draw_maze_with_paths(grid, start, goal, trajectories, title,
                         save_path, cell=40):
    """Draw maze grid and overlay N trajectories with different colours."""
    h, w = grid.shape
    fig, ax = plt.subplots(figsize=(w * cell / 80 + 1, h * cell / 80 + 1))

    # maze
    for r in range(h):
        for c in range(w):
            color = "black" if grid[r, c] == 1 else "white"
            ax.add_patch(plt.Rectangle((c, h - 1 - r), 1, 1,
                                       facecolor=color, edgecolor="gray",
                                       linewidth=0.5))
    # start / goal
    sr, sc = start
    gr, gc = goal
    ax.add_patch(plt.Rectangle((sc, h - 1 - sr), 1, 1,
                                facecolor="#5cb85c", alpha=0.6))
    ax.add_patch(plt.Rectangle((gc, h - 1 - gr), 1, 1,
                                facecolor="#d9534f", alpha=0.6))
    ax.text(sc + 0.5, h - 1 - sr + 0.5, "S", ha="center", va="center",
            fontsize=8, fontweight="bold")
    ax.text(gc + 0.5, h - 1 - gr + 0.5, "G", ha="center", va="center",
            fontsize=8, fontweight="bold")

    # trajectories
    for i, traj in enumerate(trajectories):
        color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
        xs = [c + 0.5 for (r, c) in traj]
        ys = [h - 1 - r + 0.5 for (r, c) in traj]
        # small random offset to avoid perfect overlap
        offset = (i - len(trajectories) / 2) * 0.04
        xs = [x + offset for x in xs]
        ys = [y + offset for y in ys]
        ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.7)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_summary_table(results, save_path):
    """Bar chart with success rate, avg steps, avg reward per checkpoint."""
    tags = [r["tag"] for r in results]
    success = [r["success_rate"] for r in results]
    avg_steps = [r["avg_steps"] for r in results]
    avg_rew = [r["avg_reward"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].bar(tags, success, color="#5cb85c")
    axes[0].set_title("Success rate")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(tags, avg_steps, color="#428bca")
    axes[1].set_title("Avg steps")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(tags, avg_rew, color="#f0ad4e")
    axes[2].set_title("Avg reward")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── main ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Valuta checkpoint Q-learning")
    p.add_argument("--policy",       type=str,   default=config.POLICY_MODE,
                   choices=["greedy", "eps_greedy", "softmax"])
    p.add_argument("--epsilon_min",  type=float, default=config.EPSILON_MIN)
    p.add_argument("--temperature",  type=float, default=config.TEMPERATURE)
    p.add_argument("--eval_runs",    type=int,   default=config.EVAL_RUNS)
    p.add_argument("--maze",         type=str,   default=config.MAZE_PATH)
    p.add_argument("--checkpoint_dir", type=str, default=str(config.CHECKPOINT_DIR))
    p.add_argument("--eval_dir",     type=str,   default=str(config.EVAL_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    log = setup_logger("eval", policy=args.policy)

    name, grid, start, goal, max_steps = load_maze(args.maze)
    env = MazeEnv(grid, start, goal, max_steps=max_steps)

    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if not checkpoints:
        log.info("Nessun checkpoint trovato in %s", args.checkpoint_dir)
        return

    log.info(f"Trovati {len(checkpoints)} checkpoint — "
             f"policy={args.policy}, eval_runs={args.eval_runs}")

    eval_dir = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for tag, qpath in checkpoints:
        Q = np.load(qpath)
        agent = QLearningAgent(env.n_states, env.n_actions)
        agent.Q = Q

        trajectories, rewards_list, successes = run_episodes(
            env, agent, args.eval_runs, args.policy,
            args.epsilon_min, args.temperature, max_steps)

        sr = np.mean(successes)
        avg_steps = np.mean([len(t) for t in trajectories])
        avg_rew = np.mean(rewards_list)

        result = {
            "tag": tag,
            "success_rate": sr,
            "avg_steps": avg_steps,
            "avg_reward": avg_rew,
        }
        all_results.append(result)

        log.info(f"  {tag:>10s}  success={sr:.0%}  "
                 f"avg_steps={avg_steps:.1f}  avg_reward={avg_rew:.2f}")

        # PNG con percorsi sovrapposti
        draw_maze_with_paths(
            grid, start, goal, trajectories,
            title=f"{tag}  (N={args.eval_runs}, sr={sr:.0%}, {args.policy})",
            save_path=str(eval_dir / f"paths_{tag}_{args.policy}.png"))

    # Summary chart
    generate_summary_table(all_results,
                           str(eval_dir / f"summary_metrics_{args.policy}.png"))
    log.info(f"\nImmagini salvate in {eval_dir}/")


if __name__ == "__main__":
    main()
