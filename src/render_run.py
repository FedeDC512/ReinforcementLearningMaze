"""
Render video di un singolo run oppure di N run sovrapposti (overlay).

Modalità overlay (--runs N, N>1):
  Per ogni timestep t il video mostra la posizione corrente di ciascun agente:
  se una run è già terminata, l'agente resta fermo all'ultima posizione.
  I percorsi sono visualizzati come tracce semi-trasparenti (heatmap) +
  pallini colorati per le posizioni correnti.

Compatibilità: con --runs 1 il comportamento è identico al rendering
precedente (singolo agente blu).
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pygame

import config
from env_maze import MazeEnv
from maze_loader import load_maze
from qlearning import QLearningAgent
from simulate import run_episodes

# ── colori base (RGB) ─────────────────────────────────────────────────────
WHITE = (240, 240, 240)
BLACK = (20, 20, 20)
GRAY  = (120, 120, 120)
GREEN = (80, 200, 120)
RED   = (220, 80, 80)
BLUE  = (80, 120, 220)

# palette per gli agenti overlay (RGBA-like, usato per cerchi)
AGENT_COLORS = [
    (80, 120, 220),   # blue
    (220, 80, 80),    # red
    (255, 180, 50),   # orange
    (160, 80, 220),   # purple
    (80, 200, 200),   # cyan
    (200, 200, 60),   # yellow-green
    (220, 120, 180),  # pink
    (100, 180, 100),  # green
    (180, 140, 100),  # brown
    (100, 100, 180),  # slate
]


# ── rendering helpers ─────────────────────────────────────────────────────

def _draw_maze_base(screen, grid, goal, cell):
    """Disegna mappa, muri e goal sulla surface."""
    h, w = grid.shape
    screen.fill(WHITE)
    for r in range(h):
        for c in range(w):
            rect = pygame.Rect(c * cell, r * cell, cell, cell)
            pygame.draw.rect(screen, GRAY, rect, 1)
            if grid[r, c] == 1:
                pygame.draw.rect(screen, BLACK, rect)
    gr, gc = goal
    pygame.draw.rect(screen, GREEN,
                     pygame.Rect(gc * cell, gr * cell, cell, cell))


def _capture_frame(screen):
    """Converte una pygame.Surface in array numpy (H, W, 3) RGB."""
    frame = pygame.surfarray.array3d(screen)   # (W, H, 3)
    return np.transpose(frame, (1, 0, 2)).copy()  # (H, W, 3)


# ── single run (legacy) ──────────────────────────────────────────────────

def render_single_run(Q_path, out_video, policy, epsilon_min,
                      temperature, maze_path, cell=80, fps=6):
    """Rendering di un singolo run (comportamento originale)."""
    name, grid, start, goal, max_steps = load_maze(maze_path)
    env = MazeEnv(grid, start, goal, max_steps=max_steps)

    agent = QLearningAgent(env.n_states, env.n_actions)
    if Q_path is not None:
        agent.Q = np.load(Q_path)

    h, w = grid.shape
    W, H = w * cell, h * cell

    pygame.init()
    screen = pygame.Surface((W, H))
    frames = []

    s = env.reset()
    done = False

    for _ in range(max_steps):
        _draw_maze_base(screen, grid, goal, cell)
        ar, ac = env.pos
        pygame.draw.circle(screen, BLUE,
                           (ac * cell + cell // 2, ar * cell + cell // 2),
                           cell // 3)
        frames.append(_capture_frame(screen))

        if done:
            break
        if Q_path is None:
            a = np.random.randint(4)
        else:
            a = agent.act_eval(s, policy=policy,
                               epsilon_min=epsilon_min,
                               temperature=temperature)
        s, _, done = env.step(a)

    pygame.quit()
    _write_mp4(frames, out_video, W, H, fps)


# ── overlay run ──────────────────────────────────────────────────────────

def render_overlay(Q_path, out_video, n_runs, policy, epsilon_min,
                   temperature, maze_path, cell=80, fps=4, seed=None,
                   save_png=True):
    """Rendering progressivo di N run sovrapposti.

    Per ogni timestep t:
      - disegna la mappa base
      - sovrappone una heatmap cumulativa (quante run hanno visitato
        ciascuna cella fino al passo t)
      - disegna la posizione corrente di ciascun agente come cerchietto
    """
    name, grid, start, goal, max_steps = load_maze(maze_path)
    env = MazeEnv(grid, start, goal, max_steps=max_steps)

    agent = QLearningAgent(env.n_states, env.n_actions)
    if Q_path is not None:
        agent.Q = np.load(Q_path)

    # ── simula tutti gli episodi ──────────────────────────────
    trajectories, rewards, successes = run_episodes(
        env, agent, n_runs, policy,
        epsilon_min, temperature, max_steps, seed=seed)

    # pad trajectories alla stessa lunghezza (ultima pos resta ferma)
    max_len = max(len(t) for t in trajectories)
    for t in trajectories:
        while len(t) < max_len:
            t.append(t[-1])

    h, w = grid.shape
    W, H = w * cell, h * cell

    pygame.init()
    screen = pygame.Surface((W, H))
    heat_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    frames = []

    # heatmap cumulativa (conteggio visite per cella)
    visit_count = np.zeros((h, w), dtype=np.float32)

    # Colore heatmap: arancio semi-trasparente
    HEAT_BASE = np.array([255, 140, 30], dtype=np.float32)

    for t in range(max_len):
        # aggiorna conteggio visite per questo timestep
        for traj in trajectories:
            r, c = traj[t]
            visit_count[r, c] += 1

        # ── disegna maze base ──────────────────────────────────
        _draw_maze_base(screen, grid, goal, cell)

        # ── heatmap overlay ────────────────────────────────────
        heat_surf.fill((0, 0, 0, 0))
        if visit_count.max() > 0:
            norm = visit_count / visit_count.max()
            for r in range(h):
                for c in range(w):
                    if norm[r, c] > 0:
                        alpha = int(30 + 120 * norm[r, c])
                        color = (*HEAT_BASE.astype(int).tolist(), alpha)
                        rect = pygame.Rect(c * cell, r * cell, cell, cell)
                        pygame.draw.rect(heat_surf, color, rect)
        screen.blit(heat_surf, (0, 0))

        # ── posizioni correnti agenti ──────────────────────────
        for i, traj in enumerate(trajectories):
            ar, ac = traj[t]
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            radius = max(cell // 6, 3)
            # offset per evitare sovrapposizione perfetta
            ox = int((i % 5 - 2) * cell * 0.06)
            oy = int((i // 5 - 1) * cell * 0.06)
            cx = ac * cell + cell // 2 + ox
            cy = ar * cell + cell // 2 + oy
            pygame.draw.circle(screen, color, (cx, cy), radius)

        frames.append(_capture_frame(screen))

    # ── hold finale (2 secondi) ────────────────────────────────
    for _ in range(fps * 2):
        frames.append(frames[-1])

    pygame.quit()

    _write_mp4(frames, out_video, W, H, fps)
    print(f"Video overlay salvato: {out_video}  ({n_runs} runs, "
          f"{max_len} steps max)")

    # ── metriche ───────────────────────────────────────────────
    sr = np.mean(successes)
    avg_steps = np.mean([len(t) for t in trajectories])
    avg_rew = np.mean(rewards)
    print(f"  success={sr:.0%}  avg_steps={avg_steps:.1f}  "
          f"avg_reward={avg_rew:.2f}")

    # ── PNG statico finale (percorsi sovrapposti) ──────────────
    if save_png:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        TRAJ_COLORS = list(mcolors.TABLEAU_COLORS.values())
        fig, ax = plt.subplots(
            figsize=(w * cell / 80 + 1, h * cell / 80 + 1))
        for r in range(h):
            for c in range(w):
                fc = "black" if grid[r, c] == 1 else "white"
                ax.add_patch(plt.Rectangle(
                    (c, h - 1 - r), 1, 1,
                    facecolor=fc, edgecolor="gray", linewidth=0.5))
        sr_c, sc_c = start
        gr_c, gc_c = goal
        ax.add_patch(plt.Rectangle(
            (sc_c, h - 1 - sr_c), 1, 1, facecolor="#5cb85c", alpha=0.6))
        ax.add_patch(plt.Rectangle(
            (gc_c, h - 1 - gr_c), 1, 1, facecolor="#d9534f", alpha=0.6))
        ax.text(sc_c + 0.5, h - 1 - sr_c + 0.5, "S",
                ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(gc_c + 0.5, h - 1 - gr_c + 0.5, "G",
                ha="center", va="center", fontsize=8, fontweight="bold")
        for i, traj in enumerate(trajectories):
            color = TRAJ_COLORS[i % len(TRAJ_COLORS)]
            xs = [c + 0.5 + (i - n_runs / 2) * 0.03
                  for (r, c) in traj]
            ys = [h - 1 - r + 0.5 + (i - n_runs / 2) * 0.03
                  for (r, c) in traj]
            ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.6)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_title(f"Overlay  N={n_runs}  sr={np.mean(successes):.0%}",
                     fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        png_path = str(Path(out_video).with_suffix(".png"))
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  PNG salvato: {png_path}")


# ── mp4 writer ────────────────────────────────────────────────────────────

def _write_mp4(frames, path, W, H, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


# ── CLI ───────────────────────────────────────────────────────────────────

def _resolve_q_path(raw: str | None) -> str | None:
    """Risolvi path Q relativo: prova prima il path così com'è, poi CHECKPOINT_DIR."""
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return str(p)
    # prova rispetto a REPO_ROOT
    candidate = config.REPO_ROOT / p
    if candidate.exists():
        return str(candidate)
    # prova dentro CHECKPOINT_DIR (solo nome file)
    candidate2 = config.CHECKPOINT_DIR / p.name
    if candidate2.exists():
        return str(candidate2)
    # restituisci il path originale (farà errore a np.load se non esiste)
    return raw


def parse_args():
    p = argparse.ArgumentParser(
        description="Render video di un run (o overlay di N run)")
    p.add_argument("--q_path", type=str, default=None,
                   help="Path al file Q .npy (None = random)")
    p.add_argument("--out", type=str, default=None,
                   help="Path output mp4 (auto-generato se omesso)")
    p.add_argument("--policy", type=str, default="eps_greedy",
                   choices=["greedy", "eps_greedy", "softmax"])
    p.add_argument("--epsilon_min", type=float, default=config.EPSILON_MIN)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--maze", type=str, default=config.MAZE_PATH)
    p.add_argument("--runs", type=int, default=1,
                   help="Numero di run (>1 attiva overlay automaticamente)")
    p.add_argument("--overlay", action="store_true", default=False,
                   help="Forza modalità overlay anche con runs=1")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed per riproducibilità")
    p.add_argument("--cell", type=int, default=80,
                   help="Dimensione cella in pixel")
    p.add_argument("--fps", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.q_path = _resolve_q_path(args.q_path)
    use_overlay = args.overlay or args.runs > 1

    # auto-genera nome output in RENDER_DIR se non specificato
    render_dir = config.RENDER_DIR
    render_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        stem = Path(args.q_path).stem if args.q_path else "random"
        suffix = f"_overlay_{args.runs}runs" if use_overlay else ""
        args.out = str(render_dir / f"run_{stem}{suffix}.mp4")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if use_overlay:
        render_overlay(
            args.q_path, args.out,
            n_runs=args.runs,
            policy=args.policy,
            epsilon_min=args.epsilon_min,
            temperature=args.temperature,
            maze_path=args.maze,
            cell=args.cell,
            fps=args.fps,
            seed=args.seed,
        )
    else:
        render_single_run(
            args.q_path, args.out,
            policy=args.policy,
            epsilon_min=args.epsilon_min,
            temperature=args.temperature,
            maze_path=args.maze,
            cell=args.cell,
            fps=args.fps,
        )
        print(f"Video salvato: {args.out}")
