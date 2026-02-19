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
import colorsys
import re as _re
from pathlib import Path

import cv2
import numpy as np
import pygame

import config
from env_maze import MazeEnv
from maze_loader import load_maze
from qlearning import QLearningAgent
from simulate import run_episodes
from logger import setup_logger

# ── costanti ──────────────────────────────────────────────────────────────
MAX_SIDE_PX = 1920  # soglia risoluzione: se il lato maggiore supera questo
                     # valore, cell viene ridotto automaticamente.

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


# ── checkpoint discovery & palette ────────────────────────────────────────

def _discover_checkpoints(checkpoint_dir):
    """Return list of (tag, path) sorted by episode number."""
    pattern = _re.compile(r"^Q_(.+)\.npy$")
    found = []
    for p in sorted(Path(checkpoint_dir).glob("Q_*.npy")):
        m = pattern.match(p.name)
        if not m:
            continue
        tag = m.group(1)
        if tag.endswith("_policy"):
            continue
        ep_match = _re.search(r"ep(\d+)", tag)
        sort_key = int(ep_match.group(1)) if ep_match else 999999
        found.append((sort_key, tag, str(p)))
    found.sort()
    return [(tag, path) for _, tag, path in found]


def _generation_palette(n):
    """N colori distinti: rosso (non addestrato) → blu (addestrato)."""
    colors = []
    for i in range(n):
        hue = 0.0 + 0.65 * i / max(n - 1, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.90)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


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


def _effective_cell(grid_h: int, grid_w: int, cell: int) -> int:
    """Riduce automaticamente cell se la risoluzione supera MAX_SIDE_PX."""
    max_dim = max(grid_h, grid_w)
    if max_dim * cell > MAX_SIDE_PX:
        new_cell = MAX_SIDE_PX // max_dim
        new_cell = max(new_cell, 2)  # mai sotto 2px
        # logged at call site
        return new_cell
    return cell


def _make_writer(path: str, W: int, H: int, fps: int) -> cv2.VideoWriter:
    """Crea un cv2.VideoWriter mp4v."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (W, H))


def _capture_frame(screen):
    """Converte una pygame.Surface in array numpy (H, W, 3) RGB."""
    frame = pygame.surfarray.array3d(screen)   # (W, H, 3)
    return np.transpose(frame, (1, 0, 2)).copy()  # (H, W, 3)


def _write_frame(writer: cv2.VideoWriter, frame_rgb: np.ndarray):
    """Scrive un singolo frame RGB sul writer (converte a BGR)."""
    writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))


# ── single run (legacy) ──────────────────────────────────────────────────

def render_single_run(Q_path, out_video, policy, epsilon_min,
                      temperature, maze_path, cell=80, fps=config.DEFAULT_FPS,
                      max_seconds: int = 180):
    """Rendering di un singolo run — streaming diretto su disco."""
    log = setup_logger("render", policy=policy)
    name, grid, start, goal, max_steps = load_maze(maze_path)
    env = MazeEnv(grid, start, goal, max_steps=max_steps)

    agent = QLearningAgent(env.n_states, env.n_actions)
    if Q_path is not None:
        agent.Q = np.load(Q_path)

    h, w = grid.shape
    cell = _effective_cell(h, w, cell)
    W, H = w * cell, h * cell

    pygame.init()
    screen = pygame.Surface((W, H))
    writer = _make_writer(out_video, W, H, fps)
    last_frame = None
    frames_written = 0

    s = env.reset()
    done = False

    for _ in range(max_steps):
        _draw_maze_base(screen, grid, goal, cell)
        ar, ac = env.pos
        pygame.draw.circle(screen, BLUE,
                           (ac * cell + cell // 2, ar * cell + cell // 2),
                           cell // 3)
        last_frame = _capture_frame(screen)
        # stop if we've already reached the maximum allowed duration
        if max_seconds is not None and (frames_written / fps) >= max_seconds:
            log.info(f"  Interrotto: raggiunto limite durata {max_seconds}s")
            break
        _write_frame(writer, last_frame)
        frames_written += 1

        if done:
            break
        if Q_path is None:
            a = np.random.randint(4)
        else:
            a = agent.act_eval(s, policy=policy,
                               epsilon_min=epsilon_min,
                               temperature=temperature)
        s, _, done = env.step(a)

    writer.release()
    pygame.quit()


# ── overlay run ──────────────────────────────────────────────────────────

def render_overlay(Q_path, out_video, n_runs, policy, epsilon_min,
                   temperature, maze_path, cell=80, fps=config.DEFAULT_FPS, seed=None,
                   save_png=True, max_seconds: int = 180):
    """Rendering progressivo di N run sovrapposti — streaming su disco.

    Per ogni timestep t:
      - disegna la mappa base
      - sovrappone una heatmap cumulativa (quante run hanno visitato
        ciascuna cella fino al passo t)
      - disegna la posizione corrente di ciascun agente come cerchietto

    I frame vengono scritti direttamente nel VideoWriter senza
    accumularli in RAM.
    """
    log = setup_logger("render", policy=policy)
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
    cell = _effective_cell(h, w, cell)
    W, H = w * cell, h * cell

    pygame.init()
    screen = pygame.Surface((W, H))
    heat_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    writer = _make_writer(out_video, W, H, fps)
    last_frame = None
    frames_written = 0

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

        last_frame = _capture_frame(screen)
        # stop if we've already reached the maximum allowed duration
        if max_seconds is not None and (frames_written / fps) >= max_seconds:
            log.info(f"  Interrotto: raggiunto limite durata {max_seconds}s")
            break
        _write_frame(writer, last_frame)
        frames_written += 1

    # ── hold finale (2 secondi), ma non oltre max_seconds ─────
    if last_frame is not None:
        if max_seconds is None:
            hold_frames = fps * 2
        else:
            remaining = max_seconds - (frames_written / fps)
            hold_frames = min(fps * 2, max(0, int(remaining * fps)))
        for _ in range(hold_frames):
            _write_frame(writer, last_frame)
            frames_written += 1

    writer.release()
    pygame.quit()

    print(f"Video overlay salvato: {out_video}  ({n_runs} runs, "
          f"{max_len} steps max)")

    # ── metriche ───────────────────────────────────────────
    sr = np.mean(successes)
    avg_steps = np.mean([len(t) for t in trajectories])
    avg_rew = np.mean(rewards)
    log.info(f"  success={sr:.0%}  avg_steps={avg_steps:.1f}  "
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


# ── generations render ────────────────────────────────────────────────────

def render_generations(checkpoint_dir, out_video, runs_per_gen, policy,
                       epsilon_min, temperature, maze_path,
                       cell=80, fps=config.DEFAULT_FPS, seed=None,
                       save_png=True, max_seconds=180):
    """Video unico con tutti i checkpoint: N agenti per generazione, colori distinti.

    Per ogni timestep t:
      - disegna la mappa base
      - disegna le posizioni correnti di ogni agente, colorato in base
        alla generazione (checkpoint) da cui proviene
      - mostra una legenda con il nome di ogni checkpoint e il colore
    """
    log = setup_logger("render", policy=policy)
    name, grid, start, goal, max_steps = load_maze(maze_path)
    env = MazeEnv(grid, start, goal, max_steps=max_steps)

    checkpoints = _discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        log.info(f"Nessun checkpoint trovato in {checkpoint_dir}")
        return

    n_gen = len(checkpoints)
    palette = _generation_palette(n_gen)

    log.info(f"Render generazioni: {n_gen} checkpoint × {runs_per_gen} agenti")

    if seed is not None:
        np.random.seed(seed)

    # ── simula episodi per ogni checkpoint ─────────────────────
    gen_data = []
    all_trajs = []  # (gen_idx, agent_idx, trajectory)

    for gi, (tag, qpath) in enumerate(checkpoints):
        agent = QLearningAgent(env.n_states, env.n_actions)
        agent.Q = np.load(qpath)

        trajs, rewards, successes = run_episodes(
            env, agent, runs_per_gen, policy,
            epsilon_min, temperature, max_steps)

        sr = np.mean(successes)
        avg_rew = np.mean(rewards)
        gen_data.append({"tag": tag, "sr": sr, "avg_reward": avg_rew})
        log.info(f"  {tag:>10s}  success={sr:.0%}  avg_reward={avg_rew:.2f}")

        for ai, t in enumerate(trajs):
            all_trajs.append((gi, ai, t))

    # ── pad alla stessa lunghezza ──────────────────────────────
    max_len = max(len(t) for _, _, t in all_trajs)
    for _, _, t in all_trajs:
        while len(t) < max_len:
            t.append(t[-1])

    # ── setup video ────────────────────────────────────────────
    h, w = grid.shape
    cell = _effective_cell(h, w, cell)
    W, H = w * cell, h * cell

    pygame.init()
    screen = pygame.Surface((W, H))
    heat_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    writer = _make_writer(out_video, W, H, fps)
    last_frame = None
    frames_written = 0

    # heatmap cumulativa (conteggio visite per cella)
    visit_count = np.zeros((h, w), dtype=np.float32)
    HEAT_BASE = np.array([255, 140, 30], dtype=np.float32)

    for t_step in range(max_len):
        # aggiorna conteggio visite per questo timestep
        for _, _, traj in all_trajs:
            r, c = traj[t_step]
            visit_count[r, c] += 1

        _draw_maze_base(screen, grid, goal, cell)

        # ── heatmap overlay ────────────────────────────────────
        heat_surf.fill((0, 0, 0, 0))
        if visit_count.max() > 0:
            norm = visit_count / visit_count.max()
            for r in range(h):
                for c in range(w):
                    if norm[r, c] > 0:
                        alpha = int(30 + 120 * norm[r, c])
                        hcolor = (*HEAT_BASE.astype(int).tolist(), alpha)
                        rect = pygame.Rect(c * cell, r * cell, cell, cell)
                        pygame.draw.rect(heat_surf, hcolor, rect)
        screen.blit(heat_surf, (0, 0))

        # ── agenti ─────────────────────────────────────────────
        for gi, ai, traj in all_trajs:
            ar, ac = traj[t_step]
            color = palette[gi]
            radius = max(cell // 7, 2)
            # offset per evitare sovrapposizione perfetta
            ox = int((ai % 3 - 1) * cell * 0.10)
            oy = int((ai // 3 - 0.5) * cell * 0.10)
            cx = ac * cell + cell // 2 + ox
            cy = ar * cell + cell // 2 + oy
            pygame.draw.circle(screen, color, (cx, cy), radius)

        last_frame = _capture_frame(screen)
        if max_seconds is not None and (frames_written / fps) >= max_seconds:
            log.info(f"  Interrotto: raggiunto limite durata {max_seconds}s")
            break
        _write_frame(writer, last_frame)
        frames_written += 1

    # ── hold finale ────────────────────────────────────────────
    if last_frame is not None:
        if max_seconds is None:
            hold_frames = fps * 2
        else:
            remaining = max_seconds - (frames_written / fps)
            hold_frames = min(fps * 2, max(0, int(remaining * fps)))
        for _ in range(hold_frames):
            _write_frame(writer, last_frame)
            frames_written += 1

    writer.release()
    pygame.quit()

    print(f"\nVideo generazioni salvato: {out_video}")
    print(f"  {n_gen} checkpoint × {runs_per_gen} agenti = "
          f"{n_gen * runs_per_gen} agenti totali, {max_len} steps max")

    # ── PNG statico (percorsi sovrapposti per generazione) ─────
    if save_png:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(w * cell / 80 + 2, h * cell / 80 + 1))
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

        for gi, (tag, _) in enumerate(checkpoints):
            rgb = palette[gi]
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            for ai in range(runs_per_gen):
                idx = gi * runs_per_gen + ai
                _, _, traj = all_trajs[idx]
                offset = (ai - runs_per_gen / 2) * 0.03
                xs = [c + 0.5 + offset for (r, c) in traj]
                ys = [h - 1 - r + 0.5 + offset for (r, c) in traj]
                label = tag if ai == 0 else None
                ax.plot(xs, ys, color=hex_color, linewidth=0.8,
                        alpha=0.5, label=label)

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_title(f"Generazioni ({n_gen} ckpt × {runs_per_gen} agenti)",
                     fontsize=10)
        ax.axis("off")
        plt.tight_layout()
        png_path = str(Path(out_video).with_suffix(".png"))
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  PNG salvato: {png_path}")

    # ── Legenda separata ───────────────────────────────────────
    if save_png:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig_leg, ax_leg = plt.subplots(
            figsize=(3, 0.35 * n_gen + 0.6))
        fig_leg.patch.set_alpha(0.0)
        ax_leg.axis("off")
        handles = []
        for gi, (tag, _) in enumerate(checkpoints):
            rgb = palette[gi]
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            sr = gen_data[gi]["sr"]
            lbl = f"{tag}  (sr {sr:.0%})"
            handles.append(mpatches.Patch(color=hex_color, label=lbl))
        ax_leg.legend(handles=handles, loc="center", fontsize=9,
                      frameon=True, edgecolor="gray", fancybox=True,
                      facecolor="white",
                      title="Checkpoint / Generazione", title_fontsize=10)
        plt.tight_layout()
        legend_path = str(
            Path(out_video).with_name(
                Path(out_video).stem + "_legend.png"))
        plt.savefig(legend_path, dpi=150, bbox_inches="tight",
                    transparent=True)
        plt.close()
        print(f"  Legenda salvata: {legend_path}")


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
    p.add_argument("--generations", action="store_true", default=False,
                   help="Modalità generazioni: renderizza tutti i checkpoint")
    p.add_argument("--checkpoint_dir", type=str,
                   default=str(config.CHECKPOINT_DIR),
                   help="Cartella checkpoint (usata con --generations)")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed per riproducibilità")
    p.add_argument("--cell", type=int, default=80,
                   help="Dimensione cella in pixel")
    p.add_argument("--fps", type=int, default=config.DEFAULT_FPS)
    p.add_argument("--max_seconds", type=int, default=180,
                   help="Durata massima del video in secondi (es. 180 = 3 minuti)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.q_path = _resolve_q_path(args.q_path)
    use_overlay = args.overlay or args.runs > 1

    # auto-genera nome output in RENDER_DIR se non specificato
    render_dir = config.RENDER_DIR
    render_dir.mkdir(parents=True, exist_ok=True)

    if args.generations:
        # ── modalità generazioni ───────────────────────────────
        if args.out is None:
            args.out = str(render_dir /
                          f"run_generations_{args.policy}_{args.runs}each.mp4")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        render_generations(
            checkpoint_dir=args.checkpoint_dir,
            out_video=args.out,
            runs_per_gen=args.runs,
            policy=args.policy,
            epsilon_min=args.epsilon_min,
            temperature=args.temperature,
            maze_path=args.maze,
            cell=args.cell,
            fps=args.fps,
            max_seconds=args.max_seconds,
            seed=args.seed,
        )
    elif use_overlay:
        if args.out is None:
            stem = Path(args.q_path).stem if args.q_path else "random"
            args.out = str(render_dir /
                          f"run_{stem}_{args.policy}_overlay_{args.runs}runs.mp4")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        render_overlay(
            args.q_path, args.out,
            n_runs=args.runs,
            policy=args.policy,
            epsilon_min=args.epsilon_min,
            temperature=args.temperature,
            maze_path=args.maze,
            cell=args.cell,
            fps=args.fps,
            max_seconds=args.max_seconds,
            seed=args.seed,
        )
    else:
        if args.out is None:
            stem = Path(args.q_path).stem if args.q_path else "random"
            args.out = str(render_dir / f"run_{stem}_{args.policy}.mp4")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        render_single_run(
            args.q_path, args.out,
            policy=args.policy,
            epsilon_min=args.epsilon_min,
            temperature=args.temperature,
            maze_path=args.maze,
            cell=args.cell,
            fps=args.fps,
            max_seconds=None,
        )
        print(f"Video salvato: {args.out}")
