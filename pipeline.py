"""
Pipeline completa: training → evaluation → render single → render overlay → render generazioni.

Esegue in sequenza i 4 step chiamando gli script in src/ tramite subprocess.
Può essere lanciato sia dalla root che da qualsiasi altra cartella.

Esempi:
    python pipeline.py
    python pipeline.py --episodes 5000 --checkpoint_every 200 --overlay_runs 50
    python pipeline.py --skip_train --policy softmax --temperature 0.3
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Root del repo (pipeline.py è nella root) ──────────────────
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
from config import DEFAULT_FPS, BACKTRACK_PENALTY  # noqa: E402

# Cartelle output (speculari a config.py)
OUTPUT_ROOT    = REPO_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
EVAL_DIR       = OUTPUT_ROOT / "eval"
RENDER_DIR     = OUTPUT_ROOT / "renders"
LOG_DIR        = OUTPUT_ROOT / "logs"

PYTHON = sys.executable  # usa lo stesso interprete che ha lanciato pipeline.py


def run_step(label: str, cmd: list[str]):
    """Esegue un comando stampando un banner."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}\n")
    print(f"  cmd: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"\n✗ Step '{label}' fallito (exit code {result.returncode})")
        sys.exit(result.returncode)


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline completa: train → eval → render")

    # ── training ───────────────────────────────────────────────
    p.add_argument("--episodes",        type=int,   default=3000)
    p.add_argument("--checkpoint_every", type=int,  default=500)
    p.add_argument("--alpha",           type=float, default=0.1)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--epsilon_start",   type=float, default=1.0)
    p.add_argument("--epsilon_min",     type=float, default=0.05)
    p.add_argument("--epsilon_decay",   type=float, default=0.995)
    p.add_argument("--backtrack_penalty", type=float, default=BACKTRACK_PENALTY)

    # ── policy (condiviso train/eval/render) ───────────────────
    p.add_argument("--policy",       type=str, default="eps_greedy",
                   choices=["greedy", "eps_greedy", "softmax"])
    p.add_argument("--temperature",  type=float, default=0.5)
    p.add_argument("--maze",         type=str, default="mazes/maze.txt")

    # ── evaluation ─────────────────────────────────────────────
    p.add_argument("--eval_runs",    type=int, default=20)

    # ── render overlay ─────────────────────────────────────────
    p.add_argument("--overlay_runs", type=int, default=20)
    p.add_argument("--seed",         type=int, default=None)

    # ── render generazioni ─────────────────────────────────────
    p.add_argument("--gen_runs",     type=int, default=5,
                   help="Agenti per generazione nel video generazioni")

    # ── render (condiviso single + overlay) ────────────────────
    p.add_argument("--cell",         type=int, default=20)
    p.add_argument("--fps",          type=int, default=DEFAULT_FPS)

    # ── skip flags ─────────────────────────────────────────────
    p.add_argument("--skip_train",   action="store_true")
    p.add_argument("--skip_eval",    action="store_true")
    p.add_argument("--skip_render",  action="store_true")
    p.add_argument("--skip_generations", action="store_true",
                   help="Salta il render generazioni")
    p.add_argument("--all_policies", action="store_true",
                   help="Esegue eval + render per tutte e 3 le policy "
                        "(greedy, eps_greedy, softmax)")

    return p.parse_args()


ALL_POLICIES = ["greedy", "eps_greedy", "softmax"]


def _run_eval_and_render(args, policy: str, label_prefix: str = ""):
    """Esegue eval + render single + overlay + generazioni per una singola policy."""
    q_best = str(CHECKPOINT_DIR / "Q_best.npy")
    pfx = f"{label_prefix}  " if label_prefix else ""

    # ── Evaluate checkpoints ───────────────────────────────────
    if not args.skip_eval:
        cmd = [
            PYTHON, str(SRC_DIR / "evaluate_checkpoints.py"),
            "--policy",         policy,
            "--epsilon_min",    str(args.epsilon_min),
            "--temperature",    str(args.temperature),
            "--eval_runs",      str(args.eval_runs),
            "--maze",           args.maze,
            "--checkpoint_dir", str(CHECKPOINT_DIR),
            "--eval_dir",       str(EVAL_DIR),
        ]
        run_step(f"{pfx}Evaluate checkpoints  [{policy}]", cmd)

    # ── Render single run (best) ──────────────────────────────
    if not args.skip_render:
        out_single = str(RENDER_DIR / f"run_best_{policy}.mp4")
        cmd = [
            PYTHON, str(SRC_DIR / "render_run.py"),
            "--q_path",      q_best,
            "--out",         out_single,
            "--policy",      policy,
            "--epsilon_min", str(args.epsilon_min),
            "--temperature", str(args.temperature),
            "--maze",        args.maze,
            "--cell",        str(args.cell),
            "--fps",         str(args.fps),
        ]
        run_step(f"{pfx}Render single run  [{policy}]", cmd)

    # ── Render overlay (best) ─────────────────────────────────
    if not args.skip_render:
        out_overlay = str(RENDER_DIR /
                          f"run_best_{policy}_overlay_{args.overlay_runs}runs.mp4")
        cmd = [
            PYTHON, str(SRC_DIR / "render_run.py"),
            "--q_path",      q_best,
            "--out",         out_overlay,
            "--policy",      policy,
            "--epsilon_min", str(args.epsilon_min),
            "--temperature", str(args.temperature),
            "--maze",        args.maze,
            "--runs",        str(args.overlay_runs),
            "--cell",        str(args.cell),
            "--fps",         str(args.fps),
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        run_step(f"{pfx}Render overlay  [{policy}]", cmd)

    # ── Render generazioni (tutti i checkpoint) ───────────────
    if not args.skip_render and not args.skip_generations:
        out_gen = str(RENDER_DIR /
                      f"run_generations_{policy}_{args.gen_runs}each.mp4")
        cmd = [
            PYTHON, str(SRC_DIR / "render_run.py"),
            "--generations",
            "--checkpoint_dir", str(CHECKPOINT_DIR),
            "--out",         out_gen,
            "--policy",      policy,
            "--epsilon_min", str(args.epsilon_min),
            "--temperature", str(args.temperature),
            "--maze",        args.maze,
            "--runs",        str(args.gen_runs),
            "--cell",        str(args.cell),
            "--fps",         str(args.fps),
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        run_step(f"{pfx}Render generazioni  [{policy}]", cmd)


def main():
    args = parse_args()

    # ── 1) Training (una volta sola, la Q-table è policy-agnostica) ──
    if not args.skip_train:
        cmd = [
            PYTHON, str(SRC_DIR / "train.py"),
            "--episodes",        str(args.episodes),
            "--checkpoint_every", str(args.checkpoint_every),
            "--alpha",           str(args.alpha),
            "--gamma",           str(args.gamma),
            "--epsilon_start",   str(args.epsilon_start),
            "--epsilon_min",     str(args.epsilon_min),
            "--epsilon_decay",   str(args.epsilon_decay),
            "--policy",          args.policy,
            "--temperature",     str(args.temperature),
            "--maze",            args.maze,
            "--backtrack_penalty", str(args.backtrack_penalty),
            "--out_dir",         str(CHECKPOINT_DIR),
        ]
        run_step("Training", cmd)

    # ── 2-5) Eval + render ─────────────────────────────────────
    if args.all_policies:
        for i, pol in enumerate(ALL_POLICIES, 1):
            label = f"[{i}/{len(ALL_POLICIES)}]"
            _run_eval_and_render(args, pol, label_prefix=label)
    else:
        _run_eval_and_render(args, args.policy)

    # ── Riepilogo ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Pipeline completata!")
    print(f"{'='*60}\n")

    sections = [
        ("Checkpoints",  CHECKPOINT_DIR, "*.npy"),
        ("Eval grafici", EVAL_DIR,       "*.png"),
        ("Render video", RENDER_DIR,     "*.mp4"),
        ("Render PNG",   RENDER_DIR,     "*.png"),
        ("Logs",         LOG_DIR,        "*.png"),
    ]
    for label, d, glob in sections:
        files = sorted(d.glob(glob)) if d.exists() else []
        if files:
            print(f"  {label}:")
            for f in files:
                print(f"    {f.relative_to(REPO_ROOT)}")
    print()


if __name__ == "__main__":
    main()
