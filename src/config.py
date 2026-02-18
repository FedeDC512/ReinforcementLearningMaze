from pathlib import Path

# ── Root del repo (config.py è in src/) ────────────────────────
REPO_ROOT       = Path(__file__).resolve().parent.parent

# ── Struttura output standard ──────────────────────────────────
OUTPUT_ROOT     = REPO_ROOT / "outputs"
CHECKPOINT_DIR  = OUTPUT_ROOT / "checkpoints"
EVAL_DIR        = OUTPUT_ROOT / "eval"
RENDER_DIR      = OUTPUT_ROOT / "renders"
LOG_DIR         = OUTPUT_ROOT / "logs"

MAZE_PATH = "mazes/maze.txt"

# ── Training defaults ──────────────────────────────────────────
EPISODES        = 3000
ALPHA           = 0.1
GAMMA           = 0.99

# ── Epsilon schedule ───────────────────────────────────────────
EPSILON_START   = 1.0
EPSILON_MIN     = 0.05      # mai 0 → mantiene esplorazione residua
EPSILON_DECAY   = 0.995

# ── Softmax policy ─────────────────────────────────────────────
TEMPERATURE     = 0.5       # più bassa → più greedy
TEMPERATURE_MIN = 0.01      # floor per evitare collasso numerico

# ── Policy mode (greedy | eps_greedy | softmax) ────────────────
POLICY_MODE     = "eps_greedy"

# ── Checkpoint ─────────────────────────────────────────────────
CHECKPOINT_EVERY = 500      # salva Q ogni K episodi

# ── Evaluation ─────────────────────────────────────────────────
EVAL_RUNS       = 20        # N run per checkpoint in evaluate
# ── Render ─────────────────────────────────────────────────
DEFAULT_FPS     = 8         # frame per secondo nei video generati

# ── Reward shaping ───────────────────────────────────────────
BACKTRACK_PENALTY = -0.1   # penalità per azione opposta a quella precedente
                             # (0 = disattivato)