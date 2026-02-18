import numpy as np
from pathlib import Path
from config import REPO_ROOT


def load_maze(txt_path: str):
    """Carica un labirinto da file .txt (griglia 0/1, opzionali S e G).

    Formato:
        - Righe di valori separati da spazi: 0 = libero, 1 = muro.
        - Opzionali: 'S' = start, 'G' = goal (trattati come celle libere).
        - Righe vuote vengono ignorate.

    Se S/G non presenti nel file:
        - start = prima cella libera (scan top-left → bottom-right)
        - goal  = ultima cella libera (scan top-left → bottom-right)

    Returns:
        (name, grid, start, goal, max_steps)
    """
    p = Path(txt_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()

    if p.suffix.lower() != ".txt":
        raise ValueError(
            f"Maze must be a .txt grid file (0/1 with optional S/G). "
            f"Got: {p.name}"
        )

    raw_lines = p.read_text(encoding="utf-8").splitlines()
    rows = []
    start = None
    goal = None

    for ri, line in enumerate(raw_lines):
        tokens = line.split()
        if not tokens:
            continue  # ignora righe vuote
        row = []
        for ci, tok in enumerate(tokens):
            if tok == "S":
                start = (len(rows), ci)
                row.append(0)
            elif tok == "G":
                goal = (len(rows), ci)
                row.append(0)
            elif tok in ("0", "1"):
                row.append(int(tok))
            else:
                raise ValueError(
                    f"Valore non valido '{tok}' alla riga {ri+1}, "
                    f"colonna {ci+1} in {p.name}. Ammessi: 0, 1, S, G."
                )
        rows.append(row)

    if not rows:
        raise ValueError(f"File vuoto o senza righe valide: {p.name}")

    # Valida che tutte le righe abbiano la stessa larghezza
    W = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != W:
            raise ValueError(
                f"Riga {i+1} ha {len(row)} colonne, attese {W} "
                f"(come la prima riga). File: {p.name}"
            )

    grid = np.array(rows, dtype=np.int8)
    H = grid.shape[0]

    # start / goal automatici se non trovati come marker
    if start is None or goal is None:
        free_cells = list(zip(*np.where(grid == 0)))
        if not free_cells:
            raise ValueError(f"Nessuna cella libera nel labirinto: {p.name}")
        if start is None:
            start = free_cells[0]
        if goal is None:
            goal = free_cells[-1]

    # Valida start e goal
    if grid[start] != 0:
        raise ValueError(f"Start {start} è un muro nel labirinto: {p.name}")
    if grid[goal] != 0:
        raise ValueError(f"Goal {goal} è un muro nel labirinto: {p.name}")

    max_steps = H * W * 2
    name = p.stem

    return name, grid, start, goal, max_steps
