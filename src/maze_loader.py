import json
import numpy as np
from pathlib import Path

def load_maze(json_path: str):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    grid = np.array(data["grid"], dtype=np.int8)
    start = tuple(data["start"])
    goal = tuple(data["goal"])
    max_steps = int(data.get("max_steps", 200))
    name = data.get("name", Path(json_path).stem)
    return name, grid, start, goal, max_steps
