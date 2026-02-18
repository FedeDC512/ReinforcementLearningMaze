# prova-finale

Reinforcement Learning su labirinto a griglia con Q-Learning.

## Requisiti

```
pip install numpy matplotlib pygame opencv-python
```

## Formato labirinto (.txt)

Il labirinto si definisce in un file `.txt` con valori separati da spazi:

| Valore | Significato |
|---|---|
| `0` | Cella libera |
| `1` | Muro |
| `S` | Start (opzionale — trattato come `0`) |
| `G` | Goal (opzionale — trattato come `0`) |

Se `S`/`G` non sono presenti, start = prima cella libera (alto-sinistra),
goal = ultima cella libera (basso-destra).

Dimensioni (H, W) e `max_steps` vengono ricavati automaticamente dal file.
Righe vuote vengono ignorate.

Esempio mini 5×5:
```
1 1 1 1 1
S 0 0 0 1
1 1 1 0 1
1 0 0 0 G
1 1 1 1 1
```

## Quick start

```bash
# Dalla root del repo — un solo comando per tutto:
python pipeline.py

# Con parametri custom:
python pipeline.py --episodes 5000 --checkpoint_every 200 --overlay_runs 50

# Maze diverso:
python pipeline.py --maze mazes/maze.txt

# Salta il training (usa checkpoint già salvati):
python pipeline.py --skip_train --policy softmax --temperature 0.3
```

Tutti gli output finiscono in `outputs/` nella root del repo, indipendentemente
dalla cartella da cui si lancia il comando.

## Struttura output

```
outputs/
  checkpoints/        # Q_epXXXX.npy, Q_best.npy, policy JSON
  eval/               # paths_<tag>.png, summary_metrics.png
  renders/            # mp4 e png (single run + overlay)
  logs/               # steps_curve.png, metriche training
```

## Pipeline (`pipeline.py`)

Esegue in sequenza: **training → evaluation → render single → render overlay**.

| Flag | Default | Descrizione |
|---|---|---|
| `--episodes` | 3000 | Episodi di training |
| `--checkpoint_every` | 500 | Salva checkpoint ogni K episodi |
| `--policy` | eps_greedy | `greedy` / `eps_greedy` / `softmax` |
| `--epsilon_min` | 0.05 | Epsilon minimo |
| `--temperature` | 0.5 | Temperatura softmax |
| `--eval_runs` | 20 | N run per checkpoint in evaluation |
| `--overlay_runs` | 20 | N run per video overlay |
| `--seed` | None | Seed per riproducibilità render overlay |
| `--maze` | mazes/maze.txt | Path al labirinto (.txt) |
| `--skip_train` | false | Salta il training |
| `--skip_eval` | false | Salta la valutazione |
| `--skip_render` | false | Salta i render video |

## Comandi singoli

Ogni script può ancora essere lanciato singolarmente. I path sono
root-relative: `mazes/maze.txt` e `outputs/checkpoints/` funzionano
sia dalla root che da `src/`.

### Training (`src/train.py`)

```bash
python src/train.py
python src/train.py --maze mazes/maze.txt --episodes 5000 --checkpoint_every 200
```

| Flag | Default | Descrizione |
|---|---|---|
| `--episodes` | 3000 | Numero di episodi |
| `--alpha` | 0.1 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--epsilon_start` | 1.0 | Epsilon iniziale |
| `--epsilon_min` | 0.05 | Epsilon minimo (mai 0) |
| `--epsilon_decay` | 0.995 | Decay moltiplicativo per epsilon |
| `--policy` | eps_greedy | `greedy` / `eps_greedy` / `softmax` |
| `--temperature` | 0.5 | Temperatura per softmax |
| `--checkpoint_every` | 500 | Salva checkpoint ogni K episodi |
| `--maze` | mazes/maze.txt | Path al file labirinto (.txt) |
| `--out_dir` | outputs/checkpoints | Cartella checkpoint |

### Valutazione (`src/evaluate_checkpoints.py`)

```bash
python src/evaluate_checkpoints.py
python src/evaluate_checkpoints.py --policy softmax --temperature 0.3 --eval_runs 30
```

| Flag | Default | Descrizione |
|---|---|---|
| `--policy` | eps_greedy | Policy di valutazione |
| `--epsilon_min` | 0.05 | Epsilon per eps_greedy in eval |
| `--temperature` | 0.5 | Temperatura per softmax in eval |
| `--eval_runs` | 20 | Numero di run per checkpoint |
| `--maze` | mazes/maze.txt | Path al labirinto |
| `--checkpoint_dir` | outputs/checkpoints | Cartella dei checkpoint |
| `--eval_dir` | outputs/eval | Cartella output grafici |

### Render video (`src/render_run.py`)

Supporta due modalità: **singolo run** (default) e **overlay** (N run sovrapposti).

```bash
# Singolo run
python src/render_run.py --q_path outputs/checkpoints/Q_best.npy

# Overlay 50 run
python src/render_run.py --q_path outputs/checkpoints/Q_best.npy --runs 50 --policy softmax

# Overlay con seed per riproducibilità
python src/render_run.py --q_path Q_best.npy --runs 30 --seed 42
```

| Flag | Default | Descrizione |
|---|---|---|
| `--q_path` | None | Path al file Q .npy (None = random) |
| `--out` | auto | Path output mp4 (auto in outputs/renders/) |
| `--policy` | eps_greedy | `greedy` / `eps_greedy` / `softmax` |
| `--epsilon_min` | 0.05 | Epsilon per eps_greedy |
| `--temperature` | 0.5 | Temperatura per softmax |
| `--maze` | mazes/maze.txt | Path al labirinto |
| `--runs` | 1 | Numero di run (>1 attiva overlay) |
| `--overlay` | false | Forza overlay anche con runs=1 |
| `--seed` | None | Seed per riproducibilità |
| `--cell` | 80 | Dimensione cella in pixel |
| `--fps` | 4 | Frame per secondo |

Nel video overlay:
- **Heatmap arancione** — celle più visitate si accendono progressivamente
- **Cerchietti colorati** — posizione di ciascun agente al timestep t
- **Hold finale** — 2 secondi di pausa sull'ultimo frame
- **PNG statico** salvato accanto al mp4

## Policy disponibili

| Policy | Comportamento |
|---|---|
| `greedy` | Sempre argmax(Q) — deterministico |
| `eps_greedy` | Con probabilità ε sceglie random, altrimenti argmax. ε ≥ ε_min > 0 |
| `softmax` | prob(a\|s) = softmax(Q(s,·)/τ) — stocastico, τ controlla la "temperatura" |