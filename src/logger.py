"""
Logging condiviso: scrive simultaneamente su console e su file in outputs/logs/.

Uso:
    from logger import setup_logger
    log = setup_logger("train", policy="eps_greedy")
    log.info("messaggio")          # → console + file

File di log: outputs/logs/{name}_{policy}.log  (append mode).
Ogni sessione è separata da un header con timestamp.
"""

import logging
from datetime import datetime
from pathlib import Path

import config


def setup_logger(name: str, policy: str = "default") -> logging.Logger:
    """Configura e restituisce un logger con handler console + file."""
    log_dir = config.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{name}_{policy}.log"

    logger = logging.getLogger(f"{name}_{policy}")
    logger.setLevel(logging.DEBUG)

    # evita handler duplicati se chiamato più volte
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(message)s")

    # handler console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # handler file (append)
    fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # separatore sessione
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n{'='*60}")
    logger.info(f"  Sessione: {ts}")
    logger.info(f"{'='*60}")

    return logger
