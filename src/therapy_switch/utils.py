"""Cross-cutting reproducibility, logging, and timing helpers."""

from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator

import numpy as np


def configure_logging(output_dir: str | Path, verbose: bool = False) -> logging.Logger:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("therapy_switch")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_path / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_global_seed(seed: int) -> None:
    """Seed installed numerical frameworks without requiring optional packages."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


@dataclass
class RuntimeRecord:
    label: str
    seconds: float = 0.0


@contextmanager
def timed(label: str, sink: Dict[str, float] | None = None) -> Iterator[RuntimeRecord]:
    record = RuntimeRecord(label=label)
    started = time.perf_counter()
    try:
        yield record
    finally:
        record.seconds = time.perf_counter() - started
        if sink is not None:
            sink[label] = record.seconds
