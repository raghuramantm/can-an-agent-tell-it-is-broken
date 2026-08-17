"""
Deterministic seeding across NumPy, Python's `random`, PyTorch (CPU + MPS),
and Gymnasium. SB3 also exposes `set_random_seed`; we call it for parity.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed as sb3_set_seed


def set_global_seed(seed: int, deterministic_torch: bool = False) -> None:
    """
    Seed every stochastic component used in the training loop.

    `deterministic_torch=True` forces deterministic CUDA/MPS kernels (slower).
    Leave `False` for baseline training; flip on only for reproducibility runs.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # SB3 also sets its own internal RNGs (including env wrappers it owns)
    sb3_set_seed(seed)


def seed_list(base: int = 0, n: int = 5) -> list[int]:
    """Conventional multi-seed schedule used across baseline experiments."""
    return [base + i for i in range(n)]
