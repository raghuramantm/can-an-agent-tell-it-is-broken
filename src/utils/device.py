"""
Device selection utilities for Apple Silicon (M-series) hosts.

Policy
------
- Heavy DL tensor ops (SAC critics, RecurrentPPO LSTM, future forward models)
  → `mps` whenever available.
- Small MLP policies whose forward passes are dominated by vectorized env
  stepping (vanilla PPO on LunarLander)
  → `cpu` outperforms `mps` because of per-call dispatch latency. We expose a
  `prefer="auto"` switch so the user can override per-experiment.

References
----------
- PyTorch MPS notes: https://pytorch.org/docs/stable/notes/mps.html
- SB3 device guidance: https://stable-baselines3.readthedocs.io/en/master/guide/install.html
"""
from __future__ import annotations

import os
import platform

import torch


def get_torch_device(prefer: str = "auto") -> torch.device:
    """
    Resolve a torch.device under M-series constraints.

    Parameters
    ----------
    prefer : {"auto", "mps", "cpu", "cuda"}
        - "auto" : MPS if available on macOS arm64, else CPU.
        - explicit values are honored but downgraded with a warning if
          unavailable.

    Returns
    -------
    torch.device
    """
    prefer = prefer.lower()
    is_apple_silicon = (
        platform.system() == "Darwin" and platform.machine() == "arm64"
    )
    mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    cuda_ok = torch.cuda.is_available()

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if cuda_ok else "cpu")
    if prefer == "mps":
        return torch.device("mps" if mps_ok else "cpu")

    # auto
    if is_apple_silicon and mps_ok:
        return torch.device("mps")
    if cuda_ok:
        return torch.device("cuda")
    return torch.device("cpu")


def configure_threading(num_threads: int | None = None) -> None:
    """
    Cap PyTorch / BLAS thread fan-out so multiple parallel env workers do
    not contend for the same M4 P-cores. Defaults to 1 thread per process,
    which is optimal for SubprocVecEnv layouts.
    """
    n = int(num_threads) if num_threads else 1
    torch.set_num_threads(n)
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))


def describe_device(device: torch.device) -> str:
    """Human-readable line for logs / TensorBoard text."""
    return (
        f"device={device.type} | torch={torch.__version__} | "
        f"mps_built={torch.backends.mps.is_built()} | "
        f"mps_avail={torch.backends.mps.is_available()} | "
        f"cuda_avail={torch.cuda.is_available()}"
    )
