"""
Run-directory layout and small logging helpers.

A single training run lives under:

    runs/<algo>__<env>__seed<seed>__<utc_timestamp>/
        ├── tb/                # TensorBoard
        ├── monitor/           # SB3 Monitor CSVs (per env)
        ├── eval/              # Eval callback results.npz + best_model.zip
        ├── checkpoints/       # periodic .zip checkpoints (thermal safety)
        ├── final_model.zip
        ├── config.yaml        # snapshot of resolved hyperparameters
        └── run_meta.json      # device, library versions, wallclock
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import yaml


def make_run_dir(
    root: str | Path,
    algo: str,
    env_id: str,
    seed: int,
    tag: str | None = None,
) -> Path:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    pieces = [algo, env_id.replace("/", "-"), f"seed{seed}", ts]
    if tag:
        pieces.insert(-1, tag)
    name = "__".join(pieces)
    run_dir = Path(root) / name
    for sub in ("tb", "monitor", "eval", "checkpoints"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_yaml(path: str | Path, payload: dict) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def dump_json(path: str | Path, payload: dict) -> None:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
