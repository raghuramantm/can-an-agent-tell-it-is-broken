"""
Plot learning curves from SB3 Monitor CSV logs across seeds and algorithms.

Usage
-----
    python scripts/plot_results.py --runs-root runs/ --out logs/learning_curves.png

The plot shows mean episode return (smoothed via rolling window) ± SEM band
across seeds, for each algorithm tag found under `runs-root`.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


RUN_NAME_RE = re.compile(r"^(?P<algo>[a-zA-Z0-9_]+)__(?P<env>[A-Za-z0-9\-]+)__")


def load_monitor_csvs(run_dir: Path) -> pd.DataFrame | None:
    """Read all worker monitor CSVs, append timesteps cumulatively."""
    csvs = sorted((run_dir / "monitor").glob("worker_*.monitor.csv"))
    if not csvs:
        # fall back: SB3 may write monitor files at the root
        csvs = sorted(run_dir.glob("**/*.monitor.csv"))
    frames = []
    for c in csvs:
        try:
            df = pd.read_csv(c, skiprows=1)
        except Exception as e:
            print(f"[warn] cannot read {c}: {e}")
            continue
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    df["cum_timesteps"] = df["l"].cumsum()
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=str, default="runs")
    parser.add_argument("--out", type=str, default="logs/learning_curves.png")
    parser.add_argument("--smooth", type=int, default=50, help="Rolling window size.")
    args = parser.parse_args(argv)

    root = Path(args.runs_root)
    runs = [p for p in root.iterdir() if p.is_dir()]
    if not runs:
        print(f"[error] no runs under {root}")
        return 2

    records = []
    for run in runs:
        m = RUN_NAME_RE.match(run.name)
        if not m:
            continue
        df = load_monitor_csvs(run)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["reward_smooth"] = df["r"].rolling(args.smooth, min_periods=1).mean()
        df["algo"] = m.group("algo")
        df["run"] = run.name
        records.append(df)

    if not records:
        print("[error] no monitor data found")
        return 2

    full = pd.concat(records, ignore_index=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    for algo, sub in full.groupby("algo"):
        # bin by env step to align seeds
        n_bins = 200
        bins = np.linspace(
            sub["cum_timesteps"].min(),
            sub["cum_timesteps"].max(),
            n_bins,
        )
        sub = sub.copy()
        sub["bin"] = pd.cut(sub["cum_timesteps"], bins, labels=False)
        agg = sub.groupby("bin")["reward_smooth"].agg(["mean", "sem", "count"])
        agg = agg.dropna()
        x = bins[:-1][agg.index.astype(int)]
        ax.plot(x, agg["mean"], label=algo, linewidth=2)
        ax.fill_between(
            x,
            agg["mean"] - 1.96 * agg["sem"],
            agg["mean"] + 1.96 * agg["sem"],
            alpha=0.20,
        )

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return (rolling mean across seeds)")
    ax.set_title("LunarLander baselines — mean ± 95 % CI across seeds")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"[plot] saved → {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
