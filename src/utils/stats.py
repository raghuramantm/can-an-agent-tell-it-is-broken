"""
Statistical helpers for the multi-seed RL evaluation pipeline.

Mirrors the A2_Final assessment's statistical battery so the dissertation
Methods section can cite a complete set of tests (location, distribution,
effect size, multiple-comparison correction) rather than a single t-test.

References
----------
- Henderson et al. (2018) "Deep Reinforcement Learning that Matters" —
  multi-seed RL evaluation conventions.
- Romano, J.; Kromrey, J. D.; Coraggio, J.; Skowronek, J. (2006) —
  Cliff's δ effect size.
- Benjamini & Hochberg (1995) — false discovery rate correction.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


# --------------------------------------------------------------------------- #
# Confidence intervals
# --------------------------------------------------------------------------- #
def bootstrap_ci(
    values: Sequence[float],
    func: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Non-parametric bootstrap CI on a 1-D array.

    Parameters
    ----------
    values : sequence of floats
        The data to resample.
    func : callable
        Statistic to compute on each bootstrap sample (default: mean).
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence-interval width in percent (default 95).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (lo, hi) : tuple[float, float]
        Lower and upper CI bounds.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats[i] = func(sample)
    alpha = (100 - ci) / 2
    return float(np.percentile(stats, alpha)), float(np.percentile(stats, 100 - alpha))


def seed_level_bootstrap_ci(
    per_seed_means: Sequence[float],
    n_boot: int = 10_000,
    ci: float = 95.0,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Resample seeds (not episodes) with replacement and return CI of the
    algorithm-level mean.

    This is the appropriate uncertainty estimate for the question
    "would running another sweep of 5 seeds reproduce this result?".
    Use this rather than episode-level bootstrap when the unit of
    independence is the seed, not the episode.
    """
    return bootstrap_ci(per_seed_means, func=np.mean, n_boot=n_boot,
                         ci=ci, seed=seed)


# --------------------------------------------------------------------------- #
# Effect sizes
# --------------------------------------------------------------------------- #
def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Cliff's δ effect size for two independent samples.

    δ ∈ [-1, +1].
    Magnitude thresholds (Romano et al., 2006):
        |δ| < 0.147       → negligible
        0.147 ≤ |δ| < 0.33 → small
        0.33  ≤ |δ| < 0.474 → medium
        |δ| ≥ 0.474       → large

    Computed from the Mann-Whitney U statistic:
        δ = 2 * U / (n_x * n_y) - 1
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    u = mannwhitneyu(x, y, alternative="two-sided").statistic
    return 2.0 * float(u) / (nx * ny) - 1.0


def cliffs_delta_label(delta: float) -> str:
    """Human-readable magnitude label for Cliff's δ."""
    a = abs(delta)
    if np.isnan(delta):
        return "n/a"
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


# --------------------------------------------------------------------------- #
# Multiple-comparison correction
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg false-discovery-rate adjustment.

    Returns an array of adjusted p-values in the original order.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(p)
    out[order] = adj
    return out


# --------------------------------------------------------------------------- #
# Pairwise comparison table
# --------------------------------------------------------------------------- #
def pairwise_stats_table(
    per_algo_returns: dict[str, Sequence[float]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Every pair of algorithms compared with:
      - Welch's t (location shift, unequal variance)
      - Mann-Whitney U (rank-based, distribution-shift)
      - Cliff's δ + magnitude label (effect size)
      - BH-corrected p-values across the family of pairwise tests

    Parameters
    ----------
    per_algo_returns : dict[str, list[float]]
        Algorithm-name → flat list of per-episode returns.
    alpha : float
        Family-wise significance threshold for the BH-adjusted column.

    Returns
    -------
    pd.DataFrame
        Columns: algo_a, algo_b, n_a, n_b, mean_a, mean_b, delta_mean,
                 welch_t, welch_p, mwu_p, cliffs_delta, cliffs_label,
                 welch_p_bh, mwu_p_bh, significant
    """
    algos = list(per_algo_returns.keys())
    rows = []
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            xa = np.asarray(per_algo_returns[a], dtype=float)
            xb = np.asarray(per_algo_returns[b], dtype=float)
            t_stat, p_welch = ttest_ind(xa, xb, equal_var=False)
            mwu_p = mannwhitneyu(xa, xb, alternative="two-sided").pvalue
            d = cliffs_delta(xa, xb)
            rows.append({
                "algo_a": a,
                "algo_b": b,
                "n_a": int(xa.size),
                "n_b": int(xb.size),
                "mean_a": float(xa.mean()),
                "mean_b": float(xb.mean()),
                "delta_mean": float(xa.mean() - xb.mean()),
                "welch_t": float(t_stat),
                "welch_p": float(p_welch),
                "mwu_p": float(mwu_p),
                "cliffs_delta": float(d),
                "cliffs_label": cliffs_delta_label(d),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["welch_p_bh"] = benjamini_hochberg(df["welch_p"].to_numpy())
    df["mwu_p_bh"] = benjamini_hochberg(df["mwu_p"].to_numpy())
    df["significant"] = (df["welch_p_bh"] < alpha) & (df["mwu_p_bh"] < alpha)
    return df
