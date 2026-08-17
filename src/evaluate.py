"""
Multi-seed evaluation of trained baseline agents.

Given a set of run directories produced by `src.train`, this script:

  1. Loads each `final_model.zip`.
  2. Runs `--n-episodes` episodes on a target env spec (nominal by default,
     or with an actuator fault / DR wrapper applied).
  3. Reports per-seed mean returns, aggregated mean ± 95 % bootstrap CI.
  4. Pairwise Welch's t-test between algorithms (when more than one given).

Statistical formulas
--------------------
  Welch's t = (μ_A − μ_B) / sqrt(s²_A/n_A + s²_B/n_B)
  CI_95 via stratified bootstrap (10 000 resamples).

Example
-------
    python -m src.evaluate \\
        --runs runs/ppo__LunarLander-v3__seed* \\
               runs/sac__LunarLanderContinuous-v3__seed* \\
        --n-episodes 50 \\
        --fault-step 200 --fault-gain 0.5
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from src.envs.lunarlander_factory import make_single_eval_env
from src.envs.wrappers import (
    ActuatorFaultSpec,
    DomainRandomizationSpec,
    WrapperStack,
)


ALGO_LOADERS = {"ppo": PPO, "sac": SAC, "ppo_dr": PPO}


def _expand_run_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        if not matches:
            print(f"[warn] no match for {p}")
        for m in matches:
            paths.append(Path(m))
    return paths


def _load_run(run_dir: Path):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    algo = cfg["algo"].lower()
    model_path = run_dir / "final_model.zip"
    if not model_path.exists():
        # fall back to best model
        cand = run_dir / "eval" / "best_model.zip"
        if cand.exists():
            model_path = cand
        else:
            raise FileNotFoundError(f"no model under {run_dir}")
    Loader = ALGO_LOADERS[algo]
    model = Loader.load(str(model_path), device="cpu")  # eval on CPU is safe
    return cfg, model


def _build_eval_wrappers(
    fault_step: int | None,
    fault_gain: float | None,
    discrete_drop_action: int | None,
    dr: bool,
) -> WrapperStack | None:
    """
    Build the eval-time wrapper stack.

    NB: For *discrete* envs the wrapper needs `discrete_drop_action` to do
    anything — `fault_gain` is ignored. If the caller passes `--fault-step`
    without `--discrete-drop-action`, we default to action=2 (main engine)
    so the discrete fault is non-trivial; otherwise nominal and fault eval
    return byte-identical numbers and the user is misled.
    """
    if fault_step is None and not dr:
        return None
    stack = WrapperStack()
    if dr:
        stack.domain_randomization = DomainRandomizationSpec()
    if fault_step is not None:
        stack.actuator_fault = ActuatorFaultSpec(
            fault_step=int(fault_step),
            fault_gain=float(fault_gain if fault_gain is not None else 0.5),
            discrete_drop_action=(
                int(discrete_drop_action)
                if discrete_drop_action is not None
                else 2  # main-engine default for LunarLander-v3
            ),
        )
    return stack


def _bootstrap_ci(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    boots = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, help="Run-dir globs.")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--fault-step", type=int, default=None,
                        help="Time step at which the actuator fault activates.")
    parser.add_argument("--fault-gain", type=float, default=None,
                        help="Per-thruster gain multiplier (continuous envs only).")
    parser.add_argument("--discrete-drop-action", type=int, default=None,
                        help="Discrete action id rewritten to no-op post-fault. "
                             "Default 2 (main engine) for LunarLander-v3.")
    parser.add_argument("--dr", action="store_true", help="Apply DR wrapper at eval.")
    parser.add_argument(
        "--out", type=str, default="logs/eval_results.csv",
        help="Path to write per-episode rewards.",
    )
    args = parser.parse_args(argv)

    runs = _expand_run_globs(args.runs)
    if not runs:
        print("[error] no runs found")
        return 2

    wrapper_stack = _build_eval_wrappers(
        args.fault_step, args.fault_gain, args.discrete_drop_action, args.dr,
    )

    per_algo_returns: dict[str, list[float]] = defaultdict(list)
    rows: list[dict] = []

    for run_dir in runs:
        cfg, model = _load_run(run_dir)
        algo = cfg["algo"]
        env_id = cfg["env_id"]
        will_vec_wrap = bool(cfg.get("normalize_obs", False)
                              and (run_dir / "vecnormalize.pkl").exists())
        # Strip the inner Monitor when we're going to wrap in VecMonitor —
        # otherwise episode stats are double-counted (UserWarning + biased std).
        raw_env = make_single_eval_env(
            env_id=env_id,
            seed=int(cfg.get("seed", 0)) + 9999,
            wrapper_stack=wrapper_stack,
            add_monitor=not will_vec_wrap,
        )

        # If the run was trained with normalize_obs=true, the saved policy
        # consumes normalised observations. We must reload vecnormalize.pkl
        # and apply the SAME running stats (in eval mode, no further updates).
        if will_vec_wrap:
            vec = DummyVecEnv([lambda: raw_env])
            vec = VecMonitor(vec)
            env = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), vec)
            env.training = False
            env.norm_reward = False    # eval reward must be raw
        else:
            env = raw_env

        ep_rewards, ep_lengths = evaluate_policy(
            model,
            env,
            n_eval_episodes=args.n_episodes,
            deterministic=True,
            return_episode_rewards=True,
        )
        env.close()

        ep_rewards = np.asarray(ep_rewards, dtype=np.float64)
        per_algo_returns[algo].extend(ep_rewards.tolist())

        mean_r = float(ep_rewards.mean())
        std_r = float(ep_rewards.std(ddof=1))
        rows.append({
            "run_dir": str(run_dir),
            "algo": algo,
            "env_id": env_id,
            "n_episodes": len(ep_rewards),
            "mean_return": mean_r,
            "std_return": std_r,
            "min_return": float(ep_rewards.min()),
            "max_return": float(ep_rewards.max()),
            "fault_step": args.fault_step,
            "fault_gain": args.fault_gain,
            "dr_eval": args.dr,
        })
        print(
            f"[eval] {algo:7s} {run_dir.name:60s} "
            f"mean={mean_r:8.2f}  std={std_r:7.2f}"
        )

    # --- Aggregate per algo (episode-level + seed-level bootstrap) ---
    from src.utils.stats import (
        bootstrap_ci,
        seed_level_bootstrap_ci,
        pairwise_stats_table,
    )
    print("\n" + "=" * 80)
    print("AGGREGATE  (per-episode mean ± 95% CI, episode bootstrap)")
    print("=" * 80)
    summary = []
    per_seed_means_by_algo: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per_seed_means_by_algo[r["algo"]].append(r["mean_return"])

    for algo, vals in per_algo_returns.items():
        arr = np.asarray(vals)
        ep_lo, ep_hi = bootstrap_ci(arr)
        seed_means = per_seed_means_by_algo[algo]
        seed_lo, seed_hi = seed_level_bootstrap_ci(seed_means)
        summary.append({
            "algo": algo,
            "n_episodes": int(arr.size),
            "n_seeds": len(seed_means),
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "ep_ci95_lo": ep_lo,
            "ep_ci95_hi": ep_hi,
            "seed_mean": float(np.mean(seed_means)) if seed_means else float("nan"),
            "seed_std": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else float("nan"),
            "seed_ci95_lo": seed_lo,
            "seed_ci95_hi": seed_hi,
        })
        print(
            f"  {algo:7s}  mean={arr.mean():8.2f}  "
            f"ep95%CI=[{ep_lo:7.2f}, {ep_hi:7.2f}]   "
            f"seed95%CI=[{seed_lo:7.2f}, {seed_hi:7.2f}]   "
            f"n_ep={arr.size}  n_seeds={len(seed_means)}"
        )

    # --- Pairwise stats battery (Welch's t, Mann-Whitney, Cliff's δ, BH) ---
    if len(per_algo_returns) >= 2:
        print("\nPairwise stats (BH-corrected; Cliff's δ effect size):")
        pw = pairwise_stats_table(per_algo_returns)
        cols = ["algo_a", "algo_b", "delta_mean", "welch_t", "welch_p_bh",
                "mwu_p_bh", "cliffs_delta", "cliffs_label", "significant"]
        with pd.option_context("display.float_format", "{:+.4g}".format,
                               "display.width", 200, "display.max_columns", None):
            print(pw[cols].to_string(index=False))
    else:
        pw = pd.DataFrame()

    # --- Persist ---
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    with open(out_path.with_suffix(".json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    if not pw.empty:
        pw_path = out_path.with_name(out_path.stem + "_pairwise.csv")
        pw.to_csv(pw_path, index=False)
        print(f"[eval] pairwise stats     → {pw_path}")
    print(f"\n[eval] per-run rows saved → {out_path}")
    print(f"[eval] aggregate summary  → {out_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
