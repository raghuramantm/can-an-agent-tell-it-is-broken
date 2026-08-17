"""
Unified trainer for the LunarLander baseline experiments.

Usage
-----
    python -m src.train --config configs/ppo_lunarlander.yaml --seed 0
    python -m src.train --config configs/sac_lunarlander.yaml --seed 0
    python -m src.train --config configs/ppo_dr_lunarlander.yaml --seed 0

Run artefacts are written under `runs/<algo>__<env>__seed<seed>__<utc>/`.

Design notes
------------
- PPO uses `SubprocVecEnv` so env stepping is true CPU-parallel.
- SAC uses a single env (off-policy single-env regime is standard).
- Device follows the YAML's `device` field unless overridden via CLI.
- Multi-seed runs are produced by calling this script N times with different
  `--seed`; see `scripts/run_all_baselines.sh`.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from src.envs.lunarlander_factory import (
    make_single_eval_env,
    make_vec_env,
)
from src.envs.wrappers import (
    DomainRandomizationSpec,
    WrapperStack,
)
from src.utils.device import (
    configure_threading,
    describe_device,
    get_torch_device,
)
from src.utils.logging_utils import dump_json, dump_yaml, make_run_dir
from src.utils.seeding import set_global_seed


ALGOS = {"ppo": PPO, "sac": SAC, "ppo_dr": PPO}


# ------------------------------------------------------------------- #
# Config plumbing
# ------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def build_wrapper_stack(cfg: dict) -> WrapperStack | None:
    spec = cfg.get("wrappers")
    if not spec:
        return None
    stack = WrapperStack()
    if "domain_randomization" in spec and spec["domain_randomization"] is not None:
        dr = spec["domain_randomization"]
        stack.domain_randomization = DomainRandomizationSpec(
            gravity_range=tuple(dr.get("gravity_range", (-12.0, -8.0))),
            main_engine_power_range=tuple(
                dr.get("main_engine_power_range", (11.0, 17.0))
            ),
            side_engine_power_range=tuple(
                dr.get("side_engine_power_range", (0.4, 0.8))
            ),
            wind_power_range=tuple(dr["wind_power_range"])
            if dr.get("wind_power_range") is not None
            else None,
            turbulence_power_range=tuple(dr["turbulence_power_range"])
            if dr.get("turbulence_power_range") is not None
            else None,
            enable_wind=bool(dr.get("enable_wind", True)),
            log_per_episode=bool(dr.get("log_per_episode", False)),
        )
    return stack


# ------------------------------------------------------------------- #
# Model builders
# ------------------------------------------------------------------- #
def build_model(cfg: dict, env, device: torch.device, seed: int, tb_path: str):
    algo_key = cfg["algo"].lower()
    Algo = ALGOS[algo_key]
    hp = dict(cfg["hyperparameters"])

    policy = hp.pop("policy", "MlpPolicy")
    common = dict(
        policy=policy,
        env=env,
        seed=seed,
        verbose=1,
        tensorboard_log=tb_path,
        device=str(device),
        **hp,
    )
    return Algo(**common)


# ------------------------------------------------------------------- #
# Main
# ------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LunarLander baseline trainer")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--runs-root", default="runs", type=str)
    parser.add_argument(
        "--device",
        default=None,
        help="Override config device: auto|mps|cpu|cuda",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override config total_timesteps (use for smoke tests).",
    )
    parser.add_argument("--tag", default=None, type=str)
    parser.add_argument(
        "--deterministic-torch",
        action="store_true",
        help="Slower but fully reproducible.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = int(args.total_timesteps)

    # --- run dir + config snapshot ---
    run_dir = make_run_dir(
        root=args.runs_root,
        algo=cfg["algo"],
        env_id=cfg["env_id"],
        seed=args.seed,
        tag=args.tag,
    )
    dump_yaml(run_dir / "config.yaml", cfg)

    # --- device + threading ---
    device = get_torch_device(args.device or cfg.get("device", "auto"))
    configure_threading(num_threads=1)

    # --- seeding ---
    set_global_seed(args.seed, deterministic_torch=args.deterministic_torch)

    # --- envs ---
    stack = build_wrapper_stack(cfg)
    env = make_vec_env(
        env_id=cfg["env_id"],
        n_envs=int(cfg.get("n_envs", 1)),
        base_seed=args.seed * 1000,
        vec_type=cfg.get("vec_type", "subproc"),
        monitor_dir=str(run_dir / "monitor"),
        wrapper_stack=stack,
        normalize_obs=bool(cfg.get("normalize_obs", False)),
        normalize_reward=bool(cfg.get("normalize_reward", False)),
    )
    # Always evaluate on the NOMINAL env (no DR, no fault) — this isolates
    # whether the policy has generalized, vs. just specialized to its training
    # distribution.
    #
    # IMPORTANT (bug fix #4 in nb 05): when training env uses VecNormalize,
    # the EvalCallback calls `sync_envs_normalization(train, eval)`, which
    # walks BOTH wrapper stacks in lockstep and demands a matching
    # VecEnvWrapper at every depth. `make_vec_env` produces a stack of
    #
    #     VecNormalize → VecMonitor → SubprocVecEnv → [Monitor(gym.Env)]
    #
    # so the eval env must mirror EVERY layer (not just the outermost),
    # otherwise the walk asserts on DummyVecEnv (which is a VecEnv but
    # NOT a VecEnvWrapper) at the middle layer. The mirror is:
    #
    #     VecNormalize(training=False) → VecMonitor → DummyVecEnv → [Monitor(...)]
    #
    # with obs_rms/ret_rms shared by reference so sync is a no-op.
    # When the train env is VecNormalize, the eval env is wrapped in VecMonitor
    # below — so we must NOT also wrap the inner gym.Env in Monitor, otherwise
    # episode stats are double-counted (SB3 emits a UserWarning and per-episode
    # std is biased low).
    will_vec_wrap = isinstance(env, VecNormalize)
    eval_env_raw = make_single_eval_env(
        env_id=cfg["env_id"],
        seed=args.seed + 7777,
        wrapper_stack=None,
        env_kwargs=None,
        add_monitor=not will_vec_wrap,
    )
    if will_vec_wrap:
        eval_vec = DummyVecEnv([lambda: eval_env_raw])
        eval_vec = VecMonitor(eval_vec)
        eval_env = VecNormalize(
            eval_vec,
            training=False,
            norm_obs=env.norm_obs,
            norm_reward=False,            # eval reward must be raw, always
            clip_obs=env.clip_obs,
            gamma=env.gamma,
        )
        # Share running stats by reference so sync_envs_normalization is a no-op.
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms
    else:
        eval_env = eval_env_raw

    # --- model ---
    model = build_model(
        cfg=cfg,
        env=env,
        device=device,
        seed=args.seed,
        tb_path=str(run_dir / "tb"),
    )

    # --- callbacks ---
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "eval"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(1, int(cfg.get("eval_freq", 25_000))),
        n_eval_episodes=int(cfg.get("n_eval_episodes", 20)),
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, int(cfg.get("checkpoint_freq", 100_000))),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ckpt",
        save_replay_buffer=False,
        save_vecnormalize=isinstance(env, VecNormalize),
    )
    callbacks = CallbackList([eval_cb, ckpt_cb])

    # --- meta ---
    meta = {
        "algo": cfg["algo"],
        "env_id": cfg["env_id"],
        "seed": args.seed,
        "device": str(device),
        "device_info": describe_device(device),
        "python": sys.version,
        "platform": platform.platform(),
        "gymnasium": gym.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    dump_json(run_dir / "run_meta.json", meta)
    print("=" * 80)
    print(json.dumps(meta, indent=2))
    print("=" * 80)

    # --- train ---
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=int(cfg["total_timesteps"]),
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("[train] interrupted — saving partial model.")
    elapsed = time.time() - t0

    # --- finalize ---
    model.save(str(run_dir / "final_model.zip"))
    if isinstance(env, VecNormalize):
        env.save(str(run_dir / "vecnormalize.pkl"))

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["wallclock_seconds"] = float(elapsed)
    dump_json(run_dir / "run_meta.json", meta)

    # quick sanity eval
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    summary = {
        "final_eval_mean_return": float(mean_r),
        "final_eval_std_return": float(std_r),
        "wallclock_seconds": float(elapsed),
    }
    dump_json(run_dir / "final_summary.json", summary)
    print(json.dumps(summary, indent=2))

    env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
