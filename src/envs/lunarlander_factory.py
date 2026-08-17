"""
LunarLander environment factory.

Supports the two canonical variants:
    - LunarLander-v3            (Discrete(4))    → PPO discrete
    - LunarLanderContinuous-v3  (Box(2))         → SAC / PPO continuous

and three vectorization strategies:
    - "dummy"   single-process DummyVecEnv  (debugging)
    - "subproc" SubprocVecEnv               (default; CPU-side parallel rollouts)
    - "single"  one env (for evaluation)
"""
from __future__ import annotations

from typing import Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from .wrappers import WrapperStack, apply_wrappers


CONT_ENV_ID = "LunarLanderContinuous-v3"
DISC_ENV_ID = "LunarLander-v3"


def _make_single_env(
    env_id: str,
    seed: int,
    monitor_dir: str | None,
    wrapper_stack: WrapperStack | None,
    env_kwargs: dict | None,
) -> Callable[[], gym.Env]:
    """Thunk used by VecEnv constructors."""
    def _thunk() -> gym.Env:
        env = gym.make(env_id, **(env_kwargs or {}))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        if wrapper_stack is not None:
            env = apply_wrappers(env, wrapper_stack)
        if monitor_dir is not None:
            env = Monitor(env, filename=monitor_dir)
        return env
    return _thunk


def make_vec_env(
    env_id: str,
    n_envs: int = 8,
    base_seed: int = 0,
    vec_type: str = "subproc",
    monitor_dir: str | None = None,
    wrapper_stack: WrapperStack | None = None,
    env_kwargs: dict | None = None,
    normalize_obs: bool = False,
    normalize_reward: bool = False,
):
    """
    Build a SubprocVecEnv (default) of size `n_envs`. Each worker is seeded
    deterministically as base_seed + worker_idx.

    Returns a `VecEnv` (optionally wrapped in `VecNormalize`).
    """
    # SAFETY: DomainRandomizationWrapper mutates lunar_lander.MAIN_ENGINE_POWER
    # at MODULE level. Under SubprocVecEnv each worker has its own module copy
    # so it's safe; under DummyVecEnv (single process) the last env to reset
    # silently overwrites the previous one's settings, producing wrong rollouts.
    # We refuse to build a DummyVecEnv with DR active.
    if (wrapper_stack is not None
        and wrapper_stack.domain_randomization is not None
        and vec_type != "subproc"):
        raise ValueError(
            f"DomainRandomization requires vec_type='subproc' (got '{vec_type}'). "
            f"Module-level Box2D engine-power overrides are not isolated between "
            f"DummyVecEnv envs and would silently corrupt rollouts."
        )

    fns = [
        _make_single_env(
            env_id=env_id,
            seed=base_seed + i,
            monitor_dir=(monitor_dir + f"/worker_{i}" if monitor_dir else None),
            wrapper_stack=wrapper_stack,
            env_kwargs=env_kwargs,
        )
        for i in range(n_envs)
    ]
    if vec_type == "subproc":
        vec_env = SubprocVecEnv(fns, start_method="spawn")
    elif vec_type == "dummy":
        vec_env = DummyVecEnv(fns)
    else:
        raise ValueError(f"unknown vec_type={vec_type}")

    vec_env = VecMonitor(vec_env)  # episode reward/length book-keeping

    if normalize_obs or normalize_reward:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            clip_obs=10.0,
            gamma=0.99,
        )
    return vec_env


def make_single_eval_env(
    env_id: str,
    seed: int = 1234,
    wrapper_stack: WrapperStack | None = None,
    env_kwargs: dict | None = None,
    add_monitor: bool = True,
) -> gym.Env:
    """
    Plain single env for SB3 EvalCallback / evaluate.py.

    `add_monitor=True` (default) wraps in Monitor so episode-reward CSVs are
    written — correct for SB3 `evaluate_policy` on a bare gym.Env.

    `add_monitor=False` returns the unwrapped env — required when the caller
    will further wrap it in DummyVecEnv + VecMonitor (otherwise the Monitor
    statistics are double-counted and SB3 emits a UserWarning).
    """
    env = gym.make(env_id, **(env_kwargs or {}))
    env.reset(seed=seed)
    if wrapper_stack is not None:
        env = apply_wrappers(env, wrapper_stack)
    return Monitor(env) if add_monitor else env
