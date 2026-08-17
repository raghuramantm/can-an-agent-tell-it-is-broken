"""
Watch a trained agent play LunarLander — live pygame window or recorded MP4.

Usage
-----
    # Live window (default — opens a pygame window)
    python -m src.watch --run runs/ppo__LunarLander-v3__seed1__20260621T091339Z

    # Choose checkpoint instead of final model
    python -m src.watch --run runs/ppo__... --model best     # uses eval/best_model.zip

    # Apply a fault at eval time (shows visually what the eval numbers mean)
    python -m src.watch --run runs/ppo__... --fault-step 200 --fault-gain 0.5

    # Record MP4 instead of live window (requires `pip install imageio-ffmpeg`)
    python -m src.watch --run runs/ppo__... --mp4 logs/ppo_seed1.mp4 --episodes 3

    # Run multiple episodes
    python -m src.watch --run runs/ppo__... --episodes 5

Notes
-----
- Use `--seed` to control which trajectories you see. Same seed → same
  starting condition, same deterministic policy → reproducible demo clips.
- For the supervisor demo, record an MP4 of (a) nominal PPO landing
  cleanly, then (b) the same agent crashing under a 50 % thrust fault.
  That contrast is what makes the thesis question concrete.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC

from src.envs.wrappers import (
    ActuatorFaultSpec,
    ActuatorFaultWrapper,
    DomainRandomizationSpec,
    DomainRandomizationWrapper,
)


ALGO_LOADERS = {"ppo": PPO, "sac": SAC, "ppo_dr": PPO}


def _load_model(run_dir: Path, which: str):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    Loader = ALGO_LOADERS[cfg["algo"].lower()]
    if which == "best":
        model_path = run_dir / "eval" / "best_model.zip"
    else:
        model_path = run_dir / "final_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"no model at {model_path}")
    print(f"[watch] loading {model_path}")
    model = Loader.load(str(model_path), device="cpu")
    return cfg, model


def _build_env(
    env_id: str,
    render_mode: str,
    seed: int,
    fault_step: int | None,
    fault_gain: float,
    discrete_drop_action: int | None,
    dr: bool,
) -> gym.Env:
    env = gym.make(env_id, render_mode=render_mode)
    env.reset(seed=seed)
    if dr:
        env = DomainRandomizationWrapper(env, DomainRandomizationSpec(seed=seed))
    if fault_step is not None:
        is_discrete = not isinstance(env.action_space, gym.spaces.Box)
        env = ActuatorFaultWrapper(
            env,
            ActuatorFaultSpec(
                fault_step=int(fault_step),
                fault_gain=float(fault_gain),
                discrete_drop_action=(
                    int(discrete_drop_action) if discrete_drop_action is not None
                    else (2 if is_discrete else None)  # main engine for LunarLander
                ),
            ),
        )
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Watch a trained agent play.")
    parser.add_argument("--run", required=True, type=str, help="Run directory.")
    parser.add_argument("--model", choices=("final", "best"), default="final")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--mp4", type=str, default=None,
                        help="If set, record to MP4 instead of opening a window.")
    parser.add_argument("--fps", type=int, default=50,
                        help="MP4 framerate (ignored for live window).")
    parser.add_argument("--fault-step", type=int, default=None)
    parser.add_argument("--fault-gain", type=float, default=0.5)
    parser.add_argument("--discrete-drop-action", type=int, default=None,
                        help="Default: 2 (main engine) on discrete envs.")
    parser.add_argument("--dr", action="store_true",
                        help="Sample physics each episode from DR distribution.")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args(argv)

    run_dir = Path(args.run)
    cfg, model = _load_model(run_dir, args.model)

    render_mode = "rgb_array" if args.mp4 else "human"
    env = _build_env(
        env_id=cfg["env_id"],
        render_mode=render_mode,
        seed=args.seed,
        fault_step=args.fault_step,
        fault_gain=args.fault_gain,
        discrete_drop_action=args.discrete_drop_action,
        dr=args.dr,
    )

    # MP4 path: collect frames episode-by-episode, write at end.
    frames: list[np.ndarray] = []

    print(f"[watch] env={cfg['env_id']}  algo={cfg['algo']}  seed={args.seed}"
          f"  fault_step={args.fault_step}  dr={args.dr}")

    returns = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            total_r = 0.0
            steps = 0
            done = False
            while not done:
                # Drain pygame events so the OS does not flag the window
                # as "not responding" during long episodes. Without this,
                # closing via the red traffic-light raises a stale GLib
                # error that surfaces as a notebook cell exception.
                if render_mode == "human":
                    try:
                        import pygame
                        for _ev in pygame.event.get():
                            if _ev.type == pygame.QUIT:
                                raise KeyboardInterrupt(
                                    "[watch] window closed by user")
                    except ImportError:
                        pass
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, r, term, trunc, info = env.step(action)
                total_r += float(r)
                steps += 1
                if render_mode == "rgb_array":
                    frames.append(env.render())
                else:
                    time.sleep(1.0 / args.fps)
                done = bool(term or trunc)
            returns.append(total_r)
            print(f"  episode {ep+1:>2d}/{args.episodes}  return = {total_r:+8.2f}  "
                  f"length = {steps:>4d}")
    except KeyboardInterrupt as e:
        print(str(e))
    finally:
        # Always close env + pygame so the cell exits cleanly even if the
        # user closes the window mid-episode.
        try:
            env.close()
        except Exception:
            pass
        if render_mode == "human":
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass

    print(f"\n[watch] mean return over {len(returns)} ep: "
          f"{np.mean(returns):+.2f} ± {np.std(returns, ddof=1) if len(returns)>1 else 0:.2f}")

    if args.mp4:
        try:
            import imageio.v2 as imageio
        except ImportError:
            print("[watch] imageio not installed. Run:")
            print("    pip install imageio imageio-ffmpeg")
            return 2
        out_path = Path(args.mp4)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[watch] writing {len(frames)} frames → {out_path}")
        imageio.mimsave(out_path, frames, fps=args.fps, codec="libx264",
                        quality=8, macro_block_size=1)
        print(f"[watch] mp4 saved: {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
