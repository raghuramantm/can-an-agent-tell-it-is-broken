"""
Tabular Q-learning for LunarLander-v3.

Purpose
-------
Provide the simplest possible RL baseline to complement the deep-RL agents
(PPO, SAC).  A tabular Q-table over a *discretised* 8-D observation space
answers two questions for the thesis:

  1. Does even the simplest value-based agent learn the task (at least partially)?
  2. When a fault is injected, do Q-value shifts at fault onset appear in a
     table lookup as they (hopefully) appear in deep networks?  If YES, the
     self-detection phenomenon is fundamental to value-based RL; if NO, it
     requires function approximation — both are interesting findings.

State discretisation
--------------------
LunarLander-v3 observations are 8-D continuous:
    [x, y, vx, vy, θ, ω, left_leg_contact, right_leg_contact]

We tile each dimension into ``n_bins`` equal-width bins clipped to a
configurable range.  Contact booleans (dims 6, 7) are binned into 2 bins
automatically.  The Q-table is a NumPy array of shape
    (n_bins**6 * 2 * 2, n_actions)
which, with ``n_bins=20``, is 20^6 × 4 × 4 ≈ 10.2 B entries — too large.

The practical solution is **coarse binning** (``n_bins=10`` →
10^6 × 4 × 4 = 16 M entries, ~128 MB at float32) or **tile coding** where
we sum over several coarse tilings.  We implement *single coarse tiling*
(10 bins per continuous dim, 2 per contact dim) which is the simplest option
that still learns a partially functional policy.

Mathematical formulation
------------------------
The Q-learning update (Watkins & Dayan, 1992):

    Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]

with ε-greedy policy:

    π(a|s) = { 1 − ε + ε/|A|    if a = argmax Q(s, ·)
             { ε / |A|            otherwise

Epsilon is decayed linearly from ``epsilon_start`` to ``epsilon_end`` over
``epsilon_decay_steps`` environment steps.

Usage
-----
    python -m src.q_learning --config configs/q_learning_lunarlander.yaml --seed 0

    # Smoke test (fast, 5 000 steps)
    python -m src.q_learning --config configs/q_learning_lunarlander.yaml \\
           --seed 0 --total-timesteps 5000
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml

from src.envs.lunarlander_factory import make_single_eval_env
from src.envs.wrappers import WrapperStack, ActuatorFaultSpec
from src.utils.logging_utils import dump_json, dump_yaml, make_run_dir
from src.utils.seeding import set_global_seed


# ------------------------------------------------------------------- #
# State discretisation
# ------------------------------------------------------------------- #
@dataclass
class StateDiscretizer:
    """
    Tile each continuous dimension into ``n_bins`` equal-width buckets.

    LunarLander-v3 obs layout:
        [x, y, vx, vy, θ, ω, left_contact, right_contact]

    Contact booleans (indices 6, 7) are always binned into 2 bins.

    Parameters
    ----------
    n_bins : int
        Number of bins per continuous dimension (dims 0–5).
    obs_ranges : list of (lo, hi)
        Clipping ranges per continuous dimension.  Values outside are
        clipped to the nearest bin edge.
    """
    n_bins: int = 10
    obs_ranges: list[tuple[float, float]] = field(default_factory=lambda: [
        (-1.5, 1.5),    # x position
        (-0.5, 2.0),    # y position
        (-3.0, 3.0),    # x velocity
        (-3.0, 0.5),    # y velocity
        (-np.pi, np.pi),  # angle (radians)
        (-5.0, 5.0),    # angular velocity
    ])

    def __post_init__(self):
        # Precompute bin edges for each continuous dimension
        self._edges: list[np.ndarray] = []
        for lo, hi in self.obs_ranges:
            self._edges.append(np.linspace(lo, hi, self.n_bins + 1))
        # Multipliers for the flat index: dims 0-5 have n_bins each,
        # dims 6-7 have 2 each.
        self._mults = np.array([
            self.n_bins ** 5 * 4,  # dim 0
            self.n_bins ** 4 * 4,  # dim 1
            self.n_bins ** 3 * 4,  # dim 2
            self.n_bins ** 2 * 4,  # dim 3
            self.n_bins * 4,       # dim 4
            4,                     # dim 5
            2,                     # dim 6 (left contact)
            1,                     # dim 7 (right contact)
        ], dtype=np.int64)

    @property
    def n_states(self) -> int:
        """Total number of discrete states."""
        return self.n_bins ** 6 * 2 * 2

    def discretize(self, obs: np.ndarray) -> int:
        """
        Convert a continuous 8-D observation to a flat integer state index.

        Parameters
        ----------
        obs : np.ndarray, shape (8,)
            Raw LunarLander-v3 observation.

        Returns
        -------
        int
            Flat index into the Q-table.
        """
        indices = np.empty(8, dtype=np.int64)
        # Continuous dims (0–5): digitize + clip
        for d in range(6):
            val = np.clip(obs[d], self.obs_ranges[d][0], self.obs_ranges[d][1])
            idx = np.digitize(val, self._edges[d]) - 1
            indices[d] = np.clip(idx, 0, self.n_bins - 1)
        # Contact dims (6, 7): threshold at 0.5
        indices[6] = int(obs[6] > 0.5)
        indices[7] = int(obs[7] > 0.5)
        return int(np.dot(indices, self._mults))


# ------------------------------------------------------------------- #
# Q-learning agent
# ------------------------------------------------------------------- #
class QLearningAgent:
    """
    Tabular Q-learning with ε-greedy exploration.

    The Q-table is allocated lazily as a dict-of-arrays to handle the
    large state space (10^6 × 4 ≈ 16 M entries) without allocating
    unvisited states.  For evaluation, unvisited states return Q=0 for
    all actions.

    Attributes
    ----------
    q_table : dict[int, np.ndarray]
        state_index → Q-values array of shape (n_actions,).
    discretizer : StateDiscretizer
        Observation → state index converter.
    """

    def __init__(
        self,
        n_actions: int = 4,
        discretizer: StateDiscretizer | None = None,
        alpha: float = 0.1,
        gamma: float = 0.999,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 200_000,
        seed: int = 0,
    ):
        self.n_actions = n_actions
        self.discretizer = discretizer or StateDiscretizer()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.q_table: dict[int, np.ndarray] = {}
        self._rng = np.random.default_rng(seed)
        self._step = 0

    @property
    def epsilon(self) -> float:
        """Current ε (linearly decayed)."""
        frac = min(1.0, self._step / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * frac

    def _get_q(self, s: int) -> np.ndarray:
        """Return Q-values for state s, initialising to zeros if unseen."""
        if s not in self.q_table:
            self.q_table[s] = np.zeros(self.n_actions, dtype=np.float64)
        return self.q_table[s]

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """ε-greedy action selection."""
        s = self.discretizer.discretize(obs)
        q = self._get_q(s)
        if greedy or self._rng.random() > self.epsilon:
            return int(np.argmax(q))
        return int(self._rng.integers(self.n_actions))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """
        Single Q-learning update step.

        Returns
        -------
        float
            The TD error δ = r + γ max Q(s',·) − Q(s,a).
        """
        s = self.discretizer.discretize(obs)
        s_next = self.discretizer.discretize(next_obs)
        q = self._get_q(s)
        q_next = self._get_q(s_next)

        done = terminated or truncated
        target = reward + (0.0 if done else self.gamma * np.max(q_next))
        td_error = target - q[action]
        q[action] += self.alpha * td_error
        self._step += 1
        return float(td_error)

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return Q(s, ·) for a raw observation (read-only)."""
        s = self.discretizer.discretize(obs)
        return self._get_q(s).copy()

    def get_value(self, obs: np.ndarray) -> float:
        """Return V(s) = max_a Q(s, a)."""
        return float(np.max(self.get_q_values(obs)))

    def get_policy_entropy(self, obs: np.ndarray) -> float:
        """
        Entropy of the ε-greedy policy at current ε.

        For a greedy action a* and n_actions = |A|:
            p(a*) = 1 − ε + ε/|A|
            p(a≠a*) = ε/|A|
            H = −Σ p log p
        """
        eps = self.epsilon
        n = self.n_actions
        p_greedy = 1.0 - eps + eps / n
        p_other = eps / n
        h = -p_greedy * np.log(p_greedy + 1e-12)
        h -= (n - 1) * p_other * np.log(p_other + 1e-12)
        return float(h)

    # --- Persistence ---
    def save(self, path: str | Path) -> None:
        """Save Q-table and agent state to an .npz file."""
        path = Path(path)
        # Convert dict to parallel arrays for compact storage
        if self.q_table:
            states = np.array(list(self.q_table.keys()), dtype=np.int64)
            values = np.array(list(self.q_table.values()), dtype=np.float64)
        else:
            states = np.array([], dtype=np.int64)
            values = np.array([], dtype=np.float64).reshape(0, self.n_actions)
        np.savez_compressed(
            str(path),
            states=states,
            values=values,
            step=np.array([self._step]),
            n_actions=np.array([self.n_actions]),
            alpha=np.array([self.alpha]),
            gamma=np.array([self.gamma]),
            epsilon_start=np.array([self.epsilon_start]),
            epsilon_end=np.array([self.epsilon_end]),
            epsilon_decay_steps=np.array([self.epsilon_decay_steps]),
        )

    @classmethod
    def load(cls, path: str | Path, discretizer: StateDiscretizer | None = None) -> "QLearningAgent":
        """Load a saved agent from an .npz file."""
        data = np.load(str(path))
        agent = cls(
            n_actions=int(data["n_actions"][0]),
            discretizer=discretizer or StateDiscretizer(),
            alpha=float(data["alpha"][0]),
            gamma=float(data["gamma"][0]),
            epsilon_start=float(data["epsilon_start"][0]),
            epsilon_end=float(data["epsilon_end"][0]),
            epsilon_decay_steps=int(data["epsilon_decay_steps"][0]),
        )
        agent._step = int(data["step"][0])
        states = data["states"]
        values = data["values"]
        for i in range(len(states)):
            agent.q_table[int(states[i])] = values[i].copy()
        return agent

    @property
    def n_visited_states(self) -> int:
        return len(self.q_table)


# ------------------------------------------------------------------- #
# Training loop
# ------------------------------------------------------------------- #
def train_q_learning(
    agent: QLearningAgent,
    env: gym.Env,
    total_timesteps: int,
    eval_freq: int = 10_000,
    n_eval_episodes: int = 20,
    eval_env: gym.Env | None = None,
    checkpoint_freq: int = 50_000,
    checkpoint_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """
    Train the Q-learning agent with periodic evaluation.

    Returns
    -------
    dict
        Training history: episode_returns, eval_returns, visited_states,
        epsilon_log, td_error_log.
    """
    history = {
        "episode_returns": [],       # (episode_idx, cumulative_return)
        "eval_returns": [],          # (timestep, mean_return, std_return)
        "visited_states": [],        # (timestep, n_visited)
        "epsilon_log": [],           # (timestep, epsilon)
        "td_errors_per_episode": [], # (episode_idx, mean_abs_td_error)
    }

    obs, _ = env.reset()
    episode_return = 0.0
    episode_td_errors: list[float] = []
    episode_idx = 0

    for t in range(1, total_timesteps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        td_error = agent.update(obs, action, reward, next_obs, terminated, truncated)
        episode_return += reward
        episode_td_errors.append(abs(td_error))
        obs = next_obs

        if terminated or truncated:
            history["episode_returns"].append((episode_idx, float(episode_return)))
            history["td_errors_per_episode"].append(
                (episode_idx, float(np.mean(episode_td_errors)))
            )
            episode_return = 0.0
            episode_td_errors = []
            episode_idx += 1
            obs, _ = env.reset()

        # Periodic eval
        if t % eval_freq == 0:
            e_env = eval_env or env
            mean_r, std_r = _evaluate_agent(agent, e_env, n_eval_episodes)
            history["eval_returns"].append((t, mean_r, std_r))
            history["visited_states"].append((t, agent.n_visited_states))
            history["epsilon_log"].append((t, agent.epsilon))
            if verbose:
                print(
                    f"[Q-learn] step={t:>8d}  ε={agent.epsilon:.4f}  "
                    f"eval_mean={mean_r:+8.2f}  eval_std={std_r:7.2f}  "
                    f"states_visited={agent.n_visited_states}"
                )

        # Checkpoint
        if checkpoint_dir and t % checkpoint_freq == 0:
            ckpt_path = Path(checkpoint_dir) / f"q_table_step{t}.npz"
            agent.save(ckpt_path)

    return history


def _evaluate_agent(
    agent: QLearningAgent,
    env: gym.Env,
    n_episodes: int,
) -> tuple[float, float]:
    """Evaluate greedy policy over n_episodes, returning (mean, std)."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        done = False
        while not done:
            action = agent.select_action(obs, greedy=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            done = terminated or truncated
        returns.append(ep_ret)
    arr = np.array(returns)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(returns) > 1 else 0.0


# ------------------------------------------------------------------- #
# Evaluate with per-step signal logging (for self-detection)
# ------------------------------------------------------------------- #
def evaluate_with_signals(
    agent: QLearningAgent,
    env: gym.Env,
    n_episodes: int = 50,
    fault_step: int | None = None,
    fault_gain: float = 0.5,
    discrete_drop_action: int = 2,
) -> dict:
    """
    Run episodes and log per-step internal signals.

    For Q-learning the signals are:
        - V(s) = max_a Q(s, a)
        - Q(s, a_chosen) for the greedy action
        - policy entropy (ε-greedy, but with ε fixed at evaluation)
        - TD error (computed but NOT used for updates)

    If ``fault_step`` is provided, wraps the env in ActuatorFaultWrapper.

    Returns
    -------
    dict
        Keys: episode_returns, per_step (list of dicts with timestep,
        episode, value, q_chosen, entropy, td_error, reward, action).
    """
    from src.envs.wrappers import ActuatorFaultWrapper, ActuatorFaultSpec

    if fault_step is not None:
        spec = ActuatorFaultSpec(
            fault_step=fault_step,
            fault_gain=fault_gain,
            discrete_drop_action=discrete_drop_action,
        )
        eval_env = ActuatorFaultWrapper(env, spec)
    else:
        eval_env = env

    results: dict = {"episode_returns": [], "per_step": []}

    for ep in range(n_episodes):
        obs, _ = eval_env.reset()
        ep_ret = 0.0
        done = False
        t = 0
        while not done:
            q_vals = agent.get_q_values(obs)
            action = int(np.argmax(q_vals))  # greedy
            value = float(np.max(q_vals))
            q_chosen = float(q_vals[action])
            entropy = agent.get_policy_entropy(obs)

            next_obs, reward, terminated, truncated, _ = eval_env.step(action)

            # TD error (no update)
            next_q = agent.get_q_values(next_obs)
            done = terminated or truncated
            target = reward + (0.0 if done else agent.gamma * float(np.max(next_q)))
            td_error = target - q_chosen

            results["per_step"].append({
                "episode": ep,
                "timestep": t,
                "value": value,
                "q_chosen": q_chosen,
                "entropy": entropy,
                "td_error": td_error,
                "reward": reward,
                "action": action,
            })
            ep_ret += reward
            obs = next_obs
            t += 1

        results["episode_returns"].append(ep_ret)

    return results


# ------------------------------------------------------------------- #
# CLI entry point
# ------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tabular Q-learning on LunarLander-v3")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--runs-root", default="runs", type=str)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--tag", default=None, type=str)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.total_timesteps is not None:
        cfg["total_timesteps"] = args.total_timesteps

    # --- run dir ---
    run_dir = make_run_dir(
        root=args.runs_root,
        algo="q_learning",
        env_id=cfg["env_id"],
        seed=args.seed,
        tag=args.tag,
    )
    dump_yaml(run_dir / "config.yaml", cfg)

    # --- seed ---
    set_global_seed(args.seed)

    # --- env ---
    env = gym.make(cfg["env_id"])
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    eval_env = gym.make(cfg["env_id"])
    eval_env.reset(seed=args.seed + 7777)

    # --- discretiser ---
    disc_cfg = cfg.get("discretizer", {})
    discretizer = StateDiscretizer(
        n_bins=int(disc_cfg.get("n_bins", 10)),
        obs_ranges=[
            tuple(r) for r in disc_cfg.get("obs_ranges", [
                [-1.5, 1.5], [-0.5, 2.0], [-3.0, 3.0],
                [-3.0, 0.5], [-3.14159, 3.14159], [-5.0, 5.0],
            ])
        ],
    )

    # --- agent ---
    hp = cfg.get("hyperparameters", {})
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        discretizer=discretizer,
        alpha=float(hp.get("alpha", 0.1)),
        gamma=float(hp.get("gamma", 0.999)),
        epsilon_start=float(hp.get("epsilon_start", 1.0)),
        epsilon_end=float(hp.get("epsilon_end", 0.01)),
        epsilon_decay_steps=int(hp.get("epsilon_decay_steps", 200_000)),
        seed=args.seed,
    )

    # --- meta ---
    meta = {
        "algo": "q_learning",
        "env_id": cfg["env_id"],
        "seed": args.seed,
        "device": "cpu",
        "n_bins": discretizer.n_bins,
        "n_states_theoretical": discretizer.n_states,
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
    history = train_q_learning(
        agent=agent,
        env=env,
        total_timesteps=int(cfg["total_timesteps"]),
        eval_freq=int(cfg.get("eval_freq", 10_000)),
        n_eval_episodes=int(cfg.get("n_eval_episodes", 20)),
        eval_env=eval_env,
        checkpoint_freq=int(cfg.get("checkpoint_freq", 50_000)),
        checkpoint_dir=str(run_dir / "checkpoints"),
        verbose=True,
    )
    elapsed = time.time() - t0

    # --- save ---
    agent.save(run_dir / "q_table_final.npz")
    dump_json(run_dir / "training_history.json", {
        k: v for k, v in history.items()
    })

    # --- final eval ---
    mean_r, std_r = _evaluate_agent(agent, eval_env, n_episodes=50)
    summary = {
        "final_eval_mean_return": float(mean_r),
        "final_eval_std_return": float(std_r),
        "wallclock_seconds": float(elapsed),
        "n_visited_states": agent.n_visited_states,
        "total_possible_states": discretizer.n_states,
        "state_coverage_pct": 100.0 * agent.n_visited_states / discretizer.n_states,
        "final_epsilon": agent.epsilon,
    }
    dump_json(run_dir / "final_summary.json", summary)

    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["wallclock_seconds"] = elapsed
    dump_json(run_dir / "run_meta.json", meta)

    print("=" * 80)
    print(json.dumps(summary, indent=2))
    print("=" * 80)

    env.close()
    eval_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
