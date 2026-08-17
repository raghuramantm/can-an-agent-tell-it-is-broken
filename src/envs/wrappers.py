"""
Environment wrappers for the LunarLander actuator-fault experiments.

Two wrapper families are exposed:

1. `ActuatorFaultWrapper`
   ------------------------
   Injects a deterministic *step-fault* in actuator response at episode step
   `fault_step`. After the fault, applied thrust is multiplied by `fault_gain`
   (1.0 = nominal, 0.0 = total failure of that thruster, 0.5 = 50 % drop).
   Supports per-thruster gains for the continuous control variant.

   Mathematically, the post-fault transition becomes
       a_applied = M(t) ⊙ a_commanded,
   where M(t) is a diagonal gain matrix that switches at t = fault_step.

2. `DomainRandomizationWrapper`
   ------------------------------
   Per-episode resampling of physics parameters within configurable ranges:
   gravity ∈ [g_lo, g_hi], main-engine power, side-engine power, wind power,
   turbulence power. Implements DR à la Tobin et al. (2017) for the baseline
   robustness control.

Both wrappers are pure-Python and impose no Box2D recompilation cost — they
only mutate (a) action values passed to step() and (b) env construction kwargs
on reset().

Notes
-----
- LunarLander-v3 exposes `gravity`, `enable_wind`, `wind_power`,
  `turbulence_power` as constructor kwargs. Engine power constants
  (`MAIN_ENGINE_POWER`, `SIDE_ENGINE_POWER`) live as module-level globals
  inside `gymnasium.envs.box2d.lunar_lander`; we override them on the
  unwrapped env on every reset for DR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ------------------------------------------------------------------- #
# Actuator fault
# ------------------------------------------------------------------- #
@dataclass
class ActuatorFaultSpec:
    """
    Parameters
    ----------
    fault_step : int
        Time step (within the episode) at which the fault activates.
    fault_gain : float | np.ndarray
        Post-fault multiplier on commanded thrust. Scalar applies to all
        thrusters; array of shape (action_dim,) applies per-thruster.
    discrete_drop_action : int | None
        For the discrete env variant, optionally suppress this action id
        post-fault (e.g. 2 = main engine).
    """
    fault_step: int = 200
    fault_gain: float | np.ndarray = 0.5
    discrete_drop_action: int | None = None


class ActuatorFaultWrapper(gym.Wrapper):
    """
    Injects a step-fault into actuator response.

    For the *continuous* env, `fault_gain` scales the action vector
    element-wise after `fault_step`.

    For the *discrete* env, `discrete_drop_action`, if set, is rewritten to
    the no-op action (0) after `fault_step`, simulating a stuck thruster.
    """

    def __init__(self, env: gym.Env, spec: ActuatorFaultSpec, seed: int = 0):
        super().__init__(env)
        self.fault_spec = spec
        self._t = 0
        self._is_continuous = isinstance(self.action_space, spaces.Box)
        # RNG for the probabilistic discrete-action drop. Re-seeded on reset
        # if a seed is provided so episodes are reproducible.
        self._fault_rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, **kwargs):
        self._t = 0
        if seed is not None:
            self._fault_rng = np.random.default_rng(seed)
            return self.env.reset(seed=seed, **kwargs)
        return self.env.reset(**kwargs)

    def step(self, action):
        a = action
        if self._t >= self.fault_spec.fault_step:
            if self._is_continuous:
                gain = np.asarray(self.fault_spec.fault_gain, dtype=np.float32)
                a = np.asarray(action, dtype=np.float32) * gain
                a = np.clip(a, self.action_space.low, self.action_space.high)
            else:
                # NEW probabilistic semantics. `fault_gain` for a discrete env
                # is the per-step probability that a commanded main-engine
                # action SUCCEEDS:
                #   gain = 1.0 → never fail (nominal)
                #   gain = 0.5 → 50% chance the engine fires when commanded
                #   gain = 0.0 → permanent failure (matches the old binary behaviour)
                # This puts discrete and continuous on a shared gain axis so
                # the cell 6.6 sweep produces real degradation curves.
                drop = self.fault_spec.discrete_drop_action
                if drop is not None and int(action) == int(drop):
                    p_success = float(np.asarray(self.fault_spec.fault_gain).mean())
                    if self._fault_rng.random() > p_success:
                        a = 0  # commanded action dropped to no-op
        self._t += 1
        return self.env.step(a)


# ------------------------------------------------------------------- #
# Gradual (drifting) actuator fault
# ------------------------------------------------------------------- #
@dataclass
class GradualFaultSpec:
    """
    Time-varying actuator gain g(t), modelling progressive degradation
    (bearing wear, motor winding decay, thruster erosion) rather than a
    step fault. Structurally analogous to gate-fidelity drift in NISQ
    quantum devices, where fidelity decays continuously between
    calibrations rather than failing abruptly.

    Profiles
    --------
    linear:       g(t) = max(g_min, 1 - rate * (t - onset_step))   for t >= onset_step
    exponential:  g(t) = g_min + (1 - g_min) * exp(-rate * (t - onset_step))

    Parameters
    ----------
    onset_step : int
        Step at which degradation begins (g = 1 before this).
    profile : str
        'linear' or 'exponential'.
    rate : float
        Degradation rate. Linear: gain lost per step (e.g. 0.005 loses
        50% of thrust in 100 steps). Exponential: decay constant lambda.
    g_min : float
        Floor on the gain (0.0 = eventual total failure).
    discrete_drop_action : int | None
        For discrete envs, the action id subject to probabilistic drop
        with per-step success probability g(t) (same semantics as
        ActuatorFaultSpec).
    """
    onset_step: int = 100
    profile: str = "linear"
    rate: float = 0.005
    g_min: float = 0.0
    discrete_drop_action: int | None = None

    def gain_at(self, t: int) -> float:
        """Current gain g(t). Pure function of the episode step."""
        if t < self.onset_step:
            return 1.0
        dt = t - self.onset_step
        if self.profile == "linear":
            return max(self.g_min, 1.0 - self.rate * dt)
        elif self.profile == "exponential":
            return self.g_min + (1.0 - self.g_min) * float(np.exp(-self.rate * dt))
        raise ValueError(f"unknown profile={self.profile}")


class GradualFaultWrapper(gym.Wrapper):
    """
    Applies a time-varying gain g(t) to the action channel.

    Continuous env: a_applied = g(t) * a_commanded (all dims; pass a
    per-dim gain by composing with ActuatorFaultWrapper if needed).
    Discrete env: the drop action succeeds with probability g(t).

    Exposes `self.current_gain` for logging/analysis.
    """

    def __init__(self, env: gym.Env, spec: GradualFaultSpec, seed: int = 0):
        super().__init__(env)
        self.gradual_spec = spec
        self._t = 0
        self._is_continuous = isinstance(self.action_space, spaces.Box)
        self._fault_rng = np.random.default_rng(seed)
        self.current_gain: float = 1.0

    def reset(self, *, seed: int | None = None, **kwargs):
        self._t = 0
        self.current_gain = 1.0
        if seed is not None:
            self._fault_rng = np.random.default_rng(seed)
            return self.env.reset(seed=seed, **kwargs)
        return self.env.reset(**kwargs)

    def step(self, action):
        g = self.gradual_spec.gain_at(self._t)
        self.current_gain = g
        a = action
        if g < 1.0:
            if self._is_continuous:
                a = np.asarray(action, dtype=np.float32) * np.float32(g)
                a = np.clip(a, self.action_space.low, self.action_space.high)
            else:
                drop = self.gradual_spec.discrete_drop_action
                if drop is not None and int(action) == int(drop):
                    if self._fault_rng.random() > g:
                        a = 0
        self._t += 1
        return self.env.step(a)


# ------------------------------------------------------------------- #
# Evaluation-time observation normalisation (VecNormalize replay)
# ------------------------------------------------------------------- #
class ObsNormWrapper(gym.ObservationWrapper):
    """
    Replays a saved VecNormalize observation normalisation on a single
    (non-vectorised) eval env. Required for any agent trained with
    normalize_obs=True (PPO_DR): calling model.predict on RAW observations
    silently degrades the policy to ~-95 nominal (NB11 Bug 1; recurred in
    the NB16-19 harnesses).

    Exposes `last_raw_obs` (the un-normalised observation) for consumers
    that need physical state — e.g. the Gymnasium heuristic controller in
    NB17, which must see raw positions/velocities, not normalised ones.
    """

    def __init__(self, env: gym.Env, vecnorm_path, clip_obs: float = 10.0):
        super().__init__(env)
        import pickle
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        self.obs_rms = vn.obs_rms
        self.clip_obs = clip_obs
        self.last_raw_obs = None

    def observation(self, obs):
        self.last_raw_obs = np.asarray(obs, dtype=np.float32)
        norm = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(norm, -self.clip_obs, self.clip_obs).astype(np.float32)


# ------------------------------------------------------------------- #
# Stochastic drift (OU process around a deterministic trend) — NISQ analogue
# ------------------------------------------------------------------- #
@dataclass
class StochasticDriftSpec:
    """
    Ornstein-Uhlenbeck fluctuation around a deterministic degradation trend:

        mu(t)    = deterministic mean trend (linear ramp, as GradualFaultSpec)
        g(t+1)   = g(t) + theta * (mu(t) - g(t)) + sigma * eps_t,  eps_t ~ N(0,1)
        g(t)     clipped to [0, 1]

    sigma = 0 recovers the deterministic GradualFaultWrapper exactly.
    This is the structural analogue of gate-fidelity drift in NISQ devices,
    where fidelity fluctuates around a decaying mean between calibrations
    and transiently *recovers* — the regime that stresses change-point
    detectors with intermittent-fault behaviour.

    Parameters
    ----------
    onset_step : int      — trend onset (mu = 1 before).
    trend_rate : float    — linear mean-trend slope (gain lost per step).
    g_min : float         — floor on the mean trend.
    theta : float         — OU mean-reversion strength (0..1).
    sigma : float         — per-step noise std.
    discrete_drop_action : int | None — as ActuatorFaultSpec.
    """
    onset_step: int = 100
    trend_rate: float = 0.005
    g_min: float = 0.0
    theta: float = 0.15
    sigma: float = 0.02
    discrete_drop_action: int | None = None

    def mu_at(self, t: int) -> float:
        if t < self.onset_step:
            return 1.0
        return max(self.g_min, 1.0 - self.trend_rate * (t - self.onset_step))


class StochasticDriftWrapper(gym.Wrapper):
    """OU-noisy actuator gain. Exposes `gain_trace` (per-step list) for analysis."""

    def __init__(self, env: gym.Env, spec: StochasticDriftSpec, seed: int = 0):
        super().__init__(env)
        self.drift_spec = spec
        self._t = 0
        self._g = 1.0
        self._is_continuous = isinstance(self.action_space, spaces.Box)
        self._rng = np.random.default_rng(seed)
        self.gain_trace: list[float] = []

    def reset(self, *, seed: int | None = None, **kwargs):
        self._t = 0
        self._g = 1.0
        self.gain_trace = []
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            return self.env.reset(seed=seed, **kwargs)
        return self.env.reset(**kwargs)

    def step(self, action):
        s = self.drift_spec
        mu = s.mu_at(self._t)
        noise = s.sigma * self._rng.standard_normal() if self._t >= s.onset_step else 0.0
        self._g = float(np.clip(self._g + s.theta * (mu - self._g) + noise, 0.0, 1.0))
        self.gain_trace.append(self._g)
        a = action
        if self._g < 1.0:
            if self._is_continuous:
                a = np.asarray(action, dtype=np.float32) * np.float32(self._g)
                a = np.clip(a, self.action_space.low, self.action_space.high)
            else:
                drop = s.discrete_drop_action
                if drop is not None and int(action) == int(drop):
                    if self._rng.random() > self._g:
                        a = 0
        self._t += 1
        return self.env.step(a)


# ------------------------------------------------------------------- #
# Environment change (world fault) — attribution control
# ------------------------------------------------------------------- #
@dataclass
class EnvironmentChangeSpec:
    """
    Mid-episode change to the WORLD (not the agent's actuators): wind and
    turbulence switch on at `change_step`. The attribution counterpart to
    ActuatorFaultSpec — used to test whether internal signals can
    distinguish 'my body changed' from 'the world changed'.

    LunarLander applies wind per-step from `env.unwrapped.wind_power` /
    `turbulence_power`, so mutating them mid-episode takes effect
    immediately. The env must be constructed with `enable_wind=True`
    and wind_power=0 pre-change (idle wind machine), because
    `enable_wind` also gates the wind state initialisation on reset.
    """
    change_step: int = 200
    wind_power: float = 15.0
    turbulence_power: float = 1.5


class EnvironmentChangeWrapper(gym.Wrapper):
    """Switches wind on at `change_step`. Construct env with enable_wind=True, wind_power=0.0."""

    def __init__(self, env: gym.Env, spec: EnvironmentChangeSpec):
        super().__init__(env)
        self.change_spec = spec
        self._t = 0

    def reset(self, *, seed: int | None = None, **kwargs):
        self._t = 0
        u = self.env.unwrapped
        u.wind_power = 0.0
        u.turbulence_power = 0.0
        if seed is not None:
            return self.env.reset(seed=seed, **kwargs)
        return self.env.reset(**kwargs)

    def step(self, action):
        if self._t == self.change_spec.change_step:
            u = self.env.unwrapped
            u.wind_power = self.change_spec.wind_power
            u.turbulence_power = self.change_spec.turbulence_power
        self._t += 1
        return self.env.step(action)


# ------------------------------------------------------------------- #
# Domain randomization
# ------------------------------------------------------------------- #
@dataclass
class DomainRandomizationSpec:
    """
    Per-episode physics resampling ranges. All ranges are inclusive on both
    ends; `None` keeps the Box2D default.
    """
    gravity_range: tuple[float, float] = (-12.0, -8.0)        # default -10
    main_engine_power_range: tuple[float, float] = (11.0, 17.0)   # default 13
    side_engine_power_range: tuple[float, float] = (0.4, 0.8)     # default 0.6
    wind_power_range: tuple[float, float] | None = (5.0, 20.0)
    turbulence_power_range: tuple[float, float] | None = (0.5, 2.0)
    enable_wind: bool = True
    seed: int | None = None
    log_per_episode: bool = False


class DomainRandomizationWrapper(gym.Wrapper):
    """
    Resamples physics parameters at every `reset()`. The wrapper mutates
    (a) `env.unwrapped.gravity` / wind fields, and (b) the module-level
    engine-power constants used by the Box2D step function.

    NB: Engine-power overrides are *process-global* (they live on the
    lunar_lander module). When using SubprocVecEnv this is fine — each
    subprocess has its own module copy. When using DummyVecEnv on a single
    process across multiple wrapped envs, only the most recently reset env's
    powers are active at any given moment. Always prefer SubprocVecEnv for DR.
    """

    def __init__(self, env: gym.Env, spec: DomainRandomizationSpec):
        super().__init__(env)
        # NB: `spec` collides with gym.Env.spec — store under a private name.
        self.dr_spec = spec
        self._rng = np.random.default_rng(spec.seed)
        self.last_sample: dict[str, float] = {}

    def _sample(self) -> dict[str, float]:
        s = self.dr_spec
        sample: dict[str, float] = {
            "gravity": float(self._rng.uniform(*s.gravity_range)),
            "main_engine_power": float(self._rng.uniform(*s.main_engine_power_range)),
            "side_engine_power": float(self._rng.uniform(*s.side_engine_power_range)),
        }
        if s.wind_power_range is not None:
            sample["wind_power"] = float(self._rng.uniform(*s.wind_power_range))
        if s.turbulence_power_range is not None:
            sample["turbulence_power"] = float(self._rng.uniform(*s.turbulence_power_range))
        return sample

    def reset(self, **kwargs):
        sample = self._sample()
        unwrapped = self.env.unwrapped

        # (a) physics fields living on the env instance
        unwrapped.gravity = sample["gravity"]
        if hasattr(unwrapped, "enable_wind"):
            unwrapped.enable_wind = bool(self.dr_spec.enable_wind)
        if "wind_power" in sample and hasattr(unwrapped, "wind_power"):
            unwrapped.wind_power = sample["wind_power"]
        if "turbulence_power" in sample and hasattr(unwrapped, "turbulence_power"):
            unwrapped.turbulence_power = sample["turbulence_power"]

        # (b) module-level engine power constants (Box2D lookups happen there)
        from gymnasium.envs.box2d import lunar_lander as _ll
        _ll.MAIN_ENGINE_POWER = sample["main_engine_power"]
        _ll.SIDE_ENGINE_POWER = sample["side_engine_power"]

        self.last_sample = sample
        if self.dr_spec.log_per_episode:
            print(f"[DR] reset with {sample}")
        return self.env.reset(**kwargs)


# ------------------------------------------------------------------- #
# Convenience composer
# ------------------------------------------------------------------- #
@dataclass
class WrapperStack:
    domain_randomization: DomainRandomizationSpec | None = None
    actuator_fault: ActuatorFaultSpec | None = None
    gradual_fault: GradualFaultSpec | None = None
    stochastic_drift: StochasticDriftSpec | None = None
    environment_change: EnvironmentChangeSpec | None = None
    extra: list[Any] = field(default_factory=list)


def apply_wrappers(env: gym.Env, stack: WrapperStack) -> gym.Env:
    if stack.domain_randomization is not None:
        env = DomainRandomizationWrapper(env, stack.domain_randomization)
    if stack.actuator_fault is not None:
        env = ActuatorFaultWrapper(env, stack.actuator_fault)
    if stack.gradual_fault is not None:
        env = GradualFaultWrapper(env, stack.gradual_fault)
    if stack.stochastic_drift is not None:
        env = StochasticDriftWrapper(env, stack.stochastic_drift)
    if stack.environment_change is not None:
        env = EnvironmentChangeWrapper(env, stack.environment_change)
    for w in stack.extra:
        env = w(env)
    return env
