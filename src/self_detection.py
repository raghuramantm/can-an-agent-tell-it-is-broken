"""
Self-detection analysis: extracting internal RL signals under fault.

Purpose
-------
This module answers Starkey's Meeting 3 question: **"Can the learning model
itself tell you something's changed?"**

For each algorithm (PPO, SAC, Q-learning), we run trained models through
episodes with a step-fault injected at t=fault_step and log per-step
internal signals.  We then apply statistical change-point detection to
determine whether the fault onset is *visible* in the agent's own
representations — without any external forward model.

Signals extracted
-----------------
**PPO (discrete):**
    - V(s_t): value function estimate from the critic head
    - π-entropy H(π(·|s_t)): policy entropy from the actor head
    - Advantage Â_t = r_t + γ V(s_{t+1}) − V(s_t)  (TD(0) estimate)

**SAC (continuous):**
    - Q₁(s_t, a_t), Q₂(s_t, a_t): twin critic estimates for the chosen action
    - V(s_t) = min(Q₁, Q₂) − α log π(a_t|s_t)  (soft value)
    - Policy entropy (from log_std of the Gaussian policy)
    - TD error: r + γ V(s_{t+1}) − Q(s_t, a_t)

**Q-learning (tabular):**
    - V(s_t) = max_a Q(s_t, a)
    - Q(s_t, a_chosen)
    - TD error: r + γ max_a' Q(s_{t+1}, a') − Q(s_t, a_t)

Detection methods
-----------------
1. **Sliding-window z-test:**  Compare the mean of a trailing window of
   size w against the running mean/std of all *pre-fault* steps.  A
   detection is declared when |z| > z_threshold for z_consec consecutive
   steps.

2. **CUSUM (Cumulative Sum):**  Page (1954).  Accumulates positive and
   negative deviations from a reference mean; declares a change when the
   cumulative sum exceeds a threshold h.

Both methods are applied to each signal independently.  Detection latency
is measured as (t_detected − t_fault).

Mathematical formulation
------------------------
Sliding-window z-test:
    z_t = (x̄_{t-w:t} − μ_pre) / (σ_pre / √w)
    Detect when |z_t| > z_thresh for z_consec consecutive steps.

CUSUM (upper):
    S_t^+ = max(0, S_{t-1}^+ + (x_t − μ_0 − k))
    Detect when S_t^+ > h.

CUSUM (lower):
    S_t^- = max(0, S_{t-1}^- − (x_t − μ_0 + k))
    Detect when S_t^- > h.

Where μ_0 is the pre-fault reference mean, k is the allowance (typically
σ_pre / 2), and h is the decision threshold (typically 4–5 × σ_pre).

Usage
-----
    from src.self_detection import (
        extract_ppo_signals,
        extract_sac_signals,
        sliding_window_detection,
        cusum_detection,
    )

    # Extract signals from a trained PPO model under fault
    signals = extract_ppo_signals(model, env, fault_step=200, ...)

    # Detect change point in the value signal
    det = sliding_window_detection(signals["value"], fault_step=200)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC

from src.envs.lunarlander_factory import make_single_eval_env
from src.envs.wrappers import ActuatorFaultSpec, ActuatorFaultWrapper, WrapperStack


# ------------------------------------------------------------------- #
# PPO signal extraction
# ------------------------------------------------------------------- #
def extract_ppo_signals(
    model: PPO,
    env_id: str,
    fault_step: int = 200,
    fault_gain: float = 0.5,
    discrete_drop_action: int = 2,
    n_episodes: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run a trained PPO model through fault episodes, logging per-step signals.

    Returns a DataFrame with columns:
        episode, timestep, value, entropy, advantage, reward, action,
        obs_0..obs_7, fault_active
    """
    # Build fault environment
    stack = WrapperStack()
    stack.actuator_fault = ActuatorFaultSpec(
        fault_step=fault_step,
        fault_gain=fault_gain,
        discrete_drop_action=discrete_drop_action,
    )
    env = make_single_eval_env(env_id=env_id, seed=seed, wrapper_stack=stack)

    policy = model.policy
    policy.eval()
    device = next(policy.parameters()).device

    records: list[dict] = []

    for ep in range(n_episodes):
        obs_raw, _ = env.reset()
        done = False
        t = 0
        prev_value = None

        while not done:
            obs_tensor = torch.as_tensor(obs_raw, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                # Value estimate
                value = float(policy.predict_values(obs_tensor).item())

                # Policy distribution → entropy + action
                dist = policy.get_distribution(obs_tensor)
                entropy = float(dist.entropy().item())
                action_tensor = dist.mode()   # deterministic (greedy)
                action = int(action_tensor.item())

            # Step
            next_obs_raw, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Advantage (TD(0)): Â = r + γ V(s') − V(s)
            if not done:
                next_tensor = torch.as_tensor(
                    next_obs_raw, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    next_value = float(policy.predict_values(next_tensor).item())
                advantage = reward + model.gamma * next_value - value
            else:
                advantage = reward - value

            record = {
                "episode": ep,
                "timestep": t,
                "value": value,
                "entropy": entropy,
                "advantage": advantage,
                "reward": reward,
                "action": action,
                "fault_active": int(t >= fault_step),
            }
            for d in range(len(obs_raw)):
                record[f"obs_{d}"] = float(obs_raw[d])
            records.append(record)

            obs_raw = next_obs_raw
            prev_value = value
            t += 1

    env.close()
    return pd.DataFrame(records)


# ------------------------------------------------------------------- #
# SAC signal extraction
# ------------------------------------------------------------------- #
def extract_sac_signals(
    model: SAC,
    env_id: str,
    fault_step: int = 200,
    fault_gain: float = 0.5,
    n_episodes: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run a trained SAC model through fault episodes, logging per-step signals.

    Returns a DataFrame with columns:
        episode, timestep, q1, q2, q_min, soft_value, entropy, td_error,
        reward, action_0, action_1, obs_0..obs_7, fault_active
    """
    stack = WrapperStack()
    stack.actuator_fault = ActuatorFaultSpec(
        fault_step=fault_step,
        fault_gain=fault_gain,
        discrete_drop_action=None,  # continuous env — no discrete drop
    )
    env = make_single_eval_env(env_id=env_id, seed=seed, wrapper_stack=stack)

    policy = model.policy
    policy.eval()
    device = next(policy.parameters()).device

    # Retrieve log(alpha) for entropy temperature
    log_alpha = model.log_ent_coef
    if isinstance(log_alpha, torch.Tensor):
        alpha = float(log_alpha.exp().item())
    else:
        alpha = float(np.exp(log_alpha))

    records: list[dict] = []

    for ep in range(n_episodes):
        obs_raw, _ = env.reset()
        done = False
        t = 0

        while not done:
            obs_tensor = torch.as_tensor(
                obs_raw, dtype=torch.float32, device=device
            ).unsqueeze(0)

            with torch.no_grad():
                # Get action from policy (deterministic for eval)
                action_tensor = policy.actor.forward(obs_tensor, deterministic=True)

                # Twin Q-values
                q_values = policy.critic(obs_tensor, action_tensor)
                q1 = float(q_values[0].item())
                q2 = float(q_values[1].item())
                q_min = min(q1, q2)

                # Entropy via log_prob from the squashed Gaussian policy.
                # SB3's action_log_prob returns (sampled_action, log_prob).
                _, log_prob_tensor = policy.actor.action_log_prob(obs_tensor)
                log_prob = float(log_prob_tensor.item())
                entropy = -log_prob  # negated log-prob as entropy proxy

                # Soft value: V(s) = Q_min - α * log π(a|s)
                soft_value = q_min - alpha * log_prob

            action = action_tensor.cpu().numpy().flatten()
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD error: δ = r + γ V(s') − Q_min(s, a)
            if not done:
                next_tensor = torch.as_tensor(
                    obs_next, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    next_action = policy.actor.forward(next_tensor, deterministic=True)
                    next_q = policy.critic(next_tensor, next_action)
                    next_q_min = min(float(next_q[0].item()), float(next_q[1].item()))
                    _, next_log_prob_tensor = policy.actor.action_log_prob(next_tensor)
                    next_log_prob = float(next_log_prob_tensor.item())
                    next_soft_value = next_q_min - alpha * next_log_prob
                td_error = reward + model.gamma * next_soft_value - q_min
            else:
                td_error = reward - q_min

            record = {
                "episode": ep,
                "timestep": t,
                "q1": q1,
                "q2": q2,
                "q_min": q_min,
                "soft_value": soft_value,
                "entropy": entropy,
                "log_prob": float(log_prob),
                "td_error": td_error,
                "reward": reward,
                "action_0": float(action[0]),
                "action_1": float(action[1]),
                "fault_active": int(t >= fault_step),
            }
            for d in range(len(obs_raw)):
                record[f"obs_{d}"] = float(obs_raw[d])
            records.append(record)

            obs_raw = obs_next
            t += 1

    env.close()
    return pd.DataFrame(records)


# ------------------------------------------------------------------- #
# Change-point detection: sliding-window z-test
# ------------------------------------------------------------------- #
@dataclass
class DetectionResult:
    """Result of a change-point detection method."""
    detected: bool
    detection_step: int | None     # absolute step at which detection triggered
    detection_latency: int | None  # steps after fault_step
    statistic_series: np.ndarray   # z-scores or CUSUM values per step
    threshold: float
    method: str


def sliding_window_detection(
    signal: np.ndarray,
    fault_step: int,
    window: int = 20,
    z_threshold: float = 3.0,
    z_consec: int = 3,
) -> DetectionResult:
    """
    Sliding-window z-test for change detection.

    Uses the pre-fault portion (steps 0..fault_step-1) to estimate
    reference mean and std.  Then checks whether a trailing window of
    post-fault values deviates significantly.

    Parameters
    ----------
    signal : np.ndarray
        Per-step signal values (one episode or averaged across episodes).
    fault_step : int
        Known fault onset step.
    window : int
        Trailing-window size for the local mean.
    z_threshold : float
        Number of standard deviations for detection.
    z_consec : int
        Number of consecutive steps above threshold to confirm detection.

    Returns
    -------
    DetectionResult
    """
    n = len(signal)
    pre = signal[:fault_step]
    mu_pre = float(np.mean(pre))
    sigma_pre = float(np.std(pre, ddof=1)) if len(pre) > 1 else 1e-8
    if sigma_pre < 1e-12:
        sigma_pre = 1e-8  # avoid division by zero

    z_scores = np.zeros(n)
    for t in range(window, n):
        local_mean = np.mean(signal[t - window:t])
        z_scores[t] = (local_mean - mu_pre) / (sigma_pre / np.sqrt(window))

    # Find first run of z_consec consecutive steps exceeding threshold
    # AFTER the fault step
    consec_count = 0
    detection_step = None
    for t in range(fault_step, n):
        if abs(z_scores[t]) > z_threshold:
            consec_count += 1
            if consec_count >= z_consec:
                detection_step = t - z_consec + 1  # first step of the run
                break
        else:
            consec_count = 0

    return DetectionResult(
        detected=detection_step is not None,
        detection_step=detection_step,
        detection_latency=(detection_step - fault_step) if detection_step else None,
        statistic_series=z_scores,
        threshold=z_threshold,
        method="sliding_window_z",
    )


# ------------------------------------------------------------------- #
# Change-point detection: CUSUM
# ------------------------------------------------------------------- #
def cusum_detection(
    signal: np.ndarray,
    fault_step: int,
    k: float | None = None,
    h: float | None = None,
) -> DetectionResult:
    """
    Page's CUSUM (Cumulative Sum) change-point detection.

    Monitors both upward and downward shifts.  Uses the pre-fault portion
    to estimate reference mean and std, then sets defaults:
        k = σ_pre / 2   (allowance)
        h = 5 × σ_pre   (decision threshold)

    Parameters
    ----------
    signal : np.ndarray
        Per-step signal values.
    fault_step : int
        Known fault onset step.
    k : float, optional
        Allowance (slack) parameter.  Default: σ_pre / 2.
    h : float, optional
        Decision threshold.  Default: 5 × σ_pre.

    Returns
    -------
    DetectionResult
    """
    n = len(signal)
    pre = signal[:fault_step]
    mu_0 = float(np.mean(pre))
    sigma_pre = float(np.std(pre, ddof=1)) if len(pre) > 1 else 1e-8
    if sigma_pre < 1e-12:
        sigma_pre = 1e-8

    if k is None:
        k = sigma_pre / 2.0
    if h is None:
        h = 5.0 * sigma_pre

    s_pos = np.zeros(n)
    s_neg = np.zeros(n)

    for t in range(1, n):
        s_pos[t] = max(0.0, s_pos[t - 1] + (signal[t] - mu_0) - k)
        s_neg[t] = max(0.0, s_neg[t - 1] - (signal[t] - mu_0) - k)

    # Combine: take the max of upper and lower CUSUM
    cusum = np.maximum(s_pos, s_neg)

    # Detection: first post-fault step where cusum > h
    detection_step = None
    for t in range(fault_step, n):
        if cusum[t] > h:
            detection_step = t
            break

    return DetectionResult(
        detected=detection_step is not None,
        detection_step=detection_step,
        detection_latency=(detection_step - fault_step) if detection_step else None,
        statistic_series=cusum,
        threshold=h,
        method="cusum",
    )


# ------------------------------------------------------------------- #
# Aggregate analysis helper
# ------------------------------------------------------------------- #
def compute_episode_averaged_signals(
    df: pd.DataFrame,
    signal_columns: list[str],
    max_timestep: int | None = None,
) -> pd.DataFrame:
    """
    Average signals across episodes at each timestep.

    Parameters
    ----------
    df : pd.DataFrame
        Per-step data from extract_*_signals, with 'episode' and 'timestep'
        columns.
    signal_columns : list[str]
        Which signal columns to average (e.g. ['value', 'entropy', 'td_error']).
    max_timestep : int, optional
        Truncate to this many steps per episode.  If None, uses the minimum
        episode length across all episodes.

    Returns
    -------
    pd.DataFrame
        Columns: timestep, {signal}_mean, {signal}_std, {signal}_sem,
                 n_episodes, fault_active
    """
    if max_timestep is None:
        max_timestep = int(df.groupby("episode")["timestep"].max().min())

    df_trunc = df[df["timestep"] <= max_timestep].copy()
    grouped = df_trunc.groupby("timestep")

    records = []
    for t, grp in grouped:
        rec = {"timestep": int(t), "n_episodes": len(grp)}
        if "fault_active" in grp.columns:
            rec["fault_active"] = int(grp["fault_active"].iloc[0])
        for col in signal_columns:
            if col in grp.columns:
                vals = grp[col].values
                rec[f"{col}_mean"] = float(np.mean(vals))
                rec[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                rec[f"{col}_sem"] = rec[f"{col}_std"] / np.sqrt(len(vals))
        records.append(rec)

    return pd.DataFrame(records)


def run_detection_battery(
    avg_df: pd.DataFrame,
    signal_columns: list[str],
    fault_step: int,
    window: int = 20,
    z_threshold: float = 3.0,
    z_consec: int = 3,
) -> pd.DataFrame:
    """
    Run both sliding-window and CUSUM detection on each signal.

    Returns a summary table with one row per (signal, method).
    """
    rows = []
    for col in signal_columns:
        mean_col = f"{col}_mean"
        if mean_col not in avg_df.columns:
            continue
        signal = avg_df[mean_col].values

        # Sliding window
        sw = sliding_window_detection(
            signal, fault_step, window=window,
            z_threshold=z_threshold, z_consec=z_consec,
        )
        rows.append({
            "signal": col,
            "method": "sliding_window_z",
            "detected": sw.detected,
            "detection_step": sw.detection_step,
            "detection_latency": sw.detection_latency,
            "threshold": sw.threshold,
        })

        # CUSUM
        cs = cusum_detection(signal, fault_step)
        rows.append({
            "signal": col,
            "method": "cusum",
            "detected": cs.detected,
            "detection_step": cs.detection_step,
            "detection_latency": cs.detection_latency,
            "threshold": cs.threshold,
        })

    return pd.DataFrame(rows)


# =================================================================== #
# BUGFIX 2026-07-20 (BF-01): robust per-step standardisation
# -------------------------------------------------------------------
# SYMPTOM (confirmed from logs). Every nominal-relative z-test in this
# project standardised with z_t = (x_t - mu_t) / (sigma_t + 1e-8), scanning
# from t = 0. NB17 closed-loop logs show detectors firing at t = 2-3, i.e.
# BEFORE any fault can exist: 5 pre-onset firings in each detector-driven
# condition (C4, C6; both agents), contaminating 24% of PPO+DR's
# nominal-condition alarms and 17% of PPO's closed-loop "detections".
#
# MECHANISM (reproduced synthetically, 2026-07-20). NOT a literal
# sigma_t -> 0 divide-by-zero, which was the first hypothesis and did not
# reproduce. The actual cause is a reference/test mismatch at early steps:
# episode value estimates are tightly clustered in the first few steps
# (similar launch states), so sigma_t is small but finite; test episodes are
# drawn from a different seed base than the reference episodes, so an
# atypical initial condition lands many small-sigma units from mu_t and
# trips the 3-consecutive rule. In a synthetic replication with 25%
# out-of-reference launches this produced 5/20 pre-onset firings and
# inflated FPR from 0.00 to 0.35 (J 1.00 -> 0.65).
#
# FIX. Two guards, both applied here:
#   (a) sigma floor  - sigma_t is clipped below at `sigma_floor_frac` x the
#       signal's global nominal std, so a degenerate per-step sigma cannot
#       manufacture an unbounded z.
#   (b) burn-in      - the detector may not fire before `burn_in` steps.
#       With faults at t_f >= 100 this discards no genuine detection.
#
# SCOPE NOTE. For t_f = 0 experiments (NB13) the burn-in removes the first
# `burn_in` steps of evidence; those results are already characterised as
# whole-trajectory classification rather than change-point detection, so the
# interpretation is unchanged.
# =================================================================== #

BURN_IN_DEFAULT = 20
SIGMA_FLOOR_FRAC_DEFAULT = 0.05


def build_reference(
    nominal_df: pd.DataFrame,
    sig_col: str,
    step_col: str = "step",
    sigma_floor_frac: float = SIGMA_FLOOR_FRAC_DEFAULT,
) -> pd.DataFrame:
    """
    Per-step nominal reference (mu_t, sigma_t) with a variance floor.

    The floor is `sigma_floor_frac` x std of the signal pooled over all
    nominal steps, which keeps it on the signal's natural scale and makes
    the guard invariant to signal units (V(s) vs Q_min vs entropy).

    Returns a DataFrame indexed by step with columns ['mean', 'std'].
    """
    stats = nominal_df.groupby(step_col)[sig_col].agg(["mean", "std"])
    global_sd = float(nominal_df[sig_col].std())
    floor = max(sigma_floor_frac * global_sd, 1e-8)
    stats["std"] = stats["std"].fillna(floor).clip(lower=floor)
    stats["mean"] = stats["mean"].ffill()
    return stats


def episode_fires(
    ep_df: pd.DataFrame,
    sig_col: str,
    ref: pd.DataFrame,
    z_thresh: float,
    step_col: str = "step",
    n_consec: int = 3,
    burn_in: int = BURN_IN_DEFAULT,
    two_sided: bool = True,
) -> tuple[bool, int | None]:
    """
    Causal firing decision for one episode against a `build_reference` table.

    Fires when |z_t| (or z_t if one-sided) exceeds `z_thresh` for `n_consec`
    consecutive steps, ignoring all steps with t < `burn_in`.

    Steps beyond the reference horizon reuse the last available (mu, sigma) -
    the reference is built from nominal episodes which may be shorter than
    faulted ones.

    Returns (fired, first_firing_step).
    """
    steps = ep_df[step_col].to_numpy()
    x = ep_df[sig_col].to_numpy(dtype=float)
    if len(x) <= n_consec:
        return False, None

    t_max = int(ref.index.max())
    idx = np.minimum(steps, t_max)
    mu = ref["mean"].reindex(idx).to_numpy()
    sd = ref["std"].reindex(idx).to_numpy()

    z = (x - mu) / sd
    z = np.abs(z) if two_sided else z

    eligible = steps >= burn_in
    exceed = (z > z_thresh) & eligible & np.isfinite(z)

    if len(exceed) < n_consec:
        return False, None
    run = np.ones(len(exceed) - n_consec + 1, dtype=bool)
    for k in range(n_consec):
        run &= exceed[k:len(exceed) - n_consec + 1 + k]
    if not run.any():
        return False, None
    first = int(np.argmax(run))
    return True, int(steps[first + n_consec - 1])


def detection_rates(
    fault_df: pd.DataFrame,
    nominal_df: pd.DataFrame,
    sig_col: str,
    z_thresh: float,
    step_col: str = "step",
    episode_col: str = "episode",
    **kwargs,
) -> dict:
    """
    TPR / FPR / Youden's J for one (signal, threshold) using the guarded
    detector. `nominal_df` supplies both the reference and the control set.
    """
    ref = build_reference(nominal_df, sig_col, step_col=step_col)
    tpr = np.mean([episode_fires(e, sig_col, ref, z_thresh, step_col=step_col, **kwargs)[0]
                   for _, e in fault_df.groupby(episode_col)])
    fpr = np.mean([episode_fires(e, sig_col, ref, z_thresh, step_col=step_col, **kwargs)[0]
                   for _, e in nominal_df.groupby(episode_col)])
    return {"TPR": float(tpr), "FPR": float(fpr), "J": float(tpr - fpr)}
