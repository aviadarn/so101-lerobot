#!/usr/bin/env python3
"""Score every demonstration in a LeRobot dataset for quality, without looking at outcomes.

The premise this tool is built for: in a demonstration-driven product, every demonstration
an operator records nominally succeeded - they kept the ones that worked. The dataset
therefore carries no success label to sort on, and "which of these is worth training on"
is still an open question. This scores the trajectory itself.

Every signal is intrinsic to the recorded motion: how long it took, how direct it was, how
smooth, how much it hesitated, how many times the gripper changed its mind, and how far it
sits from what the rest of the fleet did. Nothing here reads the reward, the outcome, or
the object pose, so the same scoring runs unchanged on operator data where no such label
exists.

Signals (higher is worse in every case, after orientation):

  duration        frames in the episode; hesitation and retries make an episode longer
  path_ratio      joint-space path length / straight-line joint displacement; 1.0 is a
                  perfectly direct move, higher means wandering
  jerk            mean |third derivative| of joint position, scaled; the smoothness proxy
  pause_frac      fraction of frames whose joint velocity is near zero mid-episode
  grip_reversals  direction changes in the gripper channel; a clean demo closes once
  grasp_phase     when the gripper first closes, as a fraction of the episode; a late
                  grasp means the approach took most of the episode
  median_dev      distance from the fleet's median trajectory after time-normalisation

Raw signals are turned into robust z-scores (median / MAD) across the dataset, so the
score is relative to the fleet rather than to hand-picked absolute thresholds, and one
pathological episode cannot drag the scale. The reported score is a weighted sum of the
positive parts of those z-scores: 0.0 means "at or better than the fleet median on every
signal", and the worst contributing signal is reported as the rejection reason.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Weights are deliberately flat apart from the two signals that most directly describe a
# demonstration an operator would call sloppy: wandering and changing your mind on the grasp.
WEIGHTS = {
    "duration": 1.0,
    "path_ratio": 1.5,
    "jerk": 1.0,
    "pause_frac": 1.0,
    "grip_reversals": 1.5,
    "grasp_phase": 1.0,
    "median_dev": 1.0,
}
GRIPPER = 5          # gripper channel in the 6-dof state vector
RESAMPLE = 64        # common length for fleet-median comparison
VEL_EPS = 0.05       # deg/frame below which a joint is considered stationary


def load_episodes(root: str) -> dict[int, np.ndarray]:
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {root}/data")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"])
    out = {}
    for ep, g in df.groupby("episode_index"):
        out[int(ep)] = np.stack(g["observation.state"].values).astype(np.float64)
    return out


def resample(traj: np.ndarray, n: int = RESAMPLE) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(traj))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, traj[:, j]) for j in range(traj.shape[1])], axis=1)


def raw_signals(traj: np.ndarray) -> dict[str, float]:
    arm = traj[:, :GRIPPER]                     # gripper excluded: it is not part of the path
    grip = traj[:, GRIPPER]
    d = np.diff(arm, axis=0)

    path = float(np.linalg.norm(d, axis=1).sum())
    gi_ = grasp_index(grip)
    # Denominator is the task-minimal path: home -> grasp -> release, in two straight legs.
    # Using the episode's own bounding extent instead (the obvious choice) is wrong, because
    # a wandering demonstration inflates the extent as much as the path length and the ratio
    # barely moves - an injected wobble shifted it by only +0.44 sigma.
    if gi_ > 0:
        minimal = (float(np.linalg.norm(arm[gi_] - arm[0]))
                   + float(np.linalg.norm(arm[-1] - arm[gi_])))
    else:
        minimal = float(np.linalg.norm(arm[-1] - arm[0]))
    path_ratio = path / max(minimal, 1e-6)

    vel = np.linalg.norm(d, axis=1)
    acc = np.diff(d, axis=0)
    jrk = np.diff(acc, axis=0)
    jerk = float(np.linalg.norm(jrk, axis=1).mean()) if len(jrk) else 0.0

    # Ignore the settled tail: the arm legitimately stops at the end of the task.
    core = vel[: int(len(vel) * 0.9)] if len(vel) > 10 else vel
    pause_frac = float((core < VEL_EPS).mean()) if len(core) else 0.0

    # Gripper reversals: count sign changes of a deadbanded derivative, so sensor ripple on
    # a held-still gripper does not read as the operator changing their mind.
    gd = np.diff(grip)
    span = max(grip.max() - grip.min(), 1e-6)
    sig = np.sign(np.where(np.abs(gd) < 0.02 * span, 0.0, gd))
    sig = sig[sig != 0]
    grip_reversals = float((np.diff(sig) != 0).sum()) if len(sig) > 1 else 0.0

    gi = grasp_index(grip)
    grasp_phase = float(gi / len(grip)) if gi >= 0 else 1.0
    # Task difficulty, not demonstration quality: how far the arm had to travel to reach the
    # object. It is a covariate to normalise against, never a scored signal - see difficulty
    # normalisation in main(). Derived from the trajectory alone, so it needs no object pose
    # and works unchanged on operator data.
    reach = float(np.linalg.norm(arm[gi] - arm[0])) if gi >= 0 else float("nan")

    return dict(duration=float(len(traj)), path_ratio=path_ratio, jerk=jerk,
                pause_frac=pause_frac, grip_reversals=grip_reversals,
                grasp_phase=grasp_phase, reach=reach)


def grasp_index(grip: np.ndarray) -> int:
    """Frame where the gripper closes ON THE CUBE, or -1.

    The episode starts with the gripper already closed (0), opens to clear the cube, closes
    to grasp, then opens to release. Thresholding on "low" alone therefore matches frame 0,
    which made this signal identically zero for every episode. Require the open first.
    """
    hi = float(grip.max())
    if hi <= 1e-6:
        return -1
    opened = np.flatnonzero(grip > 0.60 * hi)
    if not len(opened):
        return -1
    after = np.flatnonzero(grip[opened[0]:] < 0.35 * hi)
    return int(opened[0] + after[0]) if len(after) else -1


def grasp_fraction(grip: np.ndarray) -> float:
    i = grasp_index(grip)
    return float(i / len(grip)) if i >= 0 else 1.0


def residualise(values: np.ndarray, covariate: np.ndarray) -> np.ndarray:
    """Remove the linear effect of task difficulty from a signal.

    A cube that spawns further away takes longer to reach and is grasped later in the
    episode. That is the task being harder, not the operator being sloppier, and scoring it
    makes curation quietly select easy instances - which is the failure mode that would make
    a curated-vs-random comparison meaningless. Fit signal ~ reach and keep the residual, so
    the score asks "is this demonstration sloppy *for its difficulty*".
    """
    ok = np.isfinite(values) & np.isfinite(covariate)
    if ok.sum() < 10 or np.std(covariate[ok]) < 1e-9:
        return values
    slope, intercept = np.polyfit(covariate[ok], values[ok], 1)
    out = values.copy()
    out[ok] = values[ok] - (slope * covariate[ok] + intercept)
    return out


def scale_of(values: np.ndarray) -> tuple[float, float | None]:
    """Median and robust scale for a signal; scale None means the signal is constant.

    MAD is zero whenever most of the fleet shares one value - on scripted-expert data the
    gripper opens and closes identically every time, so grip_reversals has MAD 0. Treating
    that as "no information" throws away the most discriminative case there is: in a fleet
    where every demonstration does the same thing, one that does not is exactly the outlier
    worth flagging. Fall back to the largest observed deviation so the signal still ranks.
    """
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    scale = 1.4826 * mad
    if scale >= 1e-9:
        return med, scale
    dev = float(np.abs(values - med).max())
    return (med, dev) if dev > 1e-9 else (med, None)


def robust_z(values: np.ndarray) -> np.ndarray:
    med, scale = scale_of(values)
    if scale is None:                     # truly constant: carries no information
        return np.zeros_like(values)
    return (values - med) / scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="LeRobot dataset root")
    ap.add_argument("--out", default="", help="write scores JSON here")
    ap.add_argument("--top", type=int, default=10, help="how many best/worst to print")
    ap.add_argument("--no-difficulty-norm", action="store_true",
                    help="score raw signals instead of residuals after removing reach distance")
    args = ap.parse_args()

    eps = load_episodes(args.root)
    print(f"scoring {len(eps)} episodes from {args.root}")

    raw = {ep: raw_signals(t) for ep, t in eps.items()}
    order = sorted(raw)

    # Fleet median trajectory, on time-normalised episodes, then per-episode deviation.
    stack = np.stack([resample(eps[ep]) for ep in order])
    fleet = np.median(stack, axis=0)
    for i, ep in enumerate(order):
        raw[ep]["median_dev"] = float(np.linalg.norm(stack[i] - fleet, axis=1).mean())

    reach = np.array([raw[ep]["reach"] for ep in order])
    sig = {k: np.array([raw[ep][k] for ep in order]) for k in WEIGHTS}
    if not args.no_difficulty_norm:
        sig = {k: residualise(v, reach) for k, v in sig.items()}
    zs = {k: robust_z(sig[k]) for k in WEIGHTS}
    # grasp_phase is two-sided in principle, but only a LATE grasp indicates hesitation;
    # grabbing early is not a defect. Every signal is therefore penalised one-sided.
    penal = {k: np.maximum(zs[k], 0.0) for k in WEIGHTS}

    rows = []
    for i, ep in enumerate(order):
        parts = {k: float(WEIGHTS[k] * penal[k][i]) for k in WEIGHTS}
        score = float(sum(parts.values()))
        worst = max(parts, key=parts.get)
        rows.append(dict(episode=ep, score=score,
                         reason=(worst if parts[worst] > 0.5 else "clean"),
                         raw={k: float(v) for k, v in raw[ep].items()},
                         z={k: float(zs[k][i]) for k in WEIGHTS},
                         contrib=parts))

    rows.sort(key=lambda r: r["score"])
    s = np.array([r["score"] for r in rows])
    print(f"score: min {s.min():.2f}  median {np.median(s):.2f}  p90 {np.percentile(s,90):.2f}  max {s.max():.2f}")
    from collections import Counter
    print("primary reason across fleet:",
          ", ".join(f"{k}={v}" for k, v in Counter(r["reason"] for r in rows).most_common()))

    print(f"\nbest {args.top} (lowest score):")
    for r in rows[: args.top]:
        print(f"  ep {r['episode']:>4}  {r['score']:6.2f}  {r['reason']:<15} "
              f"len={r['raw']['duration']:.0f} path={r['raw']['path_ratio']:.2f} "
              f"grip_rev={r['raw']['grip_reversals']:.0f}")
    print(f"worst {args.top} (highest score):")
    for r in rows[-args.top:][::-1]:
        print(f"  ep {r['episode']:>4}  {r['score']:6.2f}  {r['reason']:<15} "
              f"len={r['raw']['duration']:.0f} path={r['raw']['path_ratio']:.2f} "
              f"grip_rev={r['raw']['grip_reversals']:.0f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(dict(root=args.root, n_episodes=len(rows), weights=WEIGHTS, episodes=rows),
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
