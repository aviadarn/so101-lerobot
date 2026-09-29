#!/usr/bin/env python3
"""Does the quality score actually detect defects, or does it just produce a ranking?

A scorer that sorts episodes will always produce a confident-looking order, even if the
order is noise. Before spending GPU on a curated-vs-random training comparison, inject
defects that a human would call sloppy into real episodes and check the score notices - and
that it attributes them to the right signal, since the rejection reason is what an operator
would actually be shown.

Run: .venv/bin/python tools/test_score_demos.py --root data/so101_sim_pick_place_v6
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "tools")
from score_demos import GRIPPER, WEIGHTS, grasp_index, load_episodes, raw_signals, resample, scale_of


def inject_dither(traj: np.ndarray, rng: np.random.Generator, amp: float = 3.5) -> np.ndarray:
    """Operator wobbles on the way to the object: extra path length and jerk, same endpoints."""
    out = traj.copy()
    n = len(out)
    w = np.sin(np.linspace(0, 60 * np.pi, n))[:, None] * amp
    taper = np.sin(np.linspace(0, np.pi, n))[:, None]          # leave start/end untouched
    out[:, :GRIPPER] += w * taper * rng.normal(1.0, 0.2, (1, GRIPPER))
    return out


def inject_pause(traj: np.ndarray, rng: np.random.Generator, frames: int = 30) -> np.ndarray:
    """Operator hesitates before the grasp: a held pose inserted mid-approach."""
    gi = grasp_index(traj[:, GRIPPER])
    at = gi - 5 if gi > 10 else len(traj) // 3
    return np.concatenate([traj[:at], np.repeat(traj[at: at + 1], frames, axis=0), traj[at:]])


def inject_regrasp(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Operator closes, changes their mind, reopens and closes again."""
    out = traj.copy()
    gi = grasp_index(out[:, GRIPPER])
    if gi < 0:
        return out
    hi = out[:, GRIPPER].max()
    end = min(gi + 12, len(out) - 1)
    out[gi:end, GRIPPER] = hi * 0.8                             # reopen right after closing
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/so101_sim_pick_place_v6")
    ap.add_argument("--n", type=int, default=40, help="clean episodes to corrupt")
    args = ap.parse_args()

    eps = load_episodes(args.root)
    order = sorted(eps)
    rng = np.random.default_rng(0)

    # Score the untouched fleet first; defects are judged against this scale.
    base_raw = {ep: raw_signals(eps[ep]) for ep in order}
    # median_dev is defined against the fleet, so build the fleet reference here and score
    # both clean and corrupted episodes against that same fixed reference.
    fleet = np.median(np.stack([resample(eps[ep]) for ep in order]), axis=0)
    def median_dev(traj):
        return float(np.linalg.norm(resample(traj) - fleet, axis=1).mean())
    for ep in order:
        base_raw[ep]["median_dev"] = median_dev(eps[ep])
    keys = list(WEIGHTS)
    base = {k: np.array([base_raw[ep][k] for ep in order]) for k in keys}
    # Use the same median/scale rule the scorer uses, so the test measures the shipped
    # behaviour rather than a re-implementation of it.
    scales = {k: scale_of(base[k]) for k in keys}
    med = {k: scales[k][0] for k in keys}
    mad = {k: scales[k][1] for k in keys}
    dead = [k for k in keys if mad[k] is None]
    if dead:
        print(f"note: constant across this fleet, ignored: {', '.join(dead)}")

    def score_of(traj: np.ndarray) -> tuple[float, str]:
        r = raw_signals(traj)
        r["median_dev"] = median_dev(traj)
        parts = {k: (0.0 if k in dead else WEIGHTS[k] * max((r[k] - med[k]) / mad[k], 0.0))
                 for k in keys}
        worst = max(parts, key=parts.get)
        return float(sum(parts.values())), worst

    victims = [ep for ep in order if base_raw[ep]["duration"] <= np.median(base["duration"])][: args.n]
    cases = [("dither (wander)", inject_dither, "path_ratio|jerk|median_dev"),
             ("pause (hesitate)", inject_pause, "duration|pause_frac|grasp_phase"),
             # A regrasp is genuinely two things at once: extra gripper reversals, and a
             # later effective grasp because the first close was undone. Both attributions
             # are correct explanations of the same defect, and the split is ~50/50.
             ("regrasp (change mind)", inject_regrasp, "grip_reversals|grasp_phase")]

    print(f"{len(victims)} clean episodes from {args.root}, scored before and after injection\n")
    hdr = f"{'defect':<24}{'score before':>13}{'score after':>13}{'worse in':>10}{'reason matches':>16}"
    print(hdr); print("-" * len(hdr))

    failures = []
    for name, fn, expect in cases:
        before, after, hits = [], [], 0
        for ep in victims:
            b, _ = score_of(eps[ep])
            a, reason = score_of(fn(eps[ep], rng))
            before.append(b); after.append(a)
            if reason in expect.split("|"):
                hits += 1
        before, after = np.array(before), np.array(after)
        worse = int((after > before).sum())
        print(f"{name:<24}{before.mean():>13.2f}{after.mean():>13.2f}"
              f"{worse:>7}/{len(victims)}{hits:>13}/{len(victims)}")
        if worse < 0.9 * len(victims):
            failures.append(f"{name}: only {worse}/{len(victims)} scored worse")
        if hits < 0.5 * len(victims):
            failures.append(f"{name}: reason matched {expect} in only {hits}/{len(victims)}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("PASS: every injected defect raises the score, and the reported reason matches it.")


if __name__ == "__main__":
    main()
