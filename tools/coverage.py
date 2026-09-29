#!/usr/bin/env python3
"""How much of the task's variation does a subset of demonstrations actually cover?

Written to explain a negative result: selecting the best-scoring demonstrations made the
policy worse, not better, at both N=50 and N=100. The suspicion is that the quality score
rewards typicality - median_dev penalises distance from the fleet median outright, and
smoothness, directness and duration all favour the modal demonstration - so "best" quietly
means "most alike", and a set of near-identical demonstrations shows the policy a narrower
slice of the state space than a random set of the same size.

This measures that directly, from the trajectories alone:

  reach_spread    std of the reach covariate: how much of the workspace is represented
  reach_range     p95-p5 of reach, robust to one outlier widening the set
  pose_volume     log-determinant of the covariance of time-normalised poses, a proxy for
                  the volume of joint space the set spans
  mean_pairwise   mean distance between time-normalised trajectories; low means the set is
                  a bundle of near-copies
  nn_dist         mean distance from each episode to its nearest neighbour in the set; the
                  measure least sensitive to a few extremes

If the curated arm is systematically lower on these than the random arm of the same size,
the metric bought smoothness and paid for it in coverage.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, "tools")
from score_demos import GRIPPER, load_episodes, raw_signals, resample


def coverage(trajs: list[np.ndarray]) -> dict[str, float]:
    flat = np.stack([resample(t).reshape(-1) for t in trajs])
    reach = np.array([raw_signals(t)["reach"] for t in trajs])
    reach = reach[np.isfinite(reach)]

    d = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=-1)
    iu = np.triu_indices(len(flat), k=1)
    np.fill_diagonal(d, np.inf)

    # log-det of covariance in a reduced space; full dim is far larger than the sample count
    k = min(12, len(flat) - 1)
    x = flat - flat.mean(0)
    sv = np.linalg.svd(x, compute_uv=False)[:k]
    pose_volume = float(np.sum(np.log(sv + 1e-9)))

    return dict(
        n=len(trajs),
        reach_spread=float(reach.std()),
        reach_range=float(np.percentile(reach, 95) - np.percentile(reach, 5)),
        pose_volume=pose_volume,
        mean_pairwise=float(np.linalg.norm(flat[iu[0]] - flat[iu[1]], axis=-1).mean()),
        nn_dist=float(d.min(axis=1).mean()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--select", nargs="+", required=True, help="selection JSONs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    eps = load_episodes(args.root)
    report = {}
    for path in args.select:
        sel = json.load(open(path))
        n = sel["n"]
        print(f"\n=== N={n} ({path}) ===")
        hdr = (f"{'arm':<10}{'reach_spread':>14}{'reach_range':>13}{'pose_volume':>13}"
               f"{'mean_pairwise':>15}{'nn_dist':>10}")
        print(hdr); print("-" * len(hdr))
        row = {}
        for arm in ("curated", "random"):
            c = coverage([eps[e] for e in sel[arm] if e in eps])
            row[arm] = c
            print(f"{arm:<10}{c['reach_spread']:>14.2f}{c['reach_range']:>13.1f}"
                  f"{c['pose_volume']:>13.1f}{c['mean_pairwise']:>15.1f}{c['nn_dist']:>10.1f}")
        rel = {k: 100 * (row["curated"][k] / row["random"][k] - 1)
               for k in ("reach_spread", "reach_range", "mean_pairwise", "nn_dist")}
        print(f"{'curated vs random':<10}" +
              "".join(f"{rel[k]:>+13.1f}%" for k in ("reach_spread", "reach_range")) +
              f"{'':>13}" + f"{rel['mean_pairwise']:>+14.1f}%{rel['nn_dist']:>+9.1f}%")
        report[n] = row

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
