#!/usr/bin/env python3
"""Pick a curated subset and a matched random control of the same size.

This is the experiment half of the curation question: at a fixed budget of N
demonstrations, does taking the best-scoring N beat taking N at random? The control is the
whole point - "curated N beats random 2N" is a different and much weaker claim, and
"curated N beats a control that happens to be harder" is not a claim at all.

The trap this guards against: episode quality signals correlate with how far the object
spawned, because a farther object takes longer to reach and is grasped later. Sorting on
score alone therefore quietly selects easy instances, and a downstream win could just be
"we trained on the easy half". Regressing difficulty out of each signal over-corrected in
practice (it flipped the bias rather than removing it), so the selection is stratified
instead: episodes are binned into difficulty quantiles from the reach covariate, and the
curated set takes the best-scoring episodes *within each bin*, in proportion to the bin's
share of the fleet. The curated set then carries the fleet's difficulty distribution by
construction rather than by hope, and the control is drawn under the identical strata.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def stratified_curated(score: np.ndarray, reach: np.ndarray, n: int, bins: int,
                       rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(len(score))
    edges = np.quantile(reach, np.linspace(0, 1, bins + 1)[1:-1])
    strata = np.searchsorted(edges, reach)
    take = allocate(strata, bins, n)
    out = []
    for b in range(bins):
        pool = idx[strata == b]
        pool = pool[np.argsort(score[pool], kind="stable")]     # best first
        out.append(pool[: take[b]])
    return np.sort(np.concatenate(out))


def stratified_random(score: np.ndarray, reach: np.ndarray, n: int, bins: int,
                      rng: np.random.Generator) -> np.ndarray:
    idx = np.arange(len(score))
    edges = np.quantile(reach, np.linspace(0, 1, bins + 1)[1:-1])
    strata = np.searchsorted(edges, reach)
    take = allocate(strata, bins, n)
    out = []
    for b in range(bins):
        pool = idx[strata == b]
        out.append(rng.choice(pool, size=take[b], replace=False))
    return np.sort(np.concatenate(out))


def allocate(strata: np.ndarray, bins: int, n: int) -> list[int]:
    """How many episodes to draw from each difficulty bin, proportional to bin size."""
    sizes = np.array([(strata == b).sum() for b in range(bins)])
    exact = sizes / sizes.sum() * n
    take = np.floor(exact).astype(int)
    # hand out the remainder to the bins with the largest fractional part
    for b in np.argsort(-(exact - take))[: n - take.sum()]:
        take[b] += 1
    return [int(min(t, s)) for t, s in zip(take, sizes)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="scores JSON from score_demos.py")
    ap.add_argument("--n", type=int, required=True, help="demonstrations per arm")
    ap.add_argument("--bins", type=int, default=5, help="difficulty strata")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="", help="write selection JSON here")
    args = ap.parse_args()

    rows = json.load(open(args.scores))["episodes"]
    rows.sort(key=lambda r: r["episode"])
    ep = np.array([r["episode"] for r in rows])
    score = np.array([r["score"] for r in rows])
    reach = np.array([r["raw"]["reach"] for r in rows])
    if not np.isfinite(reach).all():
        med = np.nanmedian(reach)
        reach = np.where(np.isfinite(reach), reach, med)

    if args.n > len(ep):
        raise SystemExit(f"asked for {args.n} of {len(ep)} episodes")

    rng = np.random.default_rng(args.seed)
    cur = stratified_curated(score, reach, args.n, args.bins, rng)
    ctl = stratified_random(score, reach, args.n, args.bins, rng)

    sel = {"curated": sorted(int(ep[i]) for i in cur),
           "random": sorted(int(ep[i]) for i in ctl)}

    print(f"{len(ep)} episodes -> {args.n} per arm, {args.bins} difficulty strata, seed {args.seed}")
    hdr = f"{'arm':<10}{'mean score':>12}{'mean reach':>12}{'reach vs fleet':>16}{'overlap':>9}"
    print(hdr); print("-" * len(hdr))
    both = len(set(sel["curated"]) & set(sel["random"]))
    for name, take in (("curated", cur), ("random", ctl)):
        print(f"{name:<10}{score[take].mean():>12.2f}{reach[take].mean():>12.1f}"
              f"{100*(reach[take].mean()/reach.mean()-1):>+15.1f}%"
              f"{both if name=='random' else both:>9}")
    print(f"{'fleet':<10}{score.mean():>12.2f}{reach.mean():>12.1f}{0.0:>+15.1f}%")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(dict(n=args.n, bins=args.bins, seed=args.seed, scores=args.scores,
                       mean_score={k: float(score[v].mean()) for k, v in
                                   (("curated", cur), ("random", ctl))},
                       mean_reach={k: float(reach[v].mean()) for k, v in
                                   (("curated", cur), ("random", ctl))},
                       fleet_mean_reach=float(reach.mean()),
                       overlap=both, **sel),
                  open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
