#!/usr/bin/env python3
"""Curated vs random demonstrations at a fixed budget, and why the winner wins.

Left: success rate per arm with Wilson intervals. Right: how much of the task's variation
each subset covers, relative to the random arm of the same size - the explanation for the
left panel.
"""
import argparse, glob, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/curation")
    ap.add_argument("--out", default="results/curation/curation.png")
    args = ap.parse_args()

    res = {}
    for f in glob.glob(os.path.join(args.dir, "*_n*.eval.json")):
        d = json.load(open(f))
        res[(d["n_demos"], d["arm"])] = d
    ns = sorted({n for n, _ in res})
    cov = json.load(open(os.path.join(args.dir, "coverage.json")))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    w = 0.35
    x = np.arange(len(ns))
    for i, (arm, colour) in enumerate((("curated", "#c0562f"), ("random", "#2f6d8a"))):
        vals, los, his = [], [], []
        for n in ns:
            d = res[(n, arm)]
            lo, hi = wilson(d["successes"], d["episodes"])
            vals.append(d["success_rate"] * 100); los.append((d["success_rate"] - lo) * 100)
            his.append((hi - d["success_rate"]) * 100)
        ax.bar(x + (i - 0.5) * w, vals, w, color=colour,
               label=f"{arm} (top-N by quality score)" if arm == "curated" else "random N (control)")
        ax.errorbar(x + (i - 0.5) * w, vals, yerr=[los, his], fmt="none", ecolor="#333", capsize=4, lw=1)
        for xi, v in zip(x + (i - 0.5) * w, vals):
            ax.text(xi, v + 2.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_ylabel("success rate over 50 held-out layouts (%)")
    ax.set_ylim(0, 85); ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("Selecting the best-scoring demonstrations\nmade the policy worse", fontsize=11)

    keys = ["reach_range", "mean_pairwise", "nn_dist", "reach_spread"]
    labels = ["reach\nrange", "mean pairwise\ndistance", "nearest-neighbour\ndistance", "reach\nspread"]
    xx = np.arange(len(keys))
    for i, n in enumerate(ns):
        c, r = cov[str(n)]["curated"], cov[str(n)]["random"]
        rel = [100 * (c[k] / r[k] - 1) for k in keys]
        ax2.bar(xx + (i - 0.5) * w, rel, w, label=f"N={n}",
                color=("#c0562f" if i == 0 else "#e0a080"))
    ax2.axhline(0, color="#333", lw=1)
    ax2.set_xticks(xx); ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("curated vs random, same N (%)")
    ax2.grid(axis="y", alpha=0.3); ax2.legend(fontsize=8.5)
    ax2.set_title("because the curated set covers less\nof the task's variation", fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)
    for n in ns:
        for arm in ("curated", "random"):
            d = res[(n, arm)]
            print(f"  N={n:<4} {arm:<8} {d['successes']:>2}/{d['episodes']} = {d['success_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
