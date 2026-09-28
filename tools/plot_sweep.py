#!/usr/bin/env python3
"""Plot the demonstration sample-efficiency curve from the sweep's eval JSONs.

Everything except the demonstration count is held fixed across the points: policy
(ACT), steps, batch size, training seed, and the evaluation itself (same 50 held-out
layouts from the same seed). So the only thing the curve varies is how many
demonstrations the policy was trained on.
"""
import argparse, csv, glob, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wilson(k, n, z=1.96):
    """Wilson score interval. At 50 rollouts the normal approximation misbehaves near
    0 and 1, which is exactly where the small-demonstration points sit."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sweep")
    ap.add_argument("--out", default="results/sweep/curve.png")
    ap.add_argument("--csv", default="results/sweep/curve.csv")
    ap.add_argument("--anchor", type=float, default=0.42,
                    help="original 300-demo run at the SAME step budget. Its 68%% figure is "
                         "the 60k-step result; these runs stop at 40k, so 42%% is the "
                         "like-for-like comparison.")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.sweep, "*.eval.json"))):
        d = json.load(open(f))
        if "success_rate" not in d:
            print(f"  skipping {os.path.basename(f)}: no success_rate (eval did not parse)")
            continue
        lo, hi = wilson(d["successes"], d["episodes"])
        rows.append(dict(n=d["n_demos"], k=d["successes"], eps=d["episodes"],
                         rate=d["success_rate"], lo=lo, hi=hi))
    if not rows:
        raise SystemExit(f"no usable eval JSONs in {args.sweep}")
    rows.sort(key=lambda r: r["n"])

    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["n", "k", "eps", "rate", "lo", "hi"])
        w.writeheader()
        w.writerows(rows)

    ns = [r["n"] for r in rows]
    rates = [r["rate"] * 100 for r in rows]
    err_lo = [(r["rate"] - r["lo"]) * 100 for r in rows]
    err_hi = [(r["hi"] - r["rate"]) * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(ns, rates, yerr=[err_lo, err_hi], marker="o", capsize=4,
                color="#1f77b4", label="ACT, 40k steps, seed 1000")
    ax.axhline(args.anchor * 100, ls="--", lw=1, color="#888",
               label=f"original 300-demo run at 40k steps ({args.anchor*100:.0f}%)")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("demonstrations")
    ax.set_ylabel(f"success rate over {rows[0]['eps']} held-out layouts (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("SO-101 pick-and-place: demonstrations vs success at a fixed 40k-step budget")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)

    print(f"wrote {args.out} and {args.csv}")
    for r in rows:
        print(f"  N={r['n']:4d}  {r['k']:2d}/{r['eps']}  {r['rate']*100:5.1f}%"
              f"  [{r['lo']*100:.0f}, {r['hi']*100:.0f}]")


if __name__ == "__main__":
    main()
