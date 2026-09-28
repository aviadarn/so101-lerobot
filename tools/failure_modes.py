#!/usr/bin/env python3
"""Break each sweep point's failures into modes, from the per-episode eval diagnostics.

A single success rate says how often the policy failed; it does not say whether the
policy could not see the cube, could not hold it, or could not place it. Those need
different fixes, so the sweep reports them separately.
"""
import argparse, glob, json, os, re

LINE = re.compile(
    # eval_sim.py pads the verdict to four chars ('OK  '), so \s+ not a single space
    r"ep\s+(\d+) seed (\d+) (OK|FAIL)\s+first_success@(-?\d+) first_grasp@(-?\d+)\s+"
    r"max_lift=([\d.]+)\s+init_dist=([\d.]+)\s+final_dist=([\d.]+)"
)
LIFT_M = 0.03  # a grasp that never raises the cube this far never actually held it


def classify(rec):
    if rec["ok"]:
        return "success"
    if rec["first_grasp"] < 0:
        return "never_grasped"          # closed on nothing: lateral precision at approach
    if rec["max_lift"] < LIFT_M:
        return "grasped_no_lift"        # touched and closed, but lost it immediately
    return "lifted_missed_target"       # carried it and put it down in the wrong place


def parse(path):
    out = []
    for m in LINE.finditer(open(path, errors="replace").read().replace("\r", "\n")):
        ep, seed, ok, fs, fg, lift, d0, d1 = m.groups()
        out.append(dict(ep=int(ep), seed=int(seed), ok=(ok == "OK"),
                        first_success=int(fs), first_grasp=int(fg),
                        max_lift=float(lift), init_dist=float(d0), final_dist=float(d1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="results/sweep")
    ap.add_argument("--out", default="results/sweep/failure_modes.json")
    args = ap.parse_args()

    modes = ["success", "never_grasped", "grasped_no_lift", "lifted_missed_target"]
    table = {}
    for path in sorted(glob.glob(os.path.join(args.sweep, "*.eval.log"))):
        n = int(re.search(r"n(\d+)\.eval\.log", os.path.basename(path)).group(1))
        recs = parse(path)
        if not recs:
            print(f"  {os.path.basename(path)}: no parsable episodes"); continue
        counts = {m: 0 for m in modes}
        for r in recs:
            counts[classify(r)] += 1
        grasped = [r for r in recs if r["first_grasp"] >= 0]
        table[n] = dict(
            episodes=len(recs), **counts,
            grasp_rate=len(grasped) / len(recs),
            # once it has the cube, does it finish? separates perception from control
            place_given_grasp=(sum(r["ok"] for r in grasped) / len(grasped)) if grasped else None,
        )

    if not table:
        raise SystemExit(f"no eval logs in {args.sweep}")
    json.dump(table, open(args.out, "w"), indent=2, sort_keys=True)

    hdr = f"{'N':>5} {'eps':>4} {'ok':>4} {'no grasp':>9} {'no lift':>8} {'missed':>7} {'grasp%':>7} {'place|grasp':>12}"
    print(hdr); print("-" * len(hdr))
    for n in sorted(table, reverse=True):
        t = table[n]
        pg = "n/a" if t["place_given_grasp"] is None else f"{t['place_given_grasp']*100:.0f}%"
        print(f"{n:>5} {t['episodes']:>4} {t['success']:>4} {t['never_grasped']:>9} "
              f"{t['grasped_no_lift']:>8} {t['lifted_missed_target']:>7} "
              f"{t['grasp_rate']*100:>6.0f}% {pg:>12}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
