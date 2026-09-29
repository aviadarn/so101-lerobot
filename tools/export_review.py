#!/usr/bin/env python3
"""Export a reviewable slice of a scored LeRobot dataset for the browser review UI.

The scorer ranks demonstrations and says why it rejected each one. That is only useful if a
person can check it, so this writes what a reviewer needs: the ranked list with each
episode's score, its rejection reason, the per-signal breakdown behind that reason, the
joint trajectory to plot, and a short video clip of the episode itself.

Two things are deliberate:

  Codec. LeRobot writes AV1, which Safari and older Chrome will not reliably play, so clips
  are transcoded to H.264 yuv420p with the moov atom moved to the front. A review tool that
  silently shows nothing on the reviewer's machine is worse than no review tool.

  Size. Episodes live inside concatenated per-chunk mp4s, and the full dataset's video is
  half a gigabyte. Only a bounded, deliberately chosen slice is exported - the worst, the
  best, and an even spread through the middle - so the whole thing stays small enough to
  ship in the repo and open as a single page.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from score_demos import GRIPPER, grasp_index, load_episodes

TRAJ_POINTS = 96          # enough to see the shape, small enough to ship 40 of them
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def pick(scores: list[dict], n: int) -> list[int]:
    """Worst, best, and an even spread between - a reviewer needs to see both ends."""
    ranked = sorted(scores, key=lambda r: r["score"])
    if n >= len(ranked):
        return [r["episode"] for r in ranked]
    k = max(1, n // 4)
    chosen = [r["episode"] for r in ranked[:k]] + [r["episode"] for r in ranked[-k:]]
    mid = ranked[k:-k]
    step = max(1, len(mid) // max(1, n - len(chosen)))
    chosen += [mid[i]["episode"] for i in range(0, len(mid), step)][: n - len(chosen)]
    return sorted(set(chosen))


def video_windows(root: str) -> dict[int, dict[str, tuple[str, float, float]]]:
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "**", "*.parquet"),
                             recursive=True))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cams = [c.split("/")[1] for c in df.columns
            if c.startswith("videos/") and c.endswith("/from_timestamp")]
    out: dict[int, dict] = {}
    for _, r in df.iterrows():
        ep = int(r["episode_index"]); out[ep] = {}
        for cam in cams:
            ci, fi = int(r[f"videos/{cam}/chunk_index"]), int(r[f"videos/{cam}/file_index"])
            path = os.path.join(root, "videos", cam, f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
            out[ep][cam] = (path, float(r[f"videos/{cam}/from_timestamp"]),
                            float(r[f"videos/{cam}/to_timestamp"]))
    return out


def extract(src: str, start: float, end: float, dst: str, height: int) -> bool:
    """Cut one episode out of the concatenated file and re-encode for browsers."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}",
           "-t", f"{max(end - start, 0.1):.3f}", "-i", src,
           "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-crf", "30",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", dst]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        print(f"    ffmpeg failed for {os.path.basename(dst)}: "
              f"{r.stderr.decode(errors='replace')[:160]}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True, help="public/ dir of the review UI")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    if not args.no_video and not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not on PATH; pass --no-video to export data only")

    scored = json.load(open(args.scores))
    rows = {r["episode"]: r for r in scored["episodes"]}
    chosen = pick(scored["episodes"], args.episodes)
    print(f"exporting {len(chosen)} of {len(rows)} episodes")

    eps = load_episodes(args.root)
    wins = video_windows(args.root) if not args.no_video else {}
    clips_dir = os.path.join(args.out, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    out = []
    for i, ep in enumerate(chosen):
        traj = eps[ep]
        idx = np.linspace(0, len(traj) - 1, min(TRAJ_POINTS, len(traj))).astype(int)
        gi = grasp_index(traj[:, GRIPPER])
        rec = rows[ep]
        clips = {}
        for cam, (src, a, b) in wins.get(ep, {}).items():
            short = cam.replace("observation.images.", "")
            name = f"ep{ep:04d}_{short}.mp4"
            if extract(src, a, b, os.path.join(clips_dir, name), args.height):
                clips[short] = f"clips/{name}"
        out.append(dict(
            episode=ep, score=round(rec["score"], 3), reason=rec["reason"],
            frames=int(len(traj)),
            grasp_frame=int(gi), grasp_phase=round(gi / len(traj), 3) if gi >= 0 else None,
            raw={k: round(v, 4) for k, v in rec["raw"].items()},
            z={k: round(v, 3) for k, v in rec["z"].items()},
            contrib={k: round(v, 3) for k, v in rec["contrib"].items()},
            # transposed per joint so the chart can draw one polyline per channel
            traj=[[round(float(traj[t, j]), 2) for t in idx] for j in range(traj.shape[1])],
            clips=clips,
        ))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(chosen)}")

    manifest = dict(
        dataset=os.path.basename(args.root.rstrip("/")),
        n_total=len(rows), n_exported=len(out),
        joints=JOINTS, weights=scored.get("weights", {}),
        signals=sorted(scored["episodes"][0]["contrib"]),
        score_median=round(float(np.median([r["score"] for r in scored["episodes"]])), 3),
        episodes=out,
    )
    path = os.path.join(args.out, "review.json")
    json.dump(manifest, open(path, "w"), separators=(",", ":"))
    mb = os.path.getsize(path) / 1e6
    clip_mb = sum(os.path.getsize(os.path.join(clips_dir, f))
                  for f in os.listdir(clips_dir)) / 1e6 if os.path.isdir(clips_dir) else 0
    print(f"wrote {path} ({mb:.2f} MB) and {len(os.listdir(clips_dir))} clips ({clip_mb:.1f} MB)")


if __name__ == "__main__":
    main()
