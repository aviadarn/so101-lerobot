"""Evaluate a trained LeRobot policy on the SO-101 pick-and-place task in MuJoCo.

Loads a LeRobot checkpoint (ACT by default), runs N episodes on unseen seeds with the
same observation keys/units used for recording (``collect_sim.py``), and reports the
success rate. Optionally writes side-by-side overhead+wrist GIFs of the first rollouts.

Usage:
    python eval_sim.py --policy outputs/train/act_sim/checkpoints/last/pretrained_model \
        --episodes 20 --gif-dir assets/rollouts
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import collect_sim as cs
from so101_nexus.lerobot_dataset import dataset_row_to_sim_qpos, sim_qpos_to_dataset_row

ROBOT_TYPE = "so101_follower"


def load_policy(path: str, device: str, n_action_steps: int | None = None, temporal_ensemble: float | None = None):
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.factory import get_policy_class
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.pretrained_path = path
    # Inference-only knobs: how much of each predicted chunk to execute open loop
    # before re-observing, and ACT temporal ensembling (requires n_action_steps=1).
    if temporal_ensemble is not None:
        cfg.temporal_ensemble_coeff = temporal_ensemble
        cfg.n_action_steps = 1
    elif n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    print(f"inference: chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps} "
          f"temporal_ensemble_coeff={getattr(cfg, 'temporal_ensemble_coeff', None)}")
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, pre, post


def obs_to_frame(obs: dict, grip_lim) -> dict:
    return {
        "observation.state": sim_qpos_to_dataset_row(
            obs["state"].astype(np.float64), gripper_limits_rad=grip_lim
        ).astype(np.float32),
        "observation.images.wrist": obs["wrist_camera"],
        "observation.images.overhead": obs["overhead_camera"],
    }


def run(args) -> int:
    from lerobot.policies.utils import prepare_observation_for_inference

    device = args.device
    policy, pre, post = load_policy(args.policy, device, args.n_action_steps, args.temporal_ensemble)
    # Match the env camera resolution to what the policy was trained on.
    shp = policy.config.input_features["observation.images.overhead"].shape  # (C,H,W)
    img_h, img_w = int(shp[1]), int(shp[2])
    print(f"env cameras: {img_w}x{img_h}")
    env = cs.make_env(args.seed, args.spawn_min, args.spawn_max, img_w, img_h,
                      cube_mm=args.cube_mm, goal_thresh=args.goal_thresh, overhead_margin=args.overhead_margin)
    u = env.unwrapped
    grip_lim = (float(env.action_space.low[-1]), float(env.action_space.high[-1]))
    low, high = env.action_space.low, env.action_space.high

    gif_dir = Path(args.gif_dir) if args.gif_dir else None
    if gif_dir:
        gif_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    t0 = time.time()
    for ep in range(args.episodes):
        seed = args.seed * 100_000 + ep
        obs, info = env.reset(seed=seed)
        policy.reset()
        pre.reset()
        post.reset()
        frames = []
        ok = False
        first_success = -1
        first_grasp = -1
        max_lift = 0.0
        init_dist = float(info["obj_to_target_dist"])
        for t in range(args.max_steps):
            if gif_dir and ep < args.gif_episodes and t % 3 == 0:
                frames.append(np.concatenate([obs["overhead_camera"], obs["wrist_camera"]], axis=1)[::2, ::2])
            frame = obs_to_frame(obs, grip_lim)
            with torch.inference_mode():
                batch = prepare_observation_for_inference(frame, torch.device(device), cs.TASK, ROBOT_TYPE)
                batch = pre(batch)
                action = policy.select_action(batch)
                action = post(action)
            row = action.squeeze(0).cpu().numpy().astype(np.float64)
            qpos = dataset_row_to_sim_qpos(row, gripper_limits_rad=grip_lim)
            qpos = np.clip(qpos, low, high).astype(np.float32)
            obs, _r, _term, _trunc, info = env.step(qpos)
            if info.get("is_grasped", 0.0) and first_grasp < 0:
                first_grasp = t
            max_lift = max(max_lift, float(info.get("lift_height", 0.0)))
            if info.get("success", False):
                ok = True
                if first_success < 0:
                    first_success = t
                if args.stop_on_success:
                    break
        successes += int(ok)
        print(
            f"ep {ep:2d} seed {seed} {'OK  ' if ok else 'FAIL'} first_success@{first_success} "
            f"first_grasp@{first_grasp} max_lift={max_lift:.3f} "
            f"init_dist={init_dist:.3f} final_dist={info['obj_to_target_dist']:.3f}"
        )
        if gif_dir and frames:
            from PIL import Image

            ims = [Image.fromarray(f) for f in frames]
            out = gif_dir / f"rollout_{ep:02d}_{'ok' if ok else 'fail'}.gif"
            ims[0].save(out, save_all=True, append_images=ims[1:], duration=60, loop=0)
    rate = successes / args.episodes
    print(f"SUCCESS {successes}/{args.episodes} = {rate:.0%}  ({time.time() - t0:.0f}s, device={device})")
    env.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="pretrained_model dir or Hub repo id")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=9000, help="eval seeds are disjoint from recording seeds")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--spawn-min", type=float, default=0.04)
    ap.add_argument("--spawn-max", type=float, default=0.17)
    ap.add_argument("--gif-dir", default="")
    ap.add_argument("--gif-episodes", type=int, default=3)
    ap.add_argument("--stop-on-success", action="store_true")
    ap.add_argument("--cube-mm", type=float, default=25.0)
    ap.add_argument("--goal-thresh", type=float, default=0.025)
    ap.add_argument("--overhead-margin", type=float, default=0.10)
    ap.add_argument("--n-action-steps", type=int, default=None, help="execute this many steps per chunk (default: checkpoint's)")
    ap.add_argument("--temporal-ensemble", type=float, default=None, help="ACT temporal ensemble coeff (e.g. 0.01); forces n_action_steps=1")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
