# SO-101 pick-and-place with LeRobot: sim first, then hardware

Imitation learning on the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) low-cost arm using
[LeRobot](https://github.com/huggingface/lerobot) 0.6 and ACT. Phase 1 (this repo, today) runs the
whole pipeline in MuJoCo via [so101-nexus](https://github.com/johnsutor/so101-nexus) with a
scripted IK expert as the demonstrator. Phase 2 swaps the demonstrator for a physical leader arm;
dataset keys and units are already those of LeRobot's `so101_follower`, so the training and
evaluation code does not change.

| | |
|---|---|
| Task | pick up the 25 mm red cube, place it on the 5 cm-radius blue disc (success: cube centre within 4 cm of disc centre) |
| Robot | SO-101, MuJoCo Menagerie `robotstudio_so101` model, position-controlled STS3215 actuators |
| Observations | 6 joint positions (deg, gripper 0-100) + wrist and overhead RGB 320x240 |
| Actions | absolute joint targets, same units |
| Data | 600 expert episodes, 92,940 frames at 50 Hz, LeRobot v3 format (best run) |
| Policy | ACT (52M params), trained on an Apple M5 via MPS |

## Results (sim, 2026-09-22)

Success = cube released within 4 cm of the centre of the 5 cm-radius disc ("on the blue circle"), cube static, gripper open. Evaluated on cube/disc layouts never seen in training.

| Run | Data | Steps | Success (unseen layouts) |
|---|---|---|---|
| ACT, 640x480, 50 demos, random wrist cam | 7,790 frames | 20k (Mac) / 5k x bs32 (A100) | 1-2 / 20 |
| ACT, 320x240, 200 demos, random wrist cam | 31,161 frames | 10k-30k | 3-6 / 20 |
| ACT, 320x240, 300 demos, fixed wrist cam (v3) | 46,337 frames | 40k | 21 / 50 = 42 % (25/50 grasps) |
| v3, continued | same | 50k | 33 / 50 = 66 % |
| v3, continued | same | 60k | 34 / 50 = 68 % (38/50 grasps) |
| v3, continued | same | 70k / 80k | 27 / 50 = 54 % · 29 / 50 = 58 % (plateau) |
| ACT, 320x240, 600 demos, tighter overhead crop (v6), warm-started from v3-60k | 92,940 frames | 10k | 36 / 50 = 72 % |
| **v6, continued** | same | **20k** | **44 / 50 = 88 %** · 46 / 50 = 92 % on a second seed set · **90 / 100 pooled = 90 %** |
| v6, continued | same | 30k | 39 / 50 = 78 % (past the peak; run stopped here) |

What moved the needle, in order: **warm-starting a new run from the previous best checkpoint** (v6 reached 72 % after 10k steps, where v3 needed 60k to reach 68 %, and 88 % by 20k); doubling the demonstrations to 600; training long enough (v3 went 42 % to 68 % between 40k and 60k steps on data alone); halving image resolution so the same Mac could afford 3x more steps; fixing the simulated wrist camera (so101-nexus randomises its FOV/pitch/position every episode, which destroys the fine-alignment cue a rigidly mounted real camera gives); and tightening the overhead camera margin to 0.04 so the cube occupies more pixels. What did not: temporal ensembling, re-planning every 25 or 50 steps (both worse than executing the full 100-step chunk), a 35 mm cube (the scripted expert does not yaw-align the jaw, so its big-cube demos are inconsistent), and training either run past its peak (v3 54-58 % at 70k-80k, v6 78 % at 30k) - training loss keeps falling while success drops, so checkpoints have to be selected by rollout success, not loss.

Remaining failure mode is lateral precision at grasp: of the 6 failures in 50, 5 never close on the cube at all and 1 grasps without lifting. Every successful grasp becomes a successful placement, and failures are spread evenly over the 0.06-0.22 m spawn range, so this is vision precision rather than reach. Replaying recorded expert actions in the simulator succeeds 4/4, so the data path (units, timing, cameras) is verified end to end.

| v6 policy rollouts at 88 % (overhead + wrist) | |
|---|---|
| ![ok](assets/policy_v6_ok_1.gif) | ![ok](assets/policy_v6_ok_2.gif) |
| ![fail](assets/policy_rollout_fail.gif) (a failure from the earlier 42 % policy) | ![expert](assets/expert_overhead.gif) (scripted expert) |


## How many demonstrations are worth collecting? (sim, 2026-09-28)

The results above answer "how good can this get". They do not answer the question you actually
face before a data-collection session: **how much does the next demonstration buy you?** So this
sweep holds everything constant except the number of demonstrations.

Five ACT policies, 40k steps each, batch 8, seed 1000, identical architecture. All five are
evaluated on the *same* 50 held-out layouts (seed 9100), disjoint from every recording seed. The
subsets are nested - 25 ⊂ 50 ⊂ 100 ⊂ 200 ⊂ 300 - so each point adds data to the one below it
rather than redrawing, which removes subset composition as a source of variance. Dataset: 300
fresh episodes, 46,561 frames, regenerated with the settings the table above landed on.

![demonstrations vs success](results/sweep/curve.png)

| Demonstrations | Success (50 unseen layouts) | 95 % Wilson |
|---|---|---|
| 25 | 5 / 50 = 10 % | [4, 21] |
| 50 | 17 / 50 = 34 % | [22, 48] |
| 100 | 22 / 50 = 44 % | [31, 58] |
| 200 | **27 / 50 = 54 %** | [40, 67] |
| 300 | 25 / 50 = 50 % | [37, 63] |

The comparable published point is the v3 run above: 300 demonstrations, 40k steps, **42 %**. This
sweep's 300-demo point reaches 50 % on regenerated data, so the pipeline reproduces and slightly
exceeds it. Note that **68 % is the 60k-step number, not the 40k one** - every point here is
deliberately under-trained at a fixed compute budget, because the question is what data buys at
constant training cost, not what the task saturates at.

### The curve is a grasping curve

Splitting each failure by where it broke shows that one number moves and the other does not:

| Demos | Success | Never grasped | Grasped, no lift | Lifted, missed | Grasp rate | Place given grasp |
|---|---|---|---|---|---|---|
| 25 | 5 | 37 | 0 | 8 | 26 % | 38 % |
| 50 | 17 | 30 | 0 | 3 | 38 % | 84 % |
| 100 | 22 | 25 | 2 | 1 | 48 % | 88 % |
| 200 | 27 | 19 | 2 | 2 | 58 % | 86 % |
| 300 | 25 | 19 | 4 | 2 | 60 % | 80 % |

Grasp rate climbs monotonically with data, 26 → 38 → 48 → 58 → 60 %. **Place-given-grasp is flat
from 50 demonstrations onward: 84, 88, 86, 80 %.** Between 50 and 300 demonstrations - a 6x
increase - the policy learns to *find and close on* the cube, and learns nothing further about
carrying and releasing it. The one exception is 25 demonstrations, where place-given-grasp
collapses to 38 %: below some floor the policy has not learned the task at all, only a crude
reach.

Practical reading, for this task on this rig:

- **Below ~50 demonstrations, don't bother.** 25 is not a weak policy, it is a broken one.
- **Returns fall off sharply after ~100, and 200 vs 300 is inside the noise band** (2 episodes at
  n=50). Treat the top as a plateau, not a peak at 200 - this data cannot distinguish them.
- **More demonstrations will not fix placement, because placement is not what is failing.** The
  remaining 19-of-50 failures at N=300 are approach precision. Spending the next session on more
  of the same demonstrations buys less than fixing the approach (camera resolution at the grasp,
  wrist-camera geometry, or an approach-phase correction).

Each point cost ~52 min of RTX A4000 time; the whole sweep was **$0.78**.

| N=300 policy rollouts (overhead + wrist) | |
|---|---|
| ![ok](assets/sweep/rollout_02_ok.gif) | ![ok](assets/sweep/rollout_03_ok.gif) |
| ![fail](assets/sweep/rollout_00_fail.gif) (never closes on the cube) | ![fail](assets/sweep/rollout_01_fail.gif) (same failure, the dominant one) |

Reproduce:

```bash
python collect_sim.py --episodes 300 --seed 11 --spawn-min 0.04 --spawn-max 0.17 \
    --img-w 320 --img-h 240 --overhead-margin 0.04 --root data/sweep_300 \
    --repo-id $HF_USER/so101_sweep_300
bash sweep_all.sh                       # trains and evaluates all five points
python tools/plot_sweep.py              # results/sweep/curve.{png,csv}
python tools/failure_modes.py           # results/sweep/failure_modes.json
```

## Pipeline

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install 'lerobot[core_scripts,training,feetech]' so101-nexus
export MUJOCO_GL=glfw   # macOS offscreen rendering

python collect_sim.py --dry-run --episodes 20        # expert only, prints success rate
python collect_sim.py --episodes 600 --seed 11 --spawn-min 0.04 --spawn-max 0.17 \
    --img-w 320 --img-h 240 --overhead-margin 0.04 --root data/so101_sim_pick_place_v6

# from scratch:
DATASET_ROOT=data/so101_sim_pick_place_v6 RUN=act_sim_v6 SAVE_FREQ=10000 ./train.sh 60000 mps
# or warm-start from the previous best checkpoint - 88 % by 20k steps instead of 68 % by 60k
# (--policy.path replaces --policy.type, so call lerobot-train directly):
lerobot-train --policy.path=outputs/train/act_sim_v3/checkpoints/060000/pretrained_model \
    --policy.device=mps --policy.push_to_hub=false \
    --dataset.repo_id=aviadarn/so101_sim_pick_place_v6 \
    --dataset.root=data/so101_sim_pick_place_v6 \
    --output_dir=outputs/train/act_sim_v6 --job_name=act_sim_v6 \
    --steps=60000 --batch_size=8 --save_freq=10000 --num_workers=2 --wandb.enable=false

python eval_sim.py --policy outputs/train/act_sim_v6/checkpoints/020000/pretrained_model \
    --episodes 50 --seed 9100 --goal-thresh 0.04 --overhead-margin 0.04 --gif-dir assets/rollouts
```

## Scripted expert

`collect_sim.py` drives the arm through a waypoint FSM (hover, descend, close, lift, carry, lower,
open, retreat) in tool space and resolves each step with the env's damped-least-squares IK, so the
recorded action stream is smooth absolute joint targets, exactly what a leader arm produces. Three
details that took the expert from 59 % to 97-100 % success:

- **Jaw opening.** At 1.2 rad the moving jaw hit the cube top on descent and the arm stalled.
  1.5 rad clears it.
- **Grasp-offset compensation.** The cube rarely sits exactly at the TCP after the jaw closes.
  After the lift the expert measures the cube-to-TCP offset and shifts the place waypoints by it,
  so the cube lands on the disc instead of the tool.
- **IK orientation weight.** so101-nexus defaults to 0.01 (orientation nearly ignored). 0.05 keeps
  the tool vertical and converged fastest across previously failing seeds.

Workspace: spawn radius 0.04-0.17 m around (0.15, 0), i.e. up to ~0.35 m from the base. Beyond
~0.33 m a top-down grasp is not kinematically reachable for this arm.

## Sim-to-real contract

Everything the policy sees is expressed in real-robot units through
`so101_nexus.lerobot_dataset.sim_qpos_to_dataset_row` (body joints in degrees, gripper as
percent of travel), the dataset declares `robot_type: so101_follower`, and the two camera keys
match a wrist + overhead webcam layout. On hardware, recording becomes
`lerobot-record --robot.type=so101_follower --teleop.type=so101_leader ...` and deployment
`lerobot-rollout --policy.path=...`; `train.sh` is unchanged.
