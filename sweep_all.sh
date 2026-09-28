#!/usr/bin/env bash
# Demonstration sample-efficiency sweep: train one ACT policy per demonstration count and
# evaluate every one of them on the SAME held-out layouts.
#
# Everything except the number of demonstrations is held fixed - steps, batch size,
# architecture, training seed, and the evaluation seed - so the only variable is data.
#
# Reference point: the published v3 run (300 demos, 40k steps) scored 42%. Its 68% figure is
# the 60k-step result, so 42% is the like-for-like comparison at this budget.
set -u
cd "$(dirname "$0")"

# Everything is local; a Hub lookup here means the local dataset is missing.
export HF_HUB_OFFLINE=1
# Each spawned dataloader worker re-imports the stack and OpenBLAS opens one thread per core
# in every one of them. On a 64-core box with 8 workers that exhausted the container's
# pids.max, a worker died mid-import, and training hung with the GPU idle at 12 W.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# egl renders at ~516 fps on an A4000 against osmesa's ~7.9. It raises an EGLError in
# __del__ at teardown, which is cosmetic and happens after every frame is rendered.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PY=.venv/bin/python
TRAIN=.venv/bin/lerobot-train
DATA="${DATA:-data/sweep_300}"
REPO_ID="${REPO_ID:-aviadarn/so101_sweep_300}"
STEPS="${STEPS:-40000}"
SEED=1000
EVAL_EPISODES=50
EVAL_SEED=9100
TOTAL=300
SIZES="300 200 100 50 25"
mkdir -p sweep results

if [ ! -f "$DATA/meta/info.json" ]; then echo "ABORT: $DATA not found"; exit 1; fi
$PY -c "
import json; m=json.load(open('$DATA/meta/info.json'))
print('  dataset:', m['total_episodes'], 'episodes,', m['total_frames'], 'frames')"

for N in $SIZES; do
  RUN=sweep/n$N
  if [ ! -f "$RUN/trained" ]; then
    EPISODES=$($PY - "$N" "$TOTAL" <<'PY'
import json, random, sys
# One shuffled order, and each point takes a prefix of it, so the subsets nest:
# 25 subset 50 subset 100 subset 200 subset 300. Calling sample(range(N), n) per point looks
# equivalent but is not - CPython switches sample() between a pool and a set algorithm
# depending on k, so some points come from a different stream and do not nest.
n, total = int(sys.argv[1]), int(sys.argv[2])
order = random.Random(1000).sample(range(total), total)
print(json.dumps(sorted(order[:n]), separators=(",", ":")))
PY
)
    echo "$EPISODES" > sweep/n$N.episodes.json
    echo "=== train N=$N ($(date +%H:%M:%S)) ==="
    $TRAIN --dataset.repo_id="$REPO_ID" --dataset.root="$DATA" \
      --dataset.episodes="$EPISODES" --policy.type=act --policy.device=cuda \
      --policy.push_to_hub=false --output_dir="$RUN" --job_name="n$N" \
      --steps=$STEPS --batch_size=8 --save_freq=$STEPS --seed=$SEED \
      --log_freq=1000 --num_workers=4 --wandb.enable=false > sweep/n$N.train.log 2>&1 \
      && touch "$RUN/trained"
  fi

  CKPT=$(ls -d $RUN/checkpoints/*/pretrained_model 2>/dev/null | tail -1)
  if [ -n "$CKPT" ] && [ ! -f "sweep/n$N.eval.json" ]; then
    echo "=== eval N=$N ($(date +%H:%M:%S)) ==="
    $PY eval_sim.py --policy "$CKPT" --episodes $EVAL_EPISODES \
      --seed $EVAL_SEED --goal-thresh 0.04 --overhead-margin 0.04 --device cuda \
      > sweep/n$N.eval.log 2>&1
    tail -3 sweep/n$N.eval.log
    $PY - "$N" <<'PY'
import json, re, sys
n = sys.argv[1]
txt = open(f"sweep/n{n}.eval.log").read()
# Anchored on SUCCESS: an unanchored \d+/\d+ also matches a tqdm progress bar.
m = re.search(r"SUCCESS\s+(\d+)\s*/\s*(\d+)", txt)
out = {"n_demos": int(n), "raw_tail": txt[-400:]}
if m: out.update(successes=int(m.group(1)), episodes=int(m.group(2)),
                 success_rate=int(m.group(1))/int(m.group(2)))
json.dump(out, open(f"sweep/n{n}.eval.json", "w"), indent=2)
print("  ", out.get("successes"), "/", out.get("episodes"))
PY
  fi
done
echo "SWEEP_ALL_DONE"
