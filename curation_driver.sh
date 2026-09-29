#!/usr/bin/env bash
# Curated vs random demonstrations at a fixed budget, end to end and unattended.
#
# Four policies: {curated, random} x {N=50, N=100}. Everything except WHICH episodes are
# used is identical - policy, 40k steps, batch 8, seed 1000, and the same 50 held-out
# layouts at eval seed 9100, which are the settings Move 1's curve was measured at.
# N=50 and N=100 are chosen because Move 1 showed the curve is steep there; at N>=200 it
# has plateaued and curation could not show an effect either way.
set -u
cd /workspace
. /workspace/env.sh
PY=.venv/bin/python
TRAIN=.venv/bin/lerobot-train
DATA=data/curation_600
REPO=aviadarn/so101_curation_600
mkdir -p curation

if [ ! -f "$DATA/meta/info.json" ]; then
  echo "=== generating 600 episodes ($(date +%H:%M:%S)) ==="
  $PY collect_sim.py --episodes 600 --seed 11 --spawn-min 0.04 --spawn-max 0.17 \
    --img-w 320 --img-h 240 --overhead-margin 0.04 \
    --root "$DATA" --repo-id "$REPO" --overwrite --verbose > curation/gen.log 2>&1
  [ $? -ne 0 ] && { echo "ABORT: generation failed"; tail -20 curation/gen.log; exit 1; }
fi
$PY -c "
import json,sys; m=json.load(open('$DATA/meta/info.json'))
print('  dataset:', m['total_episodes'], 'episodes,', m['total_frames'], 'frames')
sys.exit(0 if m['total_episodes'] >= 600 else 1)" || { echo "ABORT: short dataset"; exit 1; }

echo "=== validating the quality metric on this dataset ($(date +%H:%M:%S)) ==="
$PY tools/test_score_demos.py --root "$DATA" > curation/metric_test.log 2>&1
if [ $? -ne 0 ]; then echo "ABORT: metric self-test failed"; tail -12 curation/metric_test.log; exit 1; fi
tail -6 curation/metric_test.log

echo "=== scoring ($(date +%H:%M:%S)) ==="
$PY tools/score_demos.py --root "$DATA" --out curation/scores.json --top 3 2>&1 | head -6
for N in 50 100; do
  $PY tools/select_demos.py --scores curation/scores.json --n $N --out curation/select_n$N.json 2>&1 | head -7
done

echo "=== smoke train ($(date +%H:%M:%S)) ==="
rm -rf /workspace/smoke
$TRAIN --dataset.repo_id="$REPO" --dataset.root="$DATA" --dataset.episodes='[0,1,2,3,4,5,6,7,8,9]' \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false --output_dir=/workspace/smoke \
  --job_name=smoke --steps=200 --batch_size=8 --save_freq=200 --seed=1000 --log_freq=100 \
  --num_workers=4 --wandb.enable=false > curation/smoke.log 2>&1
[ $? -ne 0 ] && { echo "ABORT: smoke train failed"; tail -20 curation/smoke.log; exit 1; }
echo "  smoke train OK"

for N in 50 100; do
  for ARM in curated random; do
    TAG="${ARM}_n${N}"; RUN=curation/$TAG
    EP=$($PY -c "import json;print(json.dumps(json.load(open('curation/select_n$N.json'))['$ARM'],separators=(',',':')))")
    if [ ! -f "$RUN/trained" ]; then
      echo "=== train $TAG ($(date +%H:%M:%S)) ==="
      $TRAIN --dataset.repo_id="$REPO" --dataset.root="$DATA" --dataset.episodes="$EP" \
        --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
        --output_dir="$RUN" --job_name="$TAG" --steps=40000 --batch_size=8 \
        --save_freq=40000 --seed=1000 --log_freq=1000 --num_workers=4 \
        --wandb.enable=false > curation/$TAG.train.log 2>&1 && touch "$RUN/trained"
    fi
    CKPT=$(ls -d $RUN/checkpoints/*/pretrained_model 2>/dev/null | tail -1)
    if [ -n "$CKPT" ] && [ ! -f "curation/$TAG.eval.json" ]; then
      echo "=== eval $TAG ($(date +%H:%M:%S)) ==="
      $PY eval_sim.py --policy "$CKPT" --episodes 50 --seed 9100 \
        --goal-thresh 0.04 --overhead-margin 0.04 --device cuda > curation/$TAG.eval.log 2>&1
      $PY - "$TAG" "$ARM" "$N" <<'PY'
import json, re, sys
tag, arm, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
txt = open(f"curation/{tag}.eval.log").read()
m = re.search(r"SUCCESS\s+(\d+)\s*/\s*(\d+)", txt)      # anchored: a tqdm bar also matches N/N
out = {"arm": arm, "n_demos": n, "raw_tail": txt[-300:]}
if m: out.update(successes=int(m.group(1)), episodes=int(m.group(2)),
                 success_rate=int(m.group(1))/int(m.group(2)))
json.dump(out, open(f"curation/{tag}.eval.json", "w"), indent=2)
print(f"  {arm} N={n}: {out.get('successes')}/{out.get('episodes')}")
PY
    fi
  done
done
echo "CURATION_DONE"
