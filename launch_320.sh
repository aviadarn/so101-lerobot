#!/usr/bin/env bash
cd "$(dirname "$0")"
until grep -q "dataset written" record_320.log 2>/dev/null; do sleep 30; done
sleep 5
DATASET_ROOT=data/so101_sim_pick_place_320 RUN=act_sim_320 BATCH=8 SAVE_FREQ=10000 ./train.sh 60000 mps > train_320.log 2>&1
echo "exit $?" >> train_320.log
