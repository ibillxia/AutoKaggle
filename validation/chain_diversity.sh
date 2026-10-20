#!/usr/bin/env bash
# Diversity members: vary lr / epochs / data instead of only the seed.
set -u
cd /e/Bill/AI/AutoKaggle || exit 1
PY=C:/Users/deepg/.workbuddy/binaries/python/envs/default/Scripts/python.exe
DATA=data/comp/contradictory-my-dear-watson

run () {  # name data epochs seed lr
  local name=$1 data=$2 eps=$3 seed=$4 lr=$5
  if [ -f "validation/runs/${name}-dev/rows.json" ]; then echo "skip $name"; return; fi
  echo "=== $name  $(date -u +%H:%M:%S)"
  PYTHONUNBUFFERED=1 "$PY" -u validation/train_hf_nli2.py \
    --model models/xlm-roberta-large --data "$DATA/$data" --eval "$DATA/kev_dev.jsonl" \
    --out "runs/$name" --out-rows "validation/runs/${name}-dev/rows.json" \
    --epochs "$eps" --lr "$lr" --batch 24 --max-len 128 --optim adamw8bit --precision bf16 \
    --freeze-below 12 --save 1 --log-every 1000 --warmup-frac 0.03 --eval-batch 32 --seed "$seed" \
    > "validation/${name}.log" 2>&1
  grep -E "EVAL" "validation/${name}.log" | tail -1
}

run xlmr-fb12-big-s51-lr2e5 kev_train_big.jsonl   2 51 2e-5
run xlmr-fb12-big3-s61      kev_train_big.jsonl   3 61 3e-5
run xlmr-fb12-huge1-s71     kev_train_huge.jsonl  1 71 3e-5
echo "=== chain2 done $(date -u +%H:%M:%S)"
