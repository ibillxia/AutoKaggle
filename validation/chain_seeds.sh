#!/usr/bin/env bash
# Sequential seed-ensemble training on the local 4060 Ti (8 GB).
# Each run: XLM-R large, freeze-below 12, max-len 128 (99.9% of dev fits), batch 24, bf16 + adamw8bit.
# train_hf_nli2.py evaluates on the dev instrument at the end and writes rows.json for stacking.
set -u
cd /e/Bill/AI/AutoKaggle || exit 1
PY=C:/Users/deepg/.workbuddy/binaries/python/envs/default/Scripts/python.exe
DATA=data/comp/contradictory-my-dear-watson

run () {  # name data epochs seed
  local name=$1 data=$2 eps=$3 seed=$4
  if [ -f "validation/runs/${name}-dev/rows.json" ]; then echo "skip $name (already done)"; return; fi
  echo "=== $name  $(date -u +%H:%M:%S)"
  PYTHONUNBUFFERED=1 "$PY" -u validation/train_hf_nli2.py \
    --model models/xlm-roberta-large --data "$DATA/$data" --eval "$DATA/kev_dev.jsonl" \
    --out "runs/$name" --out-rows "validation/runs/${name}-dev/rows.json" \
    --epochs "$eps" --lr 3e-5 --batch 24 --max-len 128 --optim adamw8bit --precision bf16 \
    --freeze-below 12 --save 1 --log-every 1000 --warmup-frac 0.03 --eval-batch 32 --seed "$seed" \
    > "validation/${name}.log" 2>&1
  grep -E "EVAL" "validation/${name}.log" | tail -1
}

run xlmr-fb12-big-s21  kev_train_big.jsonl  2 21
run xlmr-fb12-big-s31  kev_train_big.jsonl  2 31
run xlmr-fb12-huge2-s41 kev_train_huge.jsonl 2 41
echo "=== chain done $(date -u +%H:%M:%S)"
