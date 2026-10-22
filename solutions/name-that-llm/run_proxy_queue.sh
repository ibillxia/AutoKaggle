#!/usr/bin/env bash
# Sequential queue for the data-value proxy. One arm at a time, each writes extra/proxy/<tag>.json.
# stage-2 is always the official train only (6 epochs, calibrated so train-only sits at ~0.33 and is
# clearly overfitting); the arms differ only in what sits in stage-1.
set -u
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="C:/Users/deepg/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
COMMON="--bs 16 --maxlen 448 --ep2 6 --lr2 5e-5"

run () {
  tag=$1; shift
  if [ -f "extra/proxy/$tag.json" ]; then echo "skip $tag"; return; fi
  echo "########## $tag : $*  $(date +%H:%M:%S)"
  "$PY" -u proxy_enc.py --tag "$tag" $COMMON "$@" > "extra/proxy/$tag.log" 2>&1
  grep -E "^\[s|RESULT|stage1 rows|Error|OutOfMemory" "extra/proxy/$tag.log" | tail -6
  echo "########## $tag done $(date +%H:%M:%S)"
}

run p_none           --stage1 none
run p_pool           --stage1 pool
run p_marg           --stage1 marg
run p_poolmarg       --stage1 pool_marg
run p_poolmarg_clean --stage1 pool_marg --clean
run p_only_oleg      --stage1 marg --only oleg_corpus
run p_only_man       --stage1 marg --only maniceet
run p_drop_oleg      --stage1 pool_marg --drop oleg_corpus
echo "QUEUE DONE $(date +%H:%M:%S)"
