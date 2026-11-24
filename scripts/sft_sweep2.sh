#!/bin/zsh
# SFT sweep 2, 2026-11-13: the first sweep showed that lr and time (8 min) did not move
# noun/adjective following (37-44 % / 15 %). Test the data lever: "dense" requests
# (2-3 ingredients, noun in 76 % of requests instead of ~50 %), at the same 8 min and at 20 min.
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
LOG=$ROOT/checkpoints/sft_sweep.log
TOK=$ROOT/data/tok-sv3-4096
DATA=$ROOT/data/tokens-sft-sv3-4096-dense
cd /Users/mathiaskarlsson/Projects/smallestllm
export HF_HOME=$ROOT/hf-cache
while pgrep -f sft_sweep.sh > /dev/null; do sleep 15; done   # wait for sweep 1 (GPU)
for spec in "d 3e-4 8" "e 3e-4 20"; do
  set -- ${=spec}; tag=$1; lr=$2; min=$3
  OUT=$ROOT/checkpoints/sftsweep/pytte-3m-sv3-chat-$tag
  [[ -f $OUT/eval.json ]] && { echo "[sweep2] $tag done already" >> $LOG; continue; }
  echo "[sweep2] $(date -Iseconds) $tag: dense data, lr $lr, $min min" >> $LOG
  $PY -m pytte.train --init $ROOT/checkpoints/pytte-3m-sv3 --data $DATA --tok $TOK --out $OUT \
      --minutes $min --lr $lr --min-lr 2e-5 --warmup 0.05 --eval-every 250 >> $LOG 2>&1 || { echo "[sweep2] FAILED train $tag" >> $LOG; continue; }
  $PY scripts/eval_sft.py --model $OUT --tok $TOK --data $ROOT/data/sagosvenska-mix1/clean \
      --lexicon $ROOT/data/sft-mix1/lexicon.json --n 100 >> $LOG 2>&1 || echo "[sweep2] FAILED eval_sft $tag" >> $LOG
  $PY -m pytte.eval --model $OUT --tok $TOK --pairs eval/minimal_pairs_sv.json --n-samples 0 --out $OUT/eval_pairs.json >> $LOG 2>&1 \
      || echo "[sweep2] FAILED pairs $tag" >> $LOG
done
echo "[sweep2] $(date -Iseconds) all done (sft sweep 2)" >> $LOG
