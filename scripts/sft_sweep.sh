#!/bin/zsh
# SFT sweep 2026-11-13 on pytte-3m-sv3: the 4-min / lr 5e-4 pilot was undertrained (val loss still
# falling, noun followed in 44 %). Three settings, same data (now with "episode" follow-ups):
#   a) lr 1e-4, 8 min   b) lr 3e-4, 8 min   c) lr 5e-4, 12 min
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
LOG=$ROOT/checkpoints/sft_sweep.log
TOK=$ROOT/data/tok-sv3-4096
DATA=$ROOT/data/tokens-sft-sv3-4096
cd /Users/mathiaskarlsson/Projects/smallestllm
export HF_HOME=$ROOT/hf-cache
for spec in "a 1e-4 8" "b 3e-4 8" "c 5e-4 12"; do
  set -- ${=spec}; tag=$1; lr=$2; min=$3
  OUT=$ROOT/checkpoints/sftsweep/pytte-3m-sv3-chat-$tag
  [[ -f $OUT/eval.json ]] && { echo "[sweep] $tag done already" >> $LOG; continue; }
  echo "[sweep] $(date -Iseconds) $tag: lr $lr, $min min" >> $LOG
  $PY -m pytte.train --init $ROOT/checkpoints/pytte-3m-sv3 --data $DATA --tok $TOK --out $OUT \
      --minutes $min --lr $lr --min-lr 2e-5 --warmup 0.05 --eval-every 250 >> $LOG 2>&1 || { echo "[sweep] FAILED train $tag" >> $LOG; continue; }
  $PY scripts/eval_sft.py --model $OUT --tok $TOK --data $ROOT/data/sagosvenska-mix1/clean \
      --lexicon $ROOT/data/sft-mix1/lexicon.json --n 100 >> $LOG 2>&1 || echo "[sweep] FAILED eval_sft $tag" >> $LOG
  $PY -m pytte.eval --model $OUT --tok $TOK --pairs eval/minimal_pairs_sv.json --n-samples 0 --out $OUT/eval_pairs.json >> $LOG 2>&1 \
      || echo "[sweep] FAILED pairs $tag" >> $LOG
done
echo "[sweep] $(date -Iseconds) all done (sft sweep)" >> $LOG
