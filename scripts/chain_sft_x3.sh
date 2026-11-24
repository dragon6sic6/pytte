#!/bin/zsh
# Retrain chat models on the x3 data (three request passes per story, 2026-11-14).
# Usage: chain_sft_x3.sh <cfg> <minutes> [<cfg> <minutes> ...]   e.g. pytte-15m 75 pytte-7m 50
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
LOG=$ROOT/checkpoints/sft_family.log
TOK=$ROOT/data/tok-sv3-4096
DATA=$ROOT/data/tokens-sft-sv3-4096-dense-x3
cd /Users/mathiaskarlsson/Projects/smallestllm
export HF_HOME=$ROOT/hf-cache
echo "[sftx3] $(date -Iseconds) start: $*" >> $LOG
DONE=()
while [[ $# -ge 2 ]]; do
  cfg=$1; min=$2; shift 2
  OUT=$ROOT/checkpoints/$cfg-sv3-chat
  if [[ -f $OUT/eval.json ]]; then echo "[sftx3] $cfg already done, skipping" >> $LOG; DONE+=($cfg-sv3-chat); continue; fi
  echo "[sftx3] $(date -Iseconds) train $cfg for $min min on x3 data (lr 3e-4)" >> $LOG
  $PY -m pytte.train --init $ROOT/checkpoints/$cfg-sv3 --data $DATA --tok $TOK --out $OUT \
      --minutes $min --lr 3e-4 --min-lr 2e-5 --warmup 0.05 --eval-every 250 >> $LOG 2>&1 || { echo "[sftx3] FAILED train $cfg" >> $LOG; continue; }
  echo "[sftx3] $(date -Iseconds) eval $cfg" >> $LOG
  $PY scripts/eval_sft.py --model $OUT --tok $TOK --data $ROOT/data/sagosvenska-mix1/clean \
      --lexicon $ROOT/data/sft-mix1/lexicon.json --n 100 >> $LOG 2>&1 || echo "[sftx3] FAILED eval_sft $cfg" >> $LOG
  $PY -m pytte.eval --model $OUT --tok $TOK --pairs eval/minimal_pairs_sv.json --n-samples 0 --out $OUT/eval_pairs.json >> $LOG 2>&1 \
      || echo "[sftx3] FAILED pairs $cfg" >> $LOG
  DONE+=($cfg-sv3-chat)
done
echo "[sftx3] $(date -Iseconds) judging ${DONE[*]}" >> $LOG
$PY scripts/judge_family.py --n 50 --only ${DONE[*]} >> $LOG 2>&1 || echo "[sftx3] FAILED judge" >> $LOG
echo "[sftx3] $(date -Iseconds) all done (sft x3)" >> $LOG
