#!/bin/zsh
# Does coherence come back with a longer budget on the merged data? (2026-11-12, after Fynd 14)
# pytte-1m 16 min and pytte-3m 24 min (double the family budgets) on sagosvenska-all, then eval + judge.
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
export HF_HOME=$ROOT/hf-cache
cd /Users/mathiaskarlsson/Projects/smallestllm
OUTROOT=$ROOT/checkpoints/sv2long
TOK=$ROOT/data/tok-sv2-4096
TOKENS=$ROOT/data/tokens-sv2-4096
TRAIN_TEXT=$ROOT/data/sagosvenska-all/clean/train.jsonl
LOG=$ROOT/checkpoints/sv2long.log
mkdir -p $OUTROOT
for spec in "pytte-1m 16 2e-3" "pytte-3m 24 2e-3"; do
  read -r cfg minutes lr <<< "$spec"
  out=$OUTROOT/$cfg-sv2
  echo "[long] $(date -Iseconds) train $cfg for $minutes min (lr $lr)" >> $LOG
  $PY -m pytte.train --config $cfg --data $TOKENS --tok $TOK --out $out --minutes $minutes --lr $lr --eval-every 250 >> $LOG 2>&1 \
    || { echo "[long] train $cfg FAILED" >> $LOG; continue; }
  echo "[long] $(date -Iseconds) eval $cfg" >> $LOG
  $PY -m pytte.eval --model $out --tok $TOK --data $TOKENS --pairs eval/minimal_pairs_sv.json --train-text $TRAIN_TEXT --n-samples 100 >> $LOG 2>&1 \
    || echo "[long] eval $cfg FAILED" >> $LOG
done
echo "[long] $(date -Iseconds) judging" >> $LOG
$PY $S/judge_family.py --root $OUTROOT --n 50 >> $LOG 2>&1
echo "[long] $(date -Iseconds) all done (sv2 long budget)" >> $LOG
