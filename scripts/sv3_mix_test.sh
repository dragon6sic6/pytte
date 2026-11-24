#!/bin/zsh
# Mix test (2026-11-13, after Fynd 15): all of v1.1 + only the one-child v2 stories (data/sagosvenska-mix1,
# 85 764 training stories). Builds tok-sv3-4096 + tokens, trains pytte-1m (8 min) and pytte-3m (12 min)
# with the family budgets, evaluates, judges. Results in checkpoints/sv3mix/.
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
export HF_HOME=$ROOT/hf-cache
cd /Users/mathiaskarlsson/Projects/smallestllm
OUTROOT=$ROOT/checkpoints/sv3mix
TOK=$ROOT/data/tok-sv3-4096
TOKENS=$ROOT/data/tokens-sv3-4096
TRAIN_TEXT=$ROOT/data/sagosvenska-mix1/clean/train.jsonl
VAL_TEXT=$ROOT/data/sagosvenska-mix1/clean/val.jsonl
LOG=$ROOT/checkpoints/sv3mix.log
mkdir -p $OUTROOT
if [[ ! -f $TOKENS/meta.json ]]; then
  echo "[mix] $(date -Iseconds) building tokenizer+tokens (vocab 4096, sv3)" >> $LOG
  $PY -m pytte.tokenizer train --input $TRAIN_TEXT --vocab 4096 --out $TOK >> $LOG 2>&1
  $PY -m pytte.prepare_data --tok $TOK --train $TRAIN_TEXT --val $VAL_TEXT --out $TOKENS >> $LOG 2>&1
fi
for spec in "pytte-1m 8 2e-3" "pytte-3m 12 2e-3"; do
  read -r cfg minutes lr <<< "$spec"
  out=$OUTROOT/$cfg-sv3
  echo "[mix] $(date -Iseconds) train $cfg for $minutes min (lr $lr)" >> $LOG
  $PY -m pytte.train --config $cfg --data $TOKENS --tok $TOK --out $out --minutes $minutes --lr $lr --eval-every 250 >> $LOG 2>&1 \
    || { echo "[mix] train $cfg FAILED" >> $LOG; continue; }
  echo "[mix] $(date -Iseconds) eval $cfg" >> $LOG
  $PY -m pytte.eval --model $out --tok $TOK --data $TOKENS --pairs eval/minimal_pairs_sv.json --train-text $TRAIN_TEXT --n-samples 100 >> $LOG 2>&1 \
    || echo "[mix] eval $cfg FAILED" >> $LOG
done
echo "[mix] $(date -Iseconds) judging" >> $LOG
$PY $S/judge_family.py --root $OUTROOT --n 50 >> $LOG 2>&1
echo "[mix] $(date -Iseconds) all done (sv3 mix test)" >> $LOG
