#!/bin/zsh
# Chat-tune the six sv3 models (2026-11-13). Usage: chain_sft_family.sh <data-suffix> <lr> <scale>
#   data-suffix: "" or "-dense" (tokens-sft-sv3-<vocab><suffix>); scale multiplies the family budgets
#   (0.3m 5, 1m 8, 2m 10, 3m 12, 7m 20, 15m 30 min) -> e.g. scale 1.0 = same time as pretraining.
set -u
SUFFIX=${1:-}; LR=${2:-3e-4}; SCALE=${3:-1.0}
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
LOG=$ROOT/checkpoints/sft_family.log
cd /Users/mathiaskarlsson/Projects/smallestllm
export HF_HOME=$ROOT/hf-cache
echo "[sftfam] $(date -Iseconds) start: data suffix '$SUFFIX', lr $LR, scale $SCALE" >> $LOG
for spec in "pytte-0.3m 5 2048" "pytte-1m 8 4096" "pytte-2m 10 4096" "pytte-3m 12 4096" "pytte-7m 20 4096" "pytte-15m 30 4096"; do
  set -- ${=spec}; cfg=$1; base_min=$2; vocab=$3
  min=$(python3 -c "print(round($base_min*$SCALE, 1))")
  TOK=$ROOT/data/tok-sv3-$vocab
  DATA=$ROOT/data/tokens-sft-sv3-$vocab$SUFFIX
  OUT=$ROOT/checkpoints/$cfg-sv3-chat
  if [[ -f $OUT/eval.json ]]; then echo "[sftfam] $cfg already done, skipping" >> $LOG; continue; fi
  echo "[sftfam] $(date -Iseconds) train $cfg for $min min (lr $LR)" >> $LOG
  $PY -m pytte.train --init $ROOT/checkpoints/$cfg-sv3 --data $DATA --tok $TOK --out $OUT \
      --minutes $min --lr $LR --min-lr 2e-5 --warmup 0.05 --eval-every 250 >> $LOG 2>&1 || { echo "[sftfam] FAILED train $cfg" >> $LOG; continue; }
  echo "[sftfam] $(date -Iseconds) eval $cfg" >> $LOG
  $PY scripts/eval_sft.py --model $OUT --tok $TOK --data $ROOT/data/sagosvenska-mix1/clean \
      --lexicon $ROOT/data/sft-mix1/lexicon.json --n 100 >> $LOG 2>&1 || echo "[sftfam] FAILED eval_sft $cfg" >> $LOG
  $PY -m pytte.eval --model $OUT --tok $TOK --pairs eval/minimal_pairs_sv.json --n-samples 0 --out $OUT/eval_pairs.json >> $LOG 2>&1 \
      || echo "[sftfam] FAILED pairs $cfg" >> $LOG
done
echo "[sftfam] $(date -Iseconds) judging" >> $LOG
$PY scripts/judge_family.py --n 50 --only pytte-0.3m-sv3-chat pytte-1m-sv3-chat pytte-2m-sv3-chat pytte-3m-sv3-chat pytte-7m-sv3-chat pytte-15m-sv3-chat >> $LOG 2>&1 \
    || echo "[sftfam] FAILED judge" >> $LOG
echo "[sftfam] $(date -Iseconds) all done (sft family)" >> $LOG
