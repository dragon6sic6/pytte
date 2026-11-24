#!/bin/zsh
# Noise check for the Swedish minimal-pair test (2026-11-10): retrain pytte-0.3m-sv and pytte-1m-sv
# with seeds 1 and 2 (same budgets and data as v1.1) and evaluate. Waits for the pilot judges first.
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
export HF_HOME=$ROOT/hf-cache
cd /Users/mathiaskarlsson/Projects/smallestllm
LOG=$ROOT/checkpoints/seedcheck.log
TRAIN_TEXT=$ROOT/data/sagosvenska/clean/train.jsonl
echo "[seed] $(date -Iseconds) waiting for pilot judges" >> $LOG
while ! grep -q "all done (pilot judges)" $ROOT/data/pilot-v2/judge-pilots.log 2>/dev/null; do sleep 30; done
while pgrep -f "judge_pilots.py" >/dev/null; do sleep 15; done
for spec in "pytte-0.3m 5 2e-3 2048" "pytte-1m 8 2e-3 4096"; do
  read -r cfg minutes lr vocab <<< "$spec"
  for seed in 1 2; do
    out=$ROOT/checkpoints/seedcheck/$cfg-sv-seed$seed
    [[ -f $out/summary.json ]] && continue
    echo "[seed] $(date -Iseconds) train $cfg-sv seed $seed for $minutes min" >> $LOG
    $PY -m pytte.train --config $cfg --data $ROOT/data/tokens-sv-$vocab --tok $ROOT/data/tok-sv-$vocab --out $out \
      --minutes $minutes --lr $lr --eval-every 250 --seed $seed >> $LOG 2>&1 || { echo "[seed] train FAILED" >> $LOG; continue; }
    echo "[seed] $(date -Iseconds) eval $cfg-sv seed $seed" >> $LOG
    $PY -m pytte.eval --model $out --tok $ROOT/data/tok-sv-$vocab --data $ROOT/data/tokens-sv-$vocab \
      --pairs eval/minimal_pairs_sv.json --train-text $TRAIN_TEXT --n-samples 100 >> $LOG 2>&1 || echo "[seed] eval FAILED" >> $LOG
  done
done
echo "[seed] $(date -Iseconds) all done (seed check)" >> $LOG
