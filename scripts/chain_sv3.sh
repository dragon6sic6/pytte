#!/bin/zsh
# Train the six Swedish models on mix1 (v1.1 + one-child v2, data/sagosvenska-mix1), the release mix, 2026-11-13.
# Builds tok-sv3-{2048,4096} and tokens-sv3-* on the way, evaluates, then judges with Gemma 4 12B.
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
export HF_HOME=$ROOT/hf-cache
cd /Users/mathiaskarlsson/Projects/smallestllm
LOG=$ROOT/checkpoints/family.log
echo "[sv3] $(date -Iseconds) training the Swedish family on mix1 = v1.1 + one-child v2 (85 764 stories)" >> $LOG
$S/train_family.sh sv3 >> $LOG 2>&1
echo "[sv3] $(date -Iseconds) judging" >> $LOG
$PY $S/judge_family.py --n 50 --only pytte-0.3m-sv3 pytte-1m-sv3 pytte-2m-sv3 pytte-3m-sv3 pytte-7m-sv3 pytte-15m-sv3 >> $ROOT/checkpoints/judge.log 2>&1
echo "[sv3] $(date -Iseconds) all done (sv3 family)" >> $LOG
