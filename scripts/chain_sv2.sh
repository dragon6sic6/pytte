#!/bin/zsh
# Retrain the six Swedish models on SagoSvenska v1.1 + v2 (data/sagosvenska-all), 2026-11-12.
# Builds tok-sv2-{2048,4096} and tokens-sv2-* on the way, evaluates, then judges with Gemma 4 12B.
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
export HF_HOME=$ROOT/hf-cache
cd /Users/mathiaskarlsson/Projects/smallestllm
LOG=$ROOT/checkpoints/family.log
echo "[sv2] $(date -Iseconds) training the Swedish family on v1.1 + v2 (118 269 stories)" >> $LOG
$S/train_family.sh sv2 >> $LOG 2>&1
echo "[sv2] $(date -Iseconds) judging" >> $LOG
$PY $S/judge_family.py --n 50 --only pytte-0.3m-sv2 pytte-1m-sv2 pytte-2m-sv2 pytte-3m-sv2 pytte-7m-sv2 pytte-15m-sv2 >> $ROOT/checkpoints/judge.log 2>&1
echo "[sv2] $(date -Iseconds) all done (sv2 family)" >> $LOG
