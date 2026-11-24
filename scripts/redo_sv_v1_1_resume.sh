#!/bin/zsh
# Resume the v1.1 Swedish rebuild after the 3m GPU error (train_family skips finished models).
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
export HF_HOME=$ROOT/hf-cache
cd /Users/mathiaskarlsson/Projects/smallestllm
LOG=$ROOT/checkpoints/family.log
echo "[redo] $(date -Iseconds) resuming v1.1 after GPU error on 3m (7m restarted too)" >> $LOG
$S/train_family.sh sv >> $LOG 2>&1
echo "[redo] $(date -Iseconds) judging" >> $LOG
$PY $S/judge_family.py --n 50 --only pytte-0.3m-sv pytte-1m-sv pytte-2m-sv pytte-3m-sv pytte-7m-sv pytte-15m-sv >> $ROOT/checkpoints/judge.log 2>&1
echo "[redo] $(date -Iseconds) all done (v1.1 Swedish family)" >> $LOG
