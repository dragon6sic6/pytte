#!/bin/zsh
# After the 16:12 panic: judge the one missing model, then train+eval+judge the size-matched English family.
set -u
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
PY=/Volumes/Modelltest/pytte/.venv/bin/python
cd /Volumes/Modelltest/pytte/checkpoints
echo "[chain4] $(date -Iseconds) start" >> judge.log
$PY $S/judge_family.py --n 50 --force --only pytte-7m-sv >> judge.log 2>&1
$S/train_family.sh en8m >> family.log 2>&1
$PY $S/judge_family.py --n 50 >> judge.log 2>&1
echo "[chain4] $(date -Iseconds) all done" >> judge.log
