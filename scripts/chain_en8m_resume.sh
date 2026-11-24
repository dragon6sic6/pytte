#!/bin/zsh
# Resume after the 7m-en8m run died during lid-close sleep (2026-11-09 17:27): finish en8m, judge it.
set -u
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
PY=/Volumes/Modelltest/pytte/.venv/bin/python
cd /Volumes/Modelltest/pytte/checkpoints
echo "[family] $(date -Iseconds) 7m-en8m died during sleep, retraining from 7m" >> family.log
$S/train_family.sh en8m >> family.log 2>&1
$PY $S/judge_family.py --n 50 >> judge.log 2>&1
echo "[chain4] $(date -Iseconds) all done (resumed chain)" >> judge.log
