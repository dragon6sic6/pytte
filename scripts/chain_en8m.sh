#!/bin/zsh
# Waits for the re-eval+judge chain, then trains and judges the size-matched English family.
set -u
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
cd /Volumes/Modelltest/pytte/checkpoints
until grep -q "\[chain2\] .* all done" judge.log; do sleep 60; done
$S/train_family.sh en8m >> family.log 2>&1
/Volumes/Modelltest/pytte/.venv/bin/python $S/judge_family.py --n 50 >> judge.log 2>&1
echo "[chain3] $(date -Iseconds) en8m done" >> judge.log
