#!/bin/zsh
# Full re-evaluation of both families (stores all 100 samples) followed by the LLM judge on 50 samples each.
set -u
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
cd /Volumes/Modelltest/pytte/checkpoints
$S/reeval_family.sh sv >> reeval-sv.log 2>&1
$S/reeval_family.sh en >> reeval-en.log 2>&1
/Volumes/Modelltest/pytte/.venv/bin/python $S/judge_family.py --n 50 --force >> judge.log 2>&1
echo "[chain2] $(date -Iseconds) all done" >> judge.log
