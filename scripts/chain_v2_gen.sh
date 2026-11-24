#!/bin/zsh
# SagoSvenska v2 generation (2026-11-10): Gemma 4 26B A4B, recipe v2, temp 0.85, 60 000 stories.
# Waits for the seed check to leave the GPU, then runs the resumable loop (battery guard, restarts).
set -u
ROOT=/Volumes/Modelltest/pytte
export HF_HOME=$ROOT/hf-cache
export GEN_DIR=$ROOT/data/sagosvenska-v2
LOG=$GEN_DIR/gen.log
echo "[v2] $(date -Iseconds) waiting for the seed check to finish" >> $LOG
while ! grep -q "all done (seed check)" $ROOT/checkpoints/seedcheck.log 2>/dev/null; do sleep 30; done
while pgrep -f "pytte.train|pytte.eval|judge_pilots|judge_family" >/dev/null; do sleep 15; done
echo "[v2] $(date -Iseconds) GPU free, starting generation" >> $LOG
/Users/mathiaskarlsson/Projects/smallestllm/data/gen_loop.sh \
  --model mlx-community/gemma-4-26B-A4B-it-qat-4bit --recipe v2 --target 60000 --temp 0.85 \
  --max-tokens 520 --seed 20261110 --out $GEN_DIR/raw >> $LOG 2>&1
echo "[v2] $(date -Iseconds) loop exited with $?" >> $LOG
