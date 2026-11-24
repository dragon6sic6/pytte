#!/bin/zsh
# SFT pilot 2026-11-13: fine-tune pytte-3m-sv3 on the chat data built from mix1's recipes
# (data/tokens-sft-sv3-4096), then test instruction following, minimal pairs and the judge.
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
LOG=$ROOT/checkpoints/sft_pilot.log
TOK=$ROOT/data/tok-sv3-4096
DATA=$ROOT/data/tokens-sft-sv3-4096
OUT=$ROOT/checkpoints/pytte-3m-sv3-chat
cd /Users/mathiaskarlsson/Projects/smallestllm
export HF_HOME=$ROOT/hf-cache
echo "[sft] $(date -Iseconds) pilot: pytte-3m-sv3 -> chat, 4 min, lr 5e-4" >> $LOG
$PY -m pytte.train --init $ROOT/checkpoints/pytte-3m-sv3 --data $DATA --tok $TOK --out $OUT \
    --minutes 4 --lr 5e-4 --min-lr 5e-5 --warmup 0.05 --eval-every 250 >> $LOG 2>&1 || { echo "[sft] FAILED train" >> $LOG; exit 1; }
echo "[sft] $(date -Iseconds) eval_sft" >> $LOG
$PY scripts/eval_sft.py --model $OUT --tok $TOK --data $ROOT/data/sagosvenska-mix1/clean \
    --lexicon $ROOT/data/sft-mix1/lexicon.json --n 100 >> $LOG 2>&1 || echo "[sft] FAILED eval_sft" >> $LOG
echo "[sft] $(date -Iseconds) minimal pairs" >> $LOG
$PY -m pytte.eval --model $OUT --tok $TOK --pairs eval/minimal_pairs_sv.json --n-samples 0 --out $OUT/eval_pairs.json >> $LOG 2>&1 \
    || echo "[sft] FAILED pairs" >> $LOG
echo "[sft] $(date -Iseconds) judging" >> $LOG
$PY scripts/judge_family.py --n 50 --only pytte-3m-sv3-chat >> $LOG 2>&1 || echo "[sft] FAILED judge" >> $LOG
echo "[sft] $(date -Iseconds) all done (sft pilot)" >> $LOG
