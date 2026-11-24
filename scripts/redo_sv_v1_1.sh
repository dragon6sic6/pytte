#!/bin/zsh
# Rebuild the Swedish pipeline after the <image|> bug (2026-11-09 evening):
# archive v1.0 artifacts, re-clean the raw stories with repair_special_tokens, rebuild
# tokenizers/tokens, retrain + evaluate the six Swedish models, judge them.
# Waits for the running 26B pilot first (two 10+ GB models do not fit in 32 GB).
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
S=/Users/mathiaskarlsson/Projects/smallestllm/scripts
export HF_HOME=$ROOT/hf-cache
cd /Users/mathiaskarlsson/Projects/smallestllm
LOG=$ROOT/checkpoints/family.log

echo "[redo] $(date -Iseconds) waiting for the 26B pilot to finish" >> $LOG
while pgrep -f "gen_stories.py.*26b-t085" >/dev/null; do sleep 20; done
echo "[redo] $(date -Iseconds) pilot done; archiving v1.0 artifacts" >> $LOG

ARCH=$ROOT/archive-v1.0-imagebug
mkdir -p $ARCH/checkpoints $ARCH/data
mv $ROOT/data/sagosvenska/clean $ARCH/data/clean
for d in $ROOT/data/tok-sv-2048 $ROOT/data/tok-sv-4096 $ROOT/data/tokens-sv-2048 $ROOT/data/tokens-sv-4096; do
  [[ -e $d ]] && mv $d $ARCH/data/
done
for d in $ROOT/checkpoints/pytte-*-sv; do
  [[ -e $d ]] && mv $d $ARCH/checkpoints/
done

echo "[redo] $(date -Iseconds) re-cleaning raw stories" >> $LOG
$PY data/clean_stories.py --raw $ROOT/data/sagosvenska/raw --out $ROOT/data/sagosvenska/clean > $ROOT/data/sagosvenska/clean-v1.1.log 2>&1 \
  || { echo "[redo] clean FAILED" >> $LOG; exit 1; }
if grep -q '<image' $ROOT/data/sagosvenska/clean/train.jsonl; then echo "[redo] junk still present, aborting" >> $LOG; exit 1; fi

echo "[redo] $(date -Iseconds) training the Swedish family on v1.1" >> $LOG
$S/train_family.sh sv >> $LOG 2>&1
echo "[redo] $(date -Iseconds) judging" >> $LOG
$PY $S/judge_family.py --n 50 --only pytte-0.3m-sv pytte-1m-sv pytte-2m-sv pytte-3m-sv pytte-7m-sv pytte-15m-sv >> $ROOT/checkpoints/judge.log 2>&1
echo "[redo] $(date -Iseconds) all done (v1.1 Swedish family)" >> $LOG
