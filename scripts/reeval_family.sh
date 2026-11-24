#!/bin/zsh
# Re-run the full evaluation of a trained family (sv|en) with the current minimal-pair files.
# Keeps the previous eval.json as eval-prev.json.
set -u
LANG_=${1:?usage: reeval_family.sh sv|en|en8m}
PAIRS_LANG=${LANG_%8m}
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
export HF_HOME=$ROOT/hf-cache
cd /Users/mathiaskarlsson/Projects/smallestllm
if [[ $LANG_ == sv ]]; then TRAIN_TEXT=$ROOT/data/sagosvenska/clean/train.jsonl
elif [[ $LANG_ == en8m ]]; then TRAIN_TEXT=$ROOT/data/tinystories/small-train.txt
else TRAIN_TEXT=$ROOT/data/tinystories/TinyStoriesV2-GPT4-train.txt; fi
for cfg in pytte-0.3m pytte-1m pytte-2m pytte-3m pytte-7m pytte-15m; do
  out=$ROOT/checkpoints/$cfg-$LANG_
  [[ -f $out/summary.json ]] || { echo "[reeval] skip $cfg-$LANG_ (not trained)"; continue; }
  vocab=$($PY -c "import json;print(json.load(open('$out/summary.json'))['config']['vocab_size'])")
  echo "[reeval] $(date -Iseconds) eval $cfg-$LANG_ (vocab $vocab)"
  [[ -f $out/eval.json ]] && cp $out/eval.json $out/eval-prev.json
  $PY -m pytte.eval --model $out --tok $ROOT/data/tok-$LANG_-$vocab --data $ROOT/data/tokens-$LANG_-$vocab \
    --pairs eval/minimal_pairs_$PAIRS_LANG.json --train-text $TRAIN_TEXT --n-samples 100 \
    || echo "[reeval] eval $cfg-$LANG_ FAILED"
done
echo "[reeval] $(date -Iseconds) done $LANG_"
