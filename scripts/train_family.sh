#!/bin/zsh
# Train the whole Pytte family for one language with fixed wall-clock budgets, then evaluate each.
#   scripts/train_family.sh sv        # -> checkpoints/pytte-<cfg>-sv/
#   scripts/train_family.sh en
#   scripts/train_family.sh en8m     # TinyStories subset of SagoSvenska's size (controls for data size)
#   scripts/train_family.sh sv2      # SagoSvenska v1.1 + v2 merged (data/sagosvenska-all), 2026-11-12
#   scripts/train_family.sh sv3      # v1.1 + one-child v2 stories (data/sagosvenska-mix1), the release mix, 2026-11-13
# Budgets are minutes of training; the same budgets for both languages make the twin comparison fair.
set -u
LANG_=${1:?usage: train_family.sh sv|en|en8m|sv2|sv3}
PAIRS_LANG=${LANG_%8m}   # en8m uses the English minimal pairs
PAIRS_LANG=${PAIRS_LANG%[23]}   # sv2 and sv3 use the Swedish minimal pairs
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
export HF_HOME=$ROOT/hf-cache
TRAIN_TEXT=$ROOT/data/sagosvenska/clean/train.jsonl
[[ $LANG_ == en ]] && TRAIN_TEXT=$ROOT/data/tinystories/TinyStoriesV2-GPT4-train.txt
[[ $LANG_ == en8m ]] && TRAIN_TEXT=$ROOT/data/tinystories/small-train.txt
[[ $LANG_ == sv2 ]] && TRAIN_TEXT=$ROOT/data/sagosvenska-all/clean/train.jsonl
[[ $LANG_ == sv3 ]] && TRAIN_TEXT=$ROOT/data/sagosvenska-mix1/clean/train.jsonl
cd /Users/mathiaskarlsson/Projects/smallestllm

# config  minutes  lr
BUDGETS=(
  "pytte-0.3m  5  2e-3"
  "pytte-1m    8  2e-3"
  "pytte-2m   10  2e-3"
  "pytte-3m   12  2e-3"
  "pytte-7m   20  1e-3"
  "pytte-15m  30  1e-3"
)
VAL_TEXT=$ROOT/data/sagosvenska/clean/val.jsonl
[[ $LANG_ == en ]] && VAL_TEXT=$ROOT/data/tinystories/TinyStoriesV2-GPT4-valid.txt
[[ $LANG_ == en8m ]] && VAL_TEXT=$ROOT/data/tinystories/small-valid.txt
[[ $LANG_ == sv2 ]] && VAL_TEXT=$ROOT/data/sagosvenska-all/clean/val.jsonl
[[ $LANG_ == sv3 ]] && VAL_TEXT=$ROOT/data/sagosvenska-mix1/clean/val.jsonl

for spec in $BUDGETS; do
  read -r cfg minutes lr <<< "$spec"
  out=$ROOT/checkpoints/$cfg-$LANG_
  vocab=$($PY -c "from pytte.model import FAMILY; print(FAMILY['$cfg'].vocab_size)")
  TOK=$ROOT/data/tok-$LANG_-$vocab
  TOKENS=$ROOT/data/tokens-$LANG_-$vocab
  if [[ ! -f $TOKENS/meta.json ]]; then
    echo "[family] $(date -Iseconds) building tokenizer+tokens for vocab $vocab ($LANG_)"
    $PY -m pytte.tokenizer train --input $TRAIN_TEXT --vocab $vocab --out $TOK
    $PY -m pytte.prepare_data --tok $TOK --train $TRAIN_TEXT --val $VAL_TEXT --out $TOKENS
  fi
  if [[ -f $out/summary.json ]]; then echo "[family] $cfg-$LANG_ already trained, skipping"; continue; fi
  echo "[family] $(date -Iseconds) train $cfg-$LANG_ for $minutes min (lr $lr)"
  $PY -m pytte.train --config $cfg --data $TOKENS --tok $TOK --out $out --minutes $minutes --lr $lr --eval-every 250 \
    || { echo "[family] train $cfg-$LANG_ FAILED"; continue; }
  echo "[family] $(date -Iseconds) eval $cfg-$LANG_"
  $PY -m pytte.eval --model $out --tok $TOK --data $TOKENS --pairs eval/minimal_pairs_$PAIRS_LANG.json \
    --train-text $TRAIN_TEXT --n-samples 100 || echo "[family] eval $cfg-$LANG_ FAILED"
done
echo "[family] $(date -Iseconds) done $LANG_"
