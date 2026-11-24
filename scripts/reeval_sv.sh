#!/bin/zsh
# Re-run the full evaluation of the Swedish family with minimal pairs v2 (200 pairs).
set -u
ROOT=/Volumes/Modelltest/pytte
PY=$ROOT/.venv/bin/python
cd /Users/mathiaskarlsson/Projects/smallestllm
for cfg in pytte-0.3m pytte-1m pytte-2m pytte-3m pytte-7m pytte-15m; do
  out=$ROOT/checkpoints/$cfg-sv
  vocab=$($PY -c "import json;print(json.load(open('$out/summary.json'))['config']['vocab_size'])")
  echo "[reeval] $(date -Iseconds) eval $cfg-sv (vocab $vocab)"
  cp $out/eval.json $out/eval-v1pairs.json 2>/dev/null
  $PY -m pytte.eval --model $out --tok $ROOT/data/tok-sv-$vocab --data $ROOT/data/tokens-sv-$vocab \
    --pairs eval/minimal_pairs_sv.json --train-text $ROOT/data/sagosvenska/clean/train.jsonl --n-samples 100 \
    || echo "[reeval] eval $cfg-sv FAILED"
done
echo "[reeval] $(date -Iseconds) done sv"
