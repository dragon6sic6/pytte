#!/bin/zsh
# Runs the MLX and PyTorch/MPS training benchmarks on identical model configs.
set -e
PY=/Volumes/Modelltest/pytte/.venv/bin/python
cd "$(dirname "$0")"
echo "Chip: $(sysctl -n machdep.cpu.brand_string)"
for cfg in small mid; do
  for bs in 32 64; do
    $PY bench_mlx.py --config $cfg --batch $bs
    $PY bench_mlx.py --config $cfg --batch $bs --no-compile
    $PY bench_torch.py --config $cfg --batch $bs
  done
done
