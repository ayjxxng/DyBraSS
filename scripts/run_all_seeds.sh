#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <experiment> <dfc.npz> <split-directory>" >&2
  exit 2
fi

experiment=$1
data=$2
split_dir=$3

for seed in 0 1 2 3 42; do
  python main.py \
    experiment="$experiment" \
    dataset.data_path="$data" \
    dataset.split_path="$split_dir/seed_${seed}.json" \
    seed="$seed"
done
