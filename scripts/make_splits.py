#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified DyBraSS splits.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--include-indices",
        type=Path,
        help="Optional text file containing one archive row index per line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    with np.load(args.data, allow_pickle=True) as archive:
        labels = np.asarray(archive["label"], dtype=np.int64)

    if args.include_indices:
        positions = np.asarray(
            [
                int(line.strip())
                for line in args.include_indices.read_text().splitlines()
                if line.strip()
            ],
            dtype=np.int64,
        )
    else:
        positions = np.arange(len(labels), dtype=np.int64)
    if len(np.unique(positions)) != len(positions):
        raise ValueError("include-indices contains duplicate rows.")
    if positions.size == 0 or positions.min() < 0 or positions.max() >= len(labels):
        raise ValueError("include-indices contains an out-of-range row.")

    selected_labels = labels[positions]
    outer = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = {}
    for fold_number, (remainder, test) in enumerate(
        outer.split(positions, selected_labels), start=1
    ):
        n_valid = round(args.validation_fraction * len(positions))
        inner = StratifiedShuffleSplit(
            n_splits=1, test_size=n_valid, random_state=args.seed + fold_number
        )
        train, valid = next(inner.split(remainder, selected_labels[remainder]))
        splits[f"fold{fold_number}"] = {
            "train": positions[remainder[train]].tolist(),
            "valid": positions[remainder[valid]].tolist(),
            "test": positions[test].tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.folds} folds for {len(positions)} subjects to {args.output}")


if __name__ == "__main__":
    main()
