#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full-length dynamic functional connectivity sequences."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeseries-key", default="bold")
    parser.add_argument("--tr-key", default="tr")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--window-seconds", type=float, default=15.0)
    parser.add_argument("--stride-seconds", type=float, default=3.0)
    return parser.parse_args()


def correlation_matrix(signal: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = torch.from_numpy(np.asarray(signal)).T
    centered = values - values.mean(1, keepdim=True)
    covariance = centered @ centered.T / (values.size(1) - 1)
    scale = torch.sqrt(torch.diag(covariance) + eps)
    correlation = covariance / scale[:, None] / scale[None, :]
    return correlation.clamp(-1.0, 1.0).numpy()


def dynamic_connectivity(
    timeseries: np.ndarray,
    repetition_time: float,
    window_seconds: float,
    stride_seconds: float,
) -> tuple[np.ndarray, int, int]:
    window = int(np.floor(window_seconds / repetition_time))
    stride = int(np.floor(stride_seconds / repetition_time))
    if window < 2 or stride < 1:
        raise ValueError(
            f"Invalid window/stride after TR conversion: window={window}, stride={stride}."
        )
    if timeseries.shape[0] < window:
        raise ValueError(
            f"A scan with {timeseries.shape[0]} time points is shorter than window={window}."
        )
    starts = range(0, timeseries.shape[0] - window + 1, stride)
    matrices = [correlation_matrix(timeseries[start : start + window]) for start in starts]
    return np.stack(matrices).astype(np.float32), window, stride


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")

    with np.load(args.input, allow_pickle=True) as source:
        required = {args.timeseries_key, args.tr_key, args.label_key}
        missing = required - set(source.files)
        if missing:
            raise ValueError(f"Input archive is missing: {', '.join(sorted(missing))}")
        timeseries = source[args.timeseries_key]
        repetition_times = np.asarray(source[args.tr_key], dtype=float)
        labels = np.asarray(source[args.label_key], dtype=np.int64)
        metadata = {
            key: source[key]
            for key in source.files
            if key not in {args.timeseries_key, "bold", "data", "bold_znorm"}
        }

    if not (len(timeseries) == len(repetition_times) == len(labels)):
        raise ValueError("Time series, TR, and label arrays must have the same length.")

    sequences = []
    windows = []
    strides = []
    for index, (signal, repetition_time) in enumerate(zip(timeseries, repetition_times)):
        sequence, window, stride = dynamic_connectivity(
            np.asarray(signal),
            float(repetition_time),
            args.window_seconds,
            args.stride_seconds,
        )
        if not np.isfinite(sequence).all():
            raise ValueError(f"Non-finite dFC values for subject row {index}.")
        sequences.append(sequence)
        windows.append(window)
        strides.append(stride)

    provenance = json.dumps(
        {
            "window_seconds": args.window_seconds,
            "stride_seconds": args.stride_seconds,
            "tr_normalized": True,
            "correlation_epsilon": 1e-6,
            "dtype": "float32",
        },
        sort_keys=True,
    )
    dfc = np.empty(len(sequences), dtype=object)
    dfc[:] = sequences
    metadata.update(
        {
            "label": labels,
            "dfc": dfc,
            "n_window": np.asarray([len(sequence) for sequence in sequences]),
            "win_tr": np.asarray(windows),
            "stride_tr": np.asarray(strides),
            "provenance": np.asarray(provenance),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **metadata)
    print(f"Saved {len(sequences)} subjects to {args.output}")


if __name__ == "__main__":
    main()
