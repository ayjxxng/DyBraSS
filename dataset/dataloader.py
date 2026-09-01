from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VariableLengthDFCDataset(Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        if len(sequences) != len(labels):
            raise ValueError("The number of sequences and labels must match.")
        self.sequences = [
            torch.as_tensor(np.asarray(sequence), dtype=torch.float32)
            for sequence in sequences
        ]
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"inputs": self.sequences[index], "label": self.labels[index]}


def collate_variable_length(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    sequences = [sample["inputs"] for sample in batch]
    labels = torch.stack([sample["label"] for sample in batch])
    max_length = max(sequence.size(0) for sequence in sequences)
    n_channels = sequences[0].size(-1)

    inputs = torch.zeros(
        len(sequences), max_length, n_channels, n_channels, dtype=torch.float32
    )
    mask = torch.zeros(len(sequences), max_length, dtype=torch.float32)
    for index, sequence in enumerate(sequences):
        length = sequence.size(0)
        inputs[index, :length] = sequence
        mask[index, :length] = 1.0
    return {"inputs": inputs, "labels": labels, "mask": mask}


def load_dfc_archive(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=True) as archive:
        sequences = archive["dfc"]
        labels = np.asarray(archive["label"], dtype=np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be encoded as 0 and 1.")
    return sequences, labels


def load_splits(path: str | Path) -> dict[str, dict[str, list[int]]]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def build_loaders(
    sequences: np.ndarray,
    labels: np.ndarray,
    fold: dict[str, list[int]],
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for partition in ("train", "valid", "test"):
        indices = np.asarray(fold[partition], dtype=np.int64)
        dataset = VariableLengthDFCDataset(sequences[indices], labels[indices])
        loaders[partition] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=partition == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_variable_length,
        )
    return loaders


def mixup_batch(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    one_hot = torch.nn.functional.one_hot(labels, num_classes=2).to(inputs.dtype)
    if alpha <= 0:
        return inputs, one_hot, mask

    coefficient = torch.distributions.Beta(alpha, alpha).sample().to(inputs.device)
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_mask = mask * mask[permutation]
    mixed_inputs = coefficient * inputs + (1 - coefficient) * inputs[permutation]
    mixed_inputs = mixed_inputs * mixed_mask[:, :, None, None]
    mixed_labels = coefficient * one_hot + (1 - coefficient) * one_hot[permutation]
    return mixed_inputs, mixed_labels, mixed_mask
