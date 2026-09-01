#!/usr/bin/env python3
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from dataset.dataloader import load_dfc_archive, load_splits
from training.training import train_experiment


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    data_path = Path(to_absolute_path(cfg.dataset.data_path))
    split_path = Path(to_absolute_path(cfg.dataset.split_path))
    output_dir = Path(to_absolute_path(cfg.output_dir))

    sequences, labels = load_dfc_archive(data_path)
    splits = load_splits(split_path)

    device_name = cfg.device
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = output_dir / cfg.dataset.key / f"seed_{cfg.seed}"
    report = train_experiment(
        cfg=cfg,
        sequences=sequences,
        labels=labels,
        splits=splits,
        output_dir=run_dir,
        device=torch.device(device_name),
    )

    print(f"Results: {run_dir / 'test_result.csv'}")
    for name, values in report["summary"].items():
        print(f"{name}: {values['mean']:.4f} +/- {values['std']:.4f}")


if __name__ == "__main__":
    main()
