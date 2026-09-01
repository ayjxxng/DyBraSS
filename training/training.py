from __future__ import annotations

import copy
import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset.dataloader import build_loaders, mixup_batch
from models.dybrass import DyBraSS
from utils.lr_scheduler import LRScheduler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Train:
    def __init__(
        self,
        cfg: DictConfig,
        fold_name: str,
        model: nn.Module,
        dataloaders: dict[str, DataLoader],
        fold_path: Path,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.fold_name = fold_name
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.fold_path = fold_path
        self.device = device
        self.current_step = 0
        self.fold_path.mkdir(parents=True, exist_ok=True)

    def train_per_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: LRScheduler,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        n_batches = 0
        progress = tqdm(
            self.dataloaders["train"],
            desc=f"{self.fold_name} epoch {epoch}",
            leave=False,
        )

        for batch in progress:
            self.current_step += 1
            lr_scheduler.update(optimizer, self.current_step)
            inputs = batch["inputs"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            if self.cfg.training.mixup:
                inputs, targets, mask = mixup_batch(
                    inputs,
                    labels,
                    mask,
                    self.cfg.training.mixup_alpha,
                )
            else:
                targets = torch.nn.functional.one_hot(
                    labels,
                    num_classes=self.cfg.dataset.num_classes,
                ).to(inputs.dtype)

            optimizer.zero_grad(set_to_none=True)
            results = self.model(inputs, labels=targets, mask=mask)
            results["loss"].backward()
            optimizer.step()
            running_loss += float(results["loss"].detach())
            n_batches += 1

        return {"loss": running_loss / max(n_batches, 1)}

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        probabilities = []
        labels = []

        for batch in dataloader:
            inputs = batch["inputs"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            output = self.model(inputs, mask=mask)
            probabilities.append(output["probabilities"].detach().cpu().numpy())
            labels.append(batch["labels"].numpy())

        y_score = np.concatenate(probabilities)
        y_true = np.concatenate(labels)
        y_pred = (y_score >= 0.5).astype(np.int64)
        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()
        return {
            "auc": float(roc_auc_score(y_true, y_score)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        }

    def test(self) -> dict[str, float]:
        return self.evaluate(self.dataloaders["test"])

    def train(self) -> dict[str, float | int]:
        optimizer = instantiate(
            self.cfg.optimizer,
            params=self.model.parameters(),
        )
        lr_scheduler = LRScheduler(
            base_lr=self.cfg.optimizer.lr,
            target_rate=self.cfg.scheduler.target_rate,
            total_steps=(
                len(self.dataloaders["train"])
                * self.cfg.training.train_epochs
            ),
        )

        best_auc = -float("inf")
        best_epoch = 0
        best_state = None
        epochs_without_improvement = 0
        epoch_results = []

        for epoch in range(1, self.cfg.training.train_epochs + 1):
            train_results = self.train_per_epoch(
                optimizer,
                lr_scheduler,
                epoch,
            )
            valid_results = self.evaluate(self.dataloaders["valid"])
            epoch_results.append(
                {
                    "epoch": epoch,
                    "train_loss": train_results["loss"],
                    **{
                        f"valid_{key}": value
                        for key, value in valid_results.items()
                    },
                }
            )

            score = valid_results["auc"]
            if score >= best_auc:
                best_auc = score
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.cfg.training.patience:
                    break

        self.model.load_state_dict(best_state, strict=True)
        test_results = self.test()

        if self.cfg.save:
            torch.save(
                {
                    "model": best_state,
                    "epoch": best_epoch,
                    "validation_auc": best_auc,
                    "config": OmegaConf.to_container(
                        self.cfg,
                        resolve=True,
                    ),
                },
                self.fold_path / "model.pt",
            )
        with (self.fold_path / "log.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=epoch_results[0])
            writer.writeheader()
            writer.writerows(epoch_results)

        return {
            "best_epoch": best_epoch,
            "validation_auc": best_auc,
            **test_results,
        }


def build_model(cfg: DictConfig) -> nn.Module:
    return DyBraSS(cfg.dataset, cfg.model)


def train_experiment(
    cfg: DictConfig,
    sequences: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, dict[str, list[int]]],
    output_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fold_results = []

    for fold_name in sorted(splits):
        seed_everything(cfg.seed)
        dataloaders = build_loaders(
            sequences,
            labels,
            splits[fold_name],
            batch_size=cfg.training.batch_size,
            num_workers=cfg.training.num_workers,
        )
        trainer = Train(
            cfg=cfg,
            fold_name=fold_name,
            model=build_model(cfg),
            dataloaders=dataloaders,
            fold_path=output_path / fold_name,
            device=device,
        )
        result = trainer.train()
        result["fold"] = fold_name
        fold_results.append(result)

    summary = summarize_results(fold_results)
    report = {
        "dataset": cfg.dataset.name,
        "seed": cfg.seed,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "folds": fold_results,
        "summary": summary,
    }
    _write_results_csv(output_path / "test_result.csv", fold_results, summary)
    return report


def summarize_results(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    summary = {}
    for metric in ("auc", "accuracy", "sensitivity", "specificity"):
        values = np.asarray(
            [result[metric] for result in results],
            dtype=float,
        )
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
        }
    return summary


def _write_results_csv(
    path: Path,
    results: list[dict[str, Any]],
    summary: dict[str, dict[str, float]],
) -> None:
    metrics = ("auc", "accuracy", "sensitivity", "specificity")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["fold", "best_epoch", "validation_auc", *metrics]
        )
        for result in results:
            writer.writerow(
                [
                    result["fold"],
                    result["best_epoch"],
                    f"{result['validation_auc']:.6f}",
                    *[
                        f"{result[metric]:.6f}"
                        for metric in metrics
                    ],
                ]
            )
        writer.writerow(
            [
                "mean",
                "",
                "",
                *[
                    f"{summary[metric]['mean']:.6f}"
                    for metric in metrics
                ],
            ]
        )
        writer.writerow(
            [
                "std",
                "",
                "",
                *[
                    f"{summary[metric]['std']:.6f}"
                    for metric in metrics
                ],
            ]
        )
