import math

import torch


class LRScheduler:
    def __init__(self, base_lr: float, target_rate: float, total_steps: int) -> None:
        self.base_lr = base_lr
        self.target_lr = base_lr * target_rate
        self.total_steps = max(total_steps, 1)

    def update(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        progress = min(step / self.total_steps, 1.0)
        cosine = math.cos(math.pi * progress)
        learning_rate = self.target_lr + (
            self.base_lr - self.target_lr
        ) * (1 + cosine) / 2
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate
