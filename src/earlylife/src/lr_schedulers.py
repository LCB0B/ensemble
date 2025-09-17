import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# @torch.compiler.disable
class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_iters = max_iters
        super().__init__(optimizer)

    # something about types here https://github.com/pytorch/pytorch/issues/120934
    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        lr_factor = 0.5 * (1 + np.cos(np.pi * step / self.max_iters))
        if step <= self.warmup & self.warmup > 0:
            lr_factor *= step / self.warmup
        if step >= self.max_iters:
            lr_factor = 0
        return lr_factor


class TensorCosineWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup: int, max_iters: int):
        """
        A learning rate scheduler with cosine decay and warmup.

        Args:
            optimizer (Optimizer): Optimizer instance to apply the scheduler.
            warmup (int): Number of warmup iterations.
            max_iters (int): Total number of iterations.
        """
        self.warmup = torch.tensor(warmup, dtype=torch.float32)
        self.max_iters = torch.tensor(max_iters, dtype=torch.float32)
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        """
        Calculate the learning rate for the current step.

        Returns:
            list[float]: A list of learning rates for each parameter group.
        """
        lr_factor = self.get_lr_factor(self.last_epoch)
        return [base_lr * lr_factor.item() for base_lr in self.base_lrs]

    def get_lr_factor(self, step: int) -> torch.Tensor:
        """
        Compute the scaling factor for the learning rate at a given step.

        Args:
            step (int): The current iteration.

        Returns:
            torch.Tensor: The scaling factor for the learning rate.
        """
        step = torch.tensor(step, dtype=torch.float32)
        pi = torch.acos(torch.zeros(1)) * 2  # Calculate π using torch
        lr_factor = 0.5 * (1 + torch.cos(pi * step / self.max_iters))

        if step <= self.warmup and self.warmup > 0:
            lr_factor *= step / self.warmup
        if step >= self.max_iters:
            lr_factor = torch.tensor(0.0, dtype=torch.float32)

        return lr_factor
