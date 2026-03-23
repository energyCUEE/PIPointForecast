import torch
from torch.optim.lr_scheduler import SequentialLR


class WarmupLambda:
    def __init__(self, warmup_steps: int):
        self.warmup_steps = warmup_steps

    def __call__(self, current_step: int) -> float:
        if current_step >= self.warmup_steps:
            return 1
        return current_step / max(1, self.warmup_steps) + 1e-8


def get_lr_scheduler_with_warmup(
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    if warmup_steps <= 0:
        return lr_scheduler

    lr_lambda_warmup = WarmupLambda(warmup_steps)

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_warmup)

    combined_scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps]
    )

    return combined_scheduler
