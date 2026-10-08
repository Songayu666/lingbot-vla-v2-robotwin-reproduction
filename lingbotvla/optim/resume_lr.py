"""Explicit LR scaling after optimizer and scheduler restore."""
import math

def scale_resumed_lr(optimizer, scheduler, factor):
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("resume LR factor must be finite and in (0, 1]")
    scheduler.base_lrs = [lr * factor for lr in scheduler.base_lrs]
    scheduler._last_lr = [lr * factor for lr in scheduler.get_last_lr()]
    for group in optimizer.param_groups:
        group["lr"] *= factor
        if "initial_lr" in group:
            group["initial_lr"] *= factor
