"""Small training guards for sequential microbatch accumulation."""
import math


def normalized_microbatch_loss(loss, count):
    if not isinstance(count, int) or count < 1:
        raise ValueError("microbatch count must be positive")
    return loss / count


def check_microbatch_count(actual, expected):
    if actual != expected or actual < 1:
        raise RuntimeError(f"Expected {expected} microbatches, got {actual}")


def finite_grad_norm(value):
    if hasattr(value, "full_tensor"):
        value = value.full_tensor()
    value = float(value)
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite gradient norm; refusing optimizer update")
    return value
