"""Optional normalized, padding-aware flow loss with gripper-transition weights.

This is a local heuristic, not a reproduction of Sirius human-trust weights.
Indices refer to the padded, normalized model action layout, not robot joints.
"""
import math
import torch
import torch.nn.functional as F


def phase_flow_loss(losses, actions, joint_mask, action_is_pad, *,
                    gripper_indices=(28, 29), weight=2.0, threshold=0.05, radius=2):
    if losses.ndim != 3 or actions.shape != losses.shape:
        raise ValueError("Expected matching [batch, time, dimension] losses/actions")
    if not math.isfinite(weight) or weight < 1 or threshold <= 0 or radius < 0:
        raise ValueError("Invalid phase weighting settings")
    b, t, d = losses.shape
    if action_is_pad is None or action_is_pad.shape != (b, t):
        raise ValueError("Padding-aware training requires action_is_pad [batch,time]")
    if joint_mask is None or joint_mask.shape != losses.shape:
        raise ValueError("Padding-aware training requires joint_mask [batch,time,dim]")
    if not gripper_indices or min(gripper_indices) < 0 or max(gripper_indices) >= d:
        raise ValueError("Gripper indices outside model action layout")
    with torch.no_grad():
        valid_time = ~action_is_pad.bool()
        valid = joint_mask.bool() & valid_time.unsqueeze(-1)
        grip_valid = valid[..., list(gripper_indices)]
        grip = actions.detach()[..., list(gripper_indices)]
        transitions = ((grip[:, 1:] - grip[:, :-1]).abs() >= threshold)
        transitions &= grip_valid[:, 1:] & grip_valid[:, :-1]
        transitions = transitions.any(-1)
        event = torch.zeros((b, t), device=losses.device, dtype=torch.bool)
        event[:, 1:] |= transitions
        event[:, :-1] |= transitions
        if radius:
            event = F.max_pool1d(event.float().unsqueeze(1), 2*radius+1,
                                 stride=1, padding=radius).squeeze(1).bool()
        event &= valid_time
        weights = valid.float() * (1 + (weight-1)*event.float().unsqueeze(-1))
    weighted = losses.float() * weights
    counts = weights.sum((1, 2)).clamp(min=1)
    per_sample = weighted.sum((1, 2)) / counts
    loss = weighted.sum() / weights.sum().clamp(min=1)
    metrics = {
        "phase_loss/padded_fraction": (~valid_time).float().mean(),
        "phase_loss/event_fraction": event.float().sum()/valid_time.sum().clamp(min=1),
    }
    return loss, per_sample, metrics
