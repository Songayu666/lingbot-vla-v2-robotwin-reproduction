"""Event-triggered replanning for 14-dimensional RoboTwin joint actions."""
import numpy as np

def execution_length(actions, state, threshold=0.15, minimum=2):
    actions = np.asarray(actions)
    state = np.asarray(state)
    if actions.ndim != 2 or actions.shape[1] != 14 or state.shape != (14,):
        raise ValueError("Expected actions [T,14] and state [14]")
    if not np.isfinite(actions).all() or not np.isfinite(state).all():
        raise ValueError("Non-finite action/state")
    if threshold <= 0 or minimum < 1:
        raise ValueError("Invalid horizon settings")
    # Stop after the first substantial predicted gripper change, then observe.
    grippers = actions[:, [6, 13]]
    previous = np.concatenate([state[None, [6, 13]], grippers[:-1]], axis=0)
    events = np.flatnonzero(np.max(np.abs(grippers - previous), axis=1) >= threshold)
    return min(len(actions), max(minimum, int(events[0]) + 1)) if len(events) else len(actions)
