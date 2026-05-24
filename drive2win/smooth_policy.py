"""Action-smoothed MLP policy adapter.

Wraps the standard MLP with an exponential moving average on the outputs.
This prevents the steering from oscillating wildly (spinning) when the
network's frame-to-frame predictions jump in opposite directions.

Usage:
    py 03_benchmark.py --tag v8 --weights nav_v7.npz --module drive2win.smooth_policy --seeds 42 7
"""
import numpy as np
from drive2win import nn as nn_mod
from drive2win.normalize import sensors_to_input, clip_action

ALPHA_STEERING = 0.5  # steering smoothing (0=frozen, 1=no smoothing)
# throttle is NOT smoothed — damping it caused random stops


def make_policy(weights_path: str):
    w = nn_mod.load(weights_path)
    prev_steer = np.float32(0.0)

    def policy(state):
        nonlocal prev_steer
        x = sensors_to_input(state["sensors"])
        raw = nn_mod.forward(x, w).astype(np.float32)
        throttle = raw[0]
        steer = ALPHA_STEERING * raw[1] + (1.0 - ALPHA_STEERING) * prev_steer
        prev_steer = steer
        return clip_action(np.array([throttle, steer], dtype=np.float32))

    return policy
