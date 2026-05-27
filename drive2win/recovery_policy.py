"""Recovery-wrapped MLP policy.

Wraps the standard MLP with a stuck-escape heuristic:
  - If speed < 0.3 for STUCK_FRAMES consecutive frames (~1 second at 20 Hz),
    switch to ESCAPE mode: reverse at full throttle and steer hard.
  - Stay in escape mode for ESCAPE_FRAMES frames, then hand back to the net.

This breaks the crash->stuck->timeout death spiral that limits v15 to 3cp
on bad runs without needing any retraining.

Usage:
    py 03_benchmark.py --tag v15 --weights nav_v15.npz --module drive2win.recovery_policy --seeds 42 7
"""
import numpy as np
from drive2win import nn as nn_mod
from drive2win.normalize import sensors_to_input, clip_action

STUCK_FRAMES  = 60   # ~3 s at 20 Hz before triggering escape
ESCAPE_FRAMES = 25   # ~1.25 s of reverse before handing back to net
STUCK_SPEED   = 0.1  # m/s — only trigger when truly stopped, not slowing for corners


def make_policy(weights_path: str):
    w = nn_mod.load(weights_path)

    stuck_count  = 0
    escape_count = 0
    escape_steer = np.float32(0.0)

    def policy(state):
        nonlocal stuck_count, escape_count, escape_steer

        x   = sensors_to_input(state["sensors"])
        spd = float(state["sensors"].get("speed", 1.0))

        # --- escape mode ---
        if escape_count > 0:
            escape_count -= 1
            if escape_count == 0:
                stuck_count = 0  # reset after escape
            return clip_action(np.array([-1.0, escape_steer], dtype=np.float32))

        # --- stuck detection ---
        if spd < STUCK_SPEED:
            stuck_count += 1
        else:
            stuck_count = 0

        if stuck_count >= STUCK_FRAMES:
            # steer opposite to the nearest side wall to spin out
            ray_left  = float(state["sensors"].get("rays", [50]*8)[2])   # +90
            ray_right = float(state["sensors"].get("rays", [50]*8)[6])   # -90
            escape_steer = np.float32(1.0 if ray_left > ray_right else -1.0)
            escape_count = ESCAPE_FRAMES
            stuck_count  = 0
            return clip_action(np.array([-1.0, escape_steer], dtype=np.float32))

        # --- normal net inference ---
        raw = nn_mod.forward(x, w).astype(np.float32)
        return clip_action(raw)

    return policy
