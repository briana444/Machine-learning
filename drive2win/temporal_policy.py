"""Temporal MLP policy — K-frame sliding window inference.

The model sees the last K sensor readings stacked into one input vector.
This lets it detect being stuck (repeated identical readings) and react.

Usage:
    py 03_benchmark.py --tag v11 --weights nav_v11.pt --module drive2win.temporal_policy --seeds 42 7
"""
from collections import deque
import numpy as np
import torch
import torch.nn as nn

from .normalize import sensors_to_input, clip_action, N_FEATURES, N_ACTIONS

K = 4  # frames to stack — must match 02_train_temporal.py


class TemporalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K * N_FEATURES, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 64),             nn.LeakyReLU(0.1),
            nn.Linear(64, 32),              nn.LeakyReLU(0.1),
            nn.Linear(32, N_ACTIONS),       nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_policy(weights_path: str):
    model = TemporalMLP()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    buf = deque(maxlen=K)

    def policy(state):
        x = sensors_to_input(state["sensors"])
        # warm-start: fill buffer with copies of first frame
        while len(buf) < K - 1:
            buf.append(x.copy())
        buf.append(x)
        window = np.concatenate(list(buf)).astype(np.float32)
        with torch.no_grad():
            y = model(torch.from_numpy(window).unsqueeze(0))[0].numpy()
        return clip_action(y)

    return policy
