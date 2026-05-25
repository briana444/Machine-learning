"""Ensemble policy — average outputs of two MLP models trained with different seeds.

Averaging reduces run-to-run variance: one model might steer slightly wrong at a
corner, the other compensates. Pass both weight paths separated by a comma.

Usage:
    py 03_benchmark.py --tag v12 --weights "nav_v12a.npz,nav_v12b.npz" --module drive2win.ensemble_policy --seeds 42 7
"""
import numpy as np
from drive2win import nn as nn_mod
from drive2win.normalize import sensors_to_input, clip_action


def make_policy(weights_path: str):
    paths = [p.strip() for p in weights_path.split(",")]
    models = [nn_mod.load(p) for p in paths]
    print(f"ensemble: loaded {len(models)} models from {paths}")

    def policy(state):
        x = sensors_to_input(state["sensors"])
        outputs = np.stack([nn_mod.forward(x, w) for w in models])
        return clip_action(outputs.mean(axis=0))

    return policy
