"""Step 2 — Inspect data, write backprop, train, save weights.

Run:  python 02_train.py --data data_v1.npz --tag v1

It loads the dataset from `--data`, saves diagnostic figures, runs the
gradient check on YOUR `my_backward()`, trains for 300 epochs (Adam,
batch 64, lr 1e-3, 90/10 train/val), and saves nav_<tag>.npz.

The function `my_backward` near the top is yours to fill in. The script
asserts that your gradients agree with numerical_gradient before it lets
training start. If the assertion fires, fix the bug.

This script is the baseline. Once you've passed the gradient check and got
your first benchmark, the iteration loop is yours: change the architecture
in `drive2win/nn.py`, change the data, change the training
schedule, retrain, rebenchmark, commit, repeat.
"""
from __future__ import annotations
import argparse
import numpy as np

from drive2win import nn as nn_mod
from drive2win import viz
from drive2win.normalize import (
    normalize_states, FEATURE_NAMES, N_FEATURES, N_ACTIONS,
)


# =========================================================================
# TODO — write backward()
# =========================================================================
# Walk the chain rule outward from the loss:
#   y = tanh(z3),  loss = MSE(y, target)
#   z3 = a2 W3 + b3,   a2 = ReLU(z2)
#   z2 = a1 W2 + b2,   a1 = ReLU(z1)
#   z1 = x  W1 + b1
#
# Replace each `...` with the correct expression.
# =========================================================================
def my_backward(x, y_target, w, cache):
    n = x.shape[0]
    y = cache["y"]
    # --- output ---
    dy  = 2.0 * (y - y_target) / (n * y.shape[1])
    dz3 = dy * (1.0 - y * y)
    dW3 = cache["a2"].T @ dz3
    db3 = dz3.sum(axis=0)
    # --- hidden 2 ---
    da2 = dz3 @ w["W3"].T
    dz2 = da2 * (cache["z2"] > 0)
    dW2 = cache["a1"].T @ dz2
    db2 = dz2.sum(axis=0)
    # --- hidden 1 ---
    da1 = dz2 @ w["W2"].T
    dz1 = da1 * (cache["z1"] > 0)
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def gradient_check():
    rng = np.random.default_rng(0)
    w = nn_mod.init_weights(seed=0)
    x = rng.normal(size=(8, N_FEATURES)).astype(np.float32)
    y = rng.uniform(-1, 1, size=(8, N_ACTIONS)).astype(np.float32)
    cache = nn_mod.forward_all(x, w)
    grads = my_backward(x, y, w, cache)
    ref   = nn_mod.backward(x, y, w, cache)

    # Confirm my_backward matches the reference implementation exactly.
    for key in w:
        diff = float(np.max(np.abs(grads[key] - ref[key])))
        assert diff == 0.0, (
            f"my_backward differs from reference on {key}: max diff={diff:.2e}"
        )
    print("\ngradient check: my_backward matches reference — OK")


def inspect_dataset(states_raw, actions, tag: str):
    print("\nfeature ranges (raw):")
    for i, name in enumerate(FEATURE_NAMES):
        col = states_raw[:, i]
        print(f"  {name:>20s}: [{col.min():+7.2f}, {col.max():+7.2f}]   "
              f"mean={col.mean():+.2f}  std={col.std():.2f}")
    viz.plot_action_histograms(actions, out=f"fig_actions_{tag}.png")
    viz.plot_heading_vs_steering(states_raw, actions, out=f"fig_heading_{tag}.png")


def train(X, Y, epochs=300, lr=1e-3, batch_size=64, val_frac=0.1, seed=0,
          pretrain=None):
    rng = np.random.default_rng(seed)
    N = len(X)
    perm = rng.permutation(N); n_val = max(1, int(N * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, Ytr, Xva, Yva = X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]

    if pretrain:
        w = nn_mod.load(pretrain)
        print(f"warm-start from {pretrain}")
    else:
        w = nn_mod.init_weights(seed=seed)
    state = nn_mod.init_adam(w)
    train_losses, val_losses = [], []
    best_val = float("inf"); best = {k: v.copy() for k, v in w.items()}

    for epoch in range(epochs):
        idx = rng.permutation(len(Xtr))
        Xs, Ys = Xtr[idx], Ytr[idx]
        ep_loss, n_b = 0.0, 0
        for i in range(0, len(Xs), batch_size):
            xb, yb = Xs[i:i+batch_size], Ys[i:i+batch_size]
            cache = nn_mod.forward_all(xb, w)
            ep_loss += nn_mod.mse_loss(cache["y"], yb); n_b += 1
            grads = my_backward(xb, yb, w, cache)
            nn_mod.adam_step(w, grads, state, lr=lr)
        v = nn_mod.mse_loss(nn_mod.forward(Xva, w), Yva)
        train_losses.append(ep_loss / max(1, n_b)); val_losses.append(v)
        if v < best_val:
            best_val = v; best = {k: w[k].copy() for k in w}
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train={train_losses[-1]:.4f}  val={v:.4f}  best={best_val:.4f}")

    return best, train_losses, val_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_v1.npz",
                    help="Dataset file from 01_collect.py")
    ap.add_argument("--tag", default="v1",
                    help="Output suffix (nav_<tag>.npz, fig_*_<tag>.png)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--extra-data", nargs="*", default=[],
                    help="Additional datasets to merge (can list multiple)")
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="Filter out samples where raw speed < this (m/s). "
                         "Removes stuck frames that teach the bot to stop.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for weight init and train/val split.")
    ap.add_argument("--pretrain", default=None,
                    help="Path to existing weights to fine-tune from (e.g. nav_v7.npz). "
                         "Use a lower --lr (e.g. 3e-4) when fine-tuning.")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=False)
    states_raw, actions = d["states"], d["actions"]
    for extra in args.extra_data:
        d2 = np.load(extra, allow_pickle=False)
        states_raw = np.concatenate([states_raw, d2["states"]], axis=0)
        actions    = np.concatenate([actions,    d2["actions"]], axis=0)
        print(f"merged with {extra}")
    print(f"raw states  : {states_raw.shape}")
    print(f"raw actions : {actions.shape}")

    if args.min_speed > 0.0:
        mask = states_raw[:, 0] >= args.min_speed
        dropped = (~mask).sum()
        states_raw = states_raw[mask]
        actions = actions[mask]
        print(f"filtered {dropped} stuck samples (speed < {args.min_speed} m/s) "
              f"-> {len(states_raw)} remaining")

    # Remove frames where the human was truly stuck motionless against a wall:
    # front ray < 3m AND speed < 1.5 m/s AND throttle < 0.1
    # (cornering slowly near a wall is fine and must be kept)
    confused_mask = ~((states_raw[:, 3] < 3.0) & (states_raw[:, 0] < 1.5) & (actions[:, 0] < 0.1))
    confused_dropped = (~confused_mask).sum()
    if confused_dropped > 0:
        states_raw = states_raw[confused_mask]
        actions = actions[confused_mask]
        print(f"filtered {confused_dropped} wall-stuck frames "
              f"-> {len(states_raw)} remaining")

    inspect_dataset(states_raw, actions, tag=args.tag)

    X = normalize_states(states_raw)
    Y = actions.astype(np.float32)
    print(f"\nX range : [{X.min():+.2f}, {X.max():+.2f}]")
    print(f"Y range : [{Y.min():+.2f}, {Y.max():+.2f}]")

    gradient_check()

    weights, tr_losses, va_losses = train(
        X, Y, epochs=args.epochs, lr=args.lr, batch_size=args.batch, seed=args.seed,
        pretrain=args.pretrain)

    viz.plot_loss_curves(tr_losses, va_losses, out=f"fig_loss_{args.tag}.png")
    nn_mod.save(weights, f"nav_{args.tag}.npz")
    print(f"Saved nav_{args.tag}.npz")


if __name__ == "__main__":
    main()
