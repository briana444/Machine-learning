"""Train a K-frame temporal MLP using PyTorch.

Stacks the last K=4 sensor readings into a 48-dim input so the model can
detect being stuck (same position across frames) and steer out.

Run:
    py 02_train_temporal.py --data data_v2.npz --extra-data data_v3.npz data_v4.npz data_v6a.npz data_v6b.npz --tag v11 --epochs 500 --min-speed 1.0
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn

from drive2win.normalize import normalize_states
from drive2win.temporal_policy import TemporalMLP, K


def make_windowed(states_norm: np.ndarray, actions: np.ndarray, k: int = K):
    """Stack k consecutive frames per sample (pad start with first frame)."""
    pad = np.repeat(states_norm[:1], k - 1, axis=0)
    s = np.concatenate([pad, states_norm], axis=0)
    windows = np.stack([s[i:i + k].reshape(-1) for i in range(len(states_norm))])
    return windows.astype(np.float32), actions.astype(np.float32)


def load_datasets(files: list[str], min_speed: float = 0.0):
    all_X, all_Y = [], []
    total, dropped = 0, 0
    for f in files:
        d = np.load(f, allow_pickle=False)
        sr, ac = d["states"], d["actions"]
        total += len(sr)
        if min_speed > 0:
            mask = sr[:, 0] >= min_speed
            dropped += (~mask).sum()
            sr, ac = sr[mask], ac[mask]
        X, Y = make_windowed(normalize_states(sr), ac)
        all_X.append(X)
        all_Y.append(Y)
        print(f"  {f}: {len(X)} samples")
    if dropped:
        print(f"filtered {dropped}/{total} stuck samples (speed < {min_speed} m/s)")
    return np.concatenate(all_X), np.concatenate(all_Y)


def train(X: np.ndarray, Y: np.ndarray, epochs: int = 500, lr: float = 1e-3,
          batch_size: int = 64, val_frac: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_val = max(1, int(len(X) * val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    Xtr = torch.tensor(X[tr_idx])
    Ytr = torch.tensor(Y[tr_idx])
    Xva = torch.tensor(X[val_idx])
    Yva = torch.tensor(Y[val_idx])

    model = TemporalMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        idx = torch.randperm(len(Xtr))
        ep_loss, n_b = 0.0, 0
        for i in range(0, len(Xtr), batch_size):
            b = idx[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), Ytr[b])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1

        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva), Yva).item()
        if v < best_val:
            best_val = v
            best_state = {k: w.clone() for k, w in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:3d}  train={ep_loss / max(1, n_b):.4f}  val={v:.4f}  best={best_val:.4f}")

    model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--extra-data", nargs="*", default=[])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-speed", type=float, default=0.0)
    args = ap.parse_args()

    files = [args.data] + (args.extra_data or [])
    print(f"loading {len(files)} dataset(s)...")
    X, Y = load_datasets(files, min_speed=args.min_speed)
    print(f"total: {len(X)} windows  input_dim={X.shape[1]}")

    model = train(X, Y, epochs=args.epochs, lr=args.lr)

    path = f"nav_{args.tag}.pt"
    torch.save(model.state_dict(), path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
