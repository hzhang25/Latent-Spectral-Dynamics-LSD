#!/usr/bin/env python3
"""
Train a simple VAMPnet to learn slow modes from saved hidden-state trajectories.

Assumptions:
- Dataset NPZ contains `hidden_states` shaped (N, T, D) where N is #samples,
  T is sequence length, and D is feature dim.
- We form time-lagged pairs (x_t, x_{t+lag}) and optimize the VAMP-2 score.
- After training, we estimate Koopman eigenvalues from the projected covariances
  and plot the leading modes in 2D for bifurcation visualization.

Example:
python train_vampnet.py --data outputs/activations_and_dmd.npz \
    --lag 1 --latent-dim 4 --epochs 5 --batch-size 512 \
    --out-dir vampnet_runs --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Non-interactive backend for headless use
matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VAMPnet on saved hidden states")
    p.add_argument("--data", type=Path, required=True, help="NPZ file with hidden_states (N,T,D)")
    p.add_argument("--lag", type=int, default=1, help="Time lag in tokens between pairs")
    p.add_argument("--latent-dim", type=int, default=4, help="Latent dimension for VAMPnet outputs")
    p.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for MLP")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=512, help="Batch size for pairs")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda; if set to 'auto', will pick cuda if available else cpu",
    )
    p.add_argument("--out-dir", type=Path, default=Path("vampnet_runs"), help="Output directory")
    p.add_argument("--max-pairs", type=int, default=None, help="Optional cap on number of pairs for speed")
    p.add_argument("--plot-trajectories", action="store_true", help="Plot first two latent dims trajectory")
    return p.parse_args()


class VAMPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def vamp2_score(z0: torch.Tensor, z1: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the VAMP-2 score between two batches of latent features.
    z0, z1: (B, k)
    """
    B = z0.shape[0]
    z0c = z0 - z0.mean(dim=0, keepdim=True)
    z1c = z1 - z1.mean(dim=0, keepdim=True)

    C00 = (z0c.T @ z0c) / (B - 1 + eps)
    C11 = (z1c.T @ z1c) / (B - 1 + eps)
    C01 = (z0c.T @ z1c) / (B - 1 + eps)

    # Whitening
    evals0, evecs0 = torch.linalg.eigh(C00 + eps * torch.eye(C00.shape[0], device=z0.device))
    evals1, evecs1 = torch.linalg.eigh(C11 + eps * torch.eye(C11.shape[0], device=z1.device))
    inv_sqrt0 = evecs0 @ torch.diag((evals0 + eps).pow(-0.5)) @ evecs0.T
    inv_sqrt1 = evecs1 @ torch.diag((evals1 + eps).pow(-0.5)) @ evecs1.T

    Tmat = inv_sqrt0 @ C01 @ inv_sqrt1
    s = torch.linalg.svdvals(Tmat)
    return torch.sum(s ** 2)


def build_pairs(hidden_states: np.ndarray, lag: int, max_pairs: int | None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build time-lagged pairs (x_t, x_{t+lag}) from hidden state sequences.
    hidden_states: (N, T, D)
    """
    N, T, D = hidden_states.shape
    if T <= lag:
        raise ValueError(f"Sequence length {T} must exceed lag {lag}")
    x0_list = []
    x1_list = []
    for seq in hidden_states:
        x0_list.append(seq[:-lag, :])
        x1_list.append(seq[lag:, :])
    x0 = np.concatenate(x0_list, axis=0)
    x1 = np.concatenate(x1_list, axis=0)
    if max_pairs is not None and max_pairs < x0.shape[0]:
        idx = np.random.choice(x0.shape[0], size=max_pairs, replace=False)
        x0 = x0[idx]
        x1 = x1[idx]
    return x0, x1


def estimate_koopman(z0: np.ndarray, z1: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Koopman estimate K = (Z0^T Z0)^-1 Z0^T Z1, eigenvalues approximate slow modes.
    """
    C00 = z0.T @ z0
    C01 = z0.T @ z1
    reg = eps * np.eye(C00.shape[0])
    K = np.linalg.solve(C00 + reg, C01)
    eigvals, eigvecs = np.linalg.eig(K)
    return eigvals, eigvecs


def plot_eigs(eigvals: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    colors = np.abs(np.log(np.abs(eigvals) + 1e-9))
    plt.scatter(eigvals.real, eigvals.imag, c=colors, cmap="viridis", s=60, edgecolor="black", linewidth=0.4)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel("Re(λ)")
    plt.ylabel("Im(λ)")
    plt.title("VAMPnet Koopman eigenvalues")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def plot_latent_trajectory(z: np.ndarray, out_path: Path) -> None:
    if z.shape[1] < 2:
        return
    plt.figure(figsize=(6, 5))
    t = np.arange(z.shape[0])
    plt.scatter(z[:, 0], z[:, 1], c=t, cmap="plasma", s=8)
    plt.colorbar(label="time index")
    plt.xlabel("latent dim 1")
    plt.ylabel("latent dim 2")
    plt.title("Latent trajectory (first sample)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=170)
    plt.close()


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    data = np.load(args.data)
    if "hidden_states" not in data:
        raise KeyError("NPZ must contain 'hidden_states'")
    hidden_states = data["hidden_states"]  # (N, T, D)
    N, T, D = hidden_states.shape
    print(f"Loaded hidden_states: N={N}, T={T}, D={D}")

    x0, x1 = build_pairs(hidden_states, args.lag, args.max_pairs)
    mean = x0.mean(axis=0)
    std = x0.std(axis=0) + 1e-8
    x0 = (x0 - mean) / std
    x1 = (x1 - mean) / std

    ds = TensorDataset(
        torch.tensor(x0, dtype=torch.float32),
        torch.tensor(x1, dtype=torch.float32),
    )
    # Ensure at least one batch even for tiny datasets
    effective_bs = min(args.batch_size, max(1, len(ds)))
    dl = DataLoader(ds, batch_size=effective_bs, shuffle=True, drop_last=False)

    model = VAMPNet(D, args.hidden_dim, args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        num_batches = 0
        for xb0, xb1 in dl:
            xb0 = xb0.to(device)
            xb1 = xb1.to(device)
            z0 = model(xb0)
            z1 = model(xb1)
            loss = -vamp2_score(z0, z1)  # maximize VAMP-2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            num_batches += 1
        if num_batches == 0:
            print(f"Epoch {epoch+1}: no batches (dataset too small). Increase data or reduce lag.")
            break
        avg = total / num_batches
        print(f"Epoch {epoch+1}/{args.epochs} loss={avg:.4f} (bs={effective_bs}, batches={num_batches})")

    # Evaluation: compute Koopman eigenvalues on full (normalized) data
    model.eval()
    with torch.no_grad():
        z0_full = model(torch.tensor(x0, dtype=torch.float32, device=device)).cpu().numpy()
        z1_full = model(torch.tensor(x1, dtype=torch.float32, device=device)).cpu().numpy()
    eigvals, eigvecs = estimate_koopman(z0_full, z1_full)

    np.savez(
        out_dir / "vampnet_outputs.npz",
        eigvals=eigvals,
        eigvecs=eigvecs,
        mean=mean,
        std=std,
    )
    torch.save({"model_state": model.state_dict(), "args": vars(args)}, out_dir / "vampnet_model.pt")
    plot_eigs(eigvals, out_dir / "vampnet_eigs.png")

    if args.plot_trajectories:
        # Use first sequence from dataset to visualize latent trajectory
        seq0 = (hidden_states[0] - mean) / std
        with torch.no_grad():
            z_seq = model(torch.tensor(seq0, dtype=torch.float32, device=device)).cpu().numpy()
        plot_latent_trajectory(z_seq, out_dir / "latent_traj_seq0.png")

    # Slow mode: eigenvalue with magnitude closest to 1
    mags = np.abs(eigvals)
    slow_idx = int(np.argmin(np.abs(1.0 - mags)))
    print(f"Slow mode λ={eigvals[slow_idx]} (|λ|={mags[slow_idx]:.3f}) saved to {out_dir}")


if __name__ == "__main__":
    main()
