#!/usr/bin/env python3
"""
Train VAMPnet on FFT spectrum of raw activations for multiple layers.

This script:
1. Loads multi-layer hidden states from captured data
2. Computes FFT (token-axis) on each layer's activations
3. Trains VAMPnet on the spectral features (magnitude spectrum)
4. Supports layer selection with configurable step size (default: 5)
5. Compares per-layer dynamics and cross-layer spectral patterns

Example:
python train_spectral_vampnet.py --data-dir raw_captures/prompt_000 \
    --layer-step 5 --latent-dim 8 --epochs 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Non-interactive backend for headless use
matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VAMPnet on FFT spectrum of multi-layer activations")
    p.add_argument("--data-dir", type=Path, required=True, 
                   help="Directory containing hidden_layers.npz files (e.g., raw_captures/prompt_XXX)")
    p.add_argument("--data-file", type=Path, default=None,
                   help="Alternative: single NPZ file with multi-layer data (keys: layer_0, layer_5, ...)")
    p.add_argument("--layer-step", type=int, default=5, 
                   help="Layer increment step (e.g., 5 means layers 0, 5, 10, ...)")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Maximum number of layers to use (None = all available)")
    p.add_argument("--freq-bins", type=int, default=32,
                   help="Number of frequency bins to keep from FFT (truncation)")
    p.add_argument("--lag", type=int, default=1, help="Time lag in tokens between pairs")
    p.add_argument("--latent-dim", type=int, default=8, help="Latent dimension for VAMPnet")
    p.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for MLP")
    p.add_argument("--epochs", type=int, default=20, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--device", type=str, default="auto", help="cpu, cuda, or auto")
    p.add_argument("--out-dir", type=Path, default=Path("spectral_vampnet_runs"), help="Output directory")
    p.add_argument("--combine-layers", action="store_true",
                   help="Concatenate all layer spectra into single feature vector (vs. per-layer training)")
    p.add_argument("--max-samples", type=int, default=None, help="Max samples to load (for memory)")
    p.add_argument("--window", type=str, default="hann", choices=["none", "hann", "hamming"],
                   help="Window function for FFT")
    return p.parse_args()


class VAMPNet(nn.Module):
    """VAMPnet with configurable architecture for spectral features."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def vamp2_score(z0: torch.Tensor, z1: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute VAMP-2 score between time-lagged latent representations."""
    B = z0.shape[0]
    z0c = z0 - z0.mean(dim=0, keepdim=True)
    z1c = z1 - z1.mean(dim=0, keepdim=True)
    
    C00 = (z0c.T @ z0c) / (B - 1 + eps)
    C11 = (z1c.T @ z1c) / (B - 1 + eps)
    C01 = (z0c.T @ z1c) / (B - 1 + eps)
    
    # Whitening via eigendecomposition
    evals0, evecs0 = torch.linalg.eigh(C00 + eps * torch.eye(C00.shape[0], device=z0.device))
    evals1, evecs1 = torch.linalg.eigh(C11 + eps * torch.eye(C11.shape[0], device=z1.device))
    inv_sqrt0 = evecs0 @ torch.diag((evals0 + eps).pow(-0.5)) @ evecs0.T
    inv_sqrt1 = evecs1 @ torch.diag((evals1 + eps).pow(-0.5)) @ evecs1.T
    
    Tmat = inv_sqrt0 @ C01 @ inv_sqrt1
    s = torch.linalg.svdvals(Tmat)
    return torch.sum(s ** 2)


def token_fft(activations: np.ndarray, window: str = "hann", n_bins: Optional[int] = None) -> np.ndarray:
    """
    Compute token-axis FFT and return magnitude spectrum.
    
    Args:
        activations: (T, D) - sequence of hidden states
        window: Window function ("none", "hann", "hamming")
        n_bins: Number of frequency bins to keep (None = all)
    
    Returns:
        magnitude_spectrum: (n_bins, D) or (T//2+1, D)
    """
    T, D = activations.shape
    
    # Apply window
    if window == "hann":
        win = np.hanning(T)[:, None]
        activations = activations * win
    elif window == "hamming":
        win = np.hamming(T)[:, None]
        activations = activations * win
    
    # FFT along token axis
    h_hat = np.fft.rfft(activations, axis=0)
    magnitude = np.abs(h_hat)
    
    # Truncate frequency bins if specified
    if n_bins is not None and n_bins < magnitude.shape[0]:
        magnitude = magnitude[:n_bins, :]
    
    return magnitude


def load_multi_layer_data(
    data_dir: Path, 
    layer_step: int = 5,
    max_layers: Optional[int] = None,
    max_samples: Optional[int] = None
) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """
    Load hidden states from multiple layers across all runs in a directory.
    
    Returns:
        layer_data: Dict[layer_idx, array of shape (N, T, D)]
        layer_indices: List of layer indices loaded
    """
    layer_data: Dict[int, List[np.ndarray]] = {}
    
    # Find all run directories or NPZ files
    run_dirs = sorted(data_dir.glob("run_*"))
    if not run_dirs:
        # Try loading directly from the directory
        npz_files = sorted(data_dir.glob("hidden_layers*.npz"))
        run_dirs = [f.parent for f in npz_files] if npz_files else []
    
    if not run_dirs:
        raise ValueError(f"No run directories or hidden_layers.npz files found in {data_dir}")
    
    sample_count = 0
    for run_dir in run_dirs:
        if max_samples is not None and sample_count >= max_samples:
            break
            
        npz_path = run_dir / "hidden_layers.npz"
        if not npz_path.exists():
            npz_files = list(run_dir.glob("hidden_layers*.npz"))
            if npz_files:
                npz_path = npz_files[0]
            else:
                continue
        
        data = np.load(npz_path)
        available_layers = sorted([int(k.split("_")[1]) for k in data.keys() if k.startswith("layer_")])
        
        # Select layers based on step
        selected_layers = [l for l in available_layers if l % layer_step == 0]
        if max_layers is not None:
            selected_layers = selected_layers[:max_layers]
        
        for layer_idx in selected_layers:
            key = f"layer_{layer_idx}"
            if key in data:
                hs = data[key].astype(np.float32)
                # Remove batch dim if present (shape might be (1, T, D))
                if hs.ndim == 3 and hs.shape[0] == 1:
                    hs = hs[0]  # (T, D)
                elif hs.ndim == 2:
                    pass  # Already (T, D)
                else:
                    continue
                
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = []
                layer_data[layer_idx].append(hs)
        
        sample_count += 1
    
    # Stack into arrays
    result = {}
    for layer_idx, arrays in layer_data.items():
        # Align sequence lengths
        min_len = min(a.shape[0] for a in arrays)
        aligned = [a[:min_len] for a in arrays]
        result[layer_idx] = np.stack(aligned, axis=0)  # (N, T, D)
    
    layer_indices = sorted(result.keys())
    return result, layer_indices


def compute_spectral_features(
    layer_data: Dict[int, np.ndarray],
    layer_indices: List[int],
    freq_bins: int = 32,
    window: str = "hann",
    combine_layers: bool = False
) -> Tuple[np.ndarray, Dict]:
    """
    Compute FFT spectrum for each layer and optionally combine.
    
    Returns:
        features: (N, T', D') where D' depends on combine_layers
        metadata: Dict with feature dimensions info
    """
    spectral_per_layer = {}
    
    for layer_idx in layer_indices:
        data = layer_data[layer_idx]  # (N, T, D)
        N, T, D = data.shape
        
        # Compute FFT for each sample
        spectra = []
        for i in range(N):
            mag = token_fft(data[i], window=window, n_bins=freq_bins)  # (freq_bins, D)
            spectra.append(mag)
        
        spectral_per_layer[layer_idx] = np.stack(spectra, axis=0)  # (N, freq_bins, D)
    
    # Get common dimensions
    sample_shape = spectral_per_layer[layer_indices[0]].shape
    N, F, D = sample_shape
    
    metadata = {
        "n_samples": N,
        "freq_bins": F,
        "hidden_dim": D,
        "layer_indices": layer_indices,
        "n_layers": len(layer_indices),
    }
    
    if combine_layers:
        # Concatenate all layers: (N, F, D * n_layers)
        combined = np.concatenate(
            [spectral_per_layer[l] for l in layer_indices], 
            axis=-1
        )
        metadata["feature_dim"] = D * len(layer_indices)
        return combined, metadata
    else:
        # Stack layers as additional dimension: (N, n_layers, F, D)
        stacked = np.stack([spectral_per_layer[l] for l in layer_indices], axis=1)
        # Flatten layer and freq dims: (N, n_layers * F, D)
        reshaped = stacked.reshape(N, len(layer_indices) * F, D)
        metadata["feature_dim"] = D
        metadata["seq_len"] = len(layer_indices) * F
        return reshaped, metadata


def build_spectral_pairs(
    features: np.ndarray, 
    lag: int, 
    axis: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build time-lagged pairs from spectral features.
    
    Args:
        features: (N, T, D) spectral features
        lag: Time lag
        axis: Axis along which to apply lag (1 = sequence/freq axis)
    
    Returns:
        x0, x1: Paired arrays for VAMPnet training
    """
    N, T, D = features.shape
    if T <= lag:
        raise ValueError(f"Sequence length {T} must exceed lag {lag}")
    
    x0_list = []
    x1_list = []
    
    for i in range(N):
        seq = features[i]  # (T, D)
        x0_list.append(seq[:-lag, :])
        x1_list.append(seq[lag:, :])
    
    x0 = np.concatenate(x0_list, axis=0)
    x1 = np.concatenate(x1_list, axis=0)
    
    return x0, x1


def estimate_koopman(z0: np.ndarray, z1: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate Koopman operator eigenvalues from projected features."""
    C00 = z0.T @ z0
    C01 = z0.T @ z1
    reg = eps * np.eye(C00.shape[0])
    K = np.linalg.solve(C00 + reg, C01)
    eigvals, eigvecs = np.linalg.eig(K)
    return eigvals, eigvecs


def train_vampnet(
    x0: np.ndarray,
    x1: np.ndarray,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    verbose: bool = True
) -> Tuple[VAMPNet, List[float]]:
    """Train VAMPnet and return model + loss history."""
    
    # Normalize
    mean = x0.mean(axis=0)
    std = x0.std(axis=0) + 1e-8
    x0_norm = (x0 - mean) / std
    x1_norm = (x1 - mean) / std
    
    ds = TensorDataset(
        torch.tensor(x0_norm, dtype=torch.float32),
        torch.tensor(x1_norm, dtype=torch.float32),
    )
    effective_bs = min(batch_size, max(1, len(ds)))
    dl = DataLoader(ds, batch_size=effective_bs, shuffle=True, drop_last=len(ds) > effective_bs)
    
    model = VAMPNet(input_dim, hidden_dim, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for xb0, xb1 in dl:
            xb0, xb1 = xb0.to(device), xb1.to(device)
            z0, z1 = model(xb0), model(xb1)
            loss = -vamp2_score(z0, z1)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if num_batches == 0:
            break
            
        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
    
    return model, loss_history, mean, std


def plot_spectral_eigenvalues(
    eigvals_dict: Dict[str, np.ndarray],
    out_path: Path,
    title: str = "Spectral VAMPnet Koopman Eigenvalues"
) -> None:
    """Plot eigenvalues from multiple configurations on unit circle."""
    n_configs = len(eigvals_dict)
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 5))
    if n_configs == 1:
        axes = [axes]
    
    for ax, (name, eigvals) in zip(axes, eigvals_dict.items()):
        # Unit circle
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
        
        # Eigenvalues
        colors = np.abs(np.log(np.abs(eigvals) + 1e-9))
        sc = ax.scatter(eigvals.real, eigvals.imag, c=colors, cmap="viridis", 
                       s=60, edgecolor="black", linewidth=0.4)
        
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.set_title(name)
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
    
    plt.suptitle(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_loss_curves(
    loss_dict: Dict[str, List[float]],
    out_path: Path
) -> None:
    """Plot training loss curves for comparison."""
    plt.figure(figsize=(10, 6))
    
    for name, losses in loss_dict.items():
        plt.plot(losses, label=name, linewidth=2)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss (negative VAMP-2 score)")
    plt.title("VAMPnet Training Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectral_power(
    spectral_features: np.ndarray,
    layer_indices: List[int],
    out_path: Path
) -> None:
    """Plot average spectral power across layers."""
    # spectral_features: (N, n_layers * freq_bins, D) or (N, freq_bins, D * n_layers)
    mean_power = np.mean(np.abs(spectral_features) ** 2, axis=(0, 2))
    
    plt.figure(figsize=(12, 5))
    plt.plot(mean_power, linewidth=1.5)
    plt.xlabel("Frequency bin (across layers)")
    plt.ylabel("Mean spectral power")
    plt.title(f"Average Spectral Power Distribution (Layers: {layer_indices})")
    plt.grid(True, alpha=0.3)
    
    # Add layer boundaries
    if len(layer_indices) > 1:
        bins_per_layer = len(mean_power) // len(layer_indices)
        for i, l in enumerate(layer_indices[1:], 1):
            plt.axvline(i * bins_per_layer, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load multi-layer data
    print(f"Loading data from {args.data_dir}...")
    
    if args.data_file is not None:
        # Load from single file
        data = np.load(args.data_file)
        layer_data = {}
        for key in data.keys():
            if key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                if layer_idx % args.layer_step == 0:
                    hs = data[key].astype(np.float32)
                    if hs.ndim == 2:
                        hs = hs[None, ...]  # Add batch dim
                    layer_data[layer_idx] = hs
        layer_indices = sorted(layer_data.keys())
        if args.max_layers:
            layer_indices = layer_indices[:args.max_layers]
            layer_data = {k: v for k, v in layer_data.items() if k in layer_indices}
    else:
        layer_data, layer_indices = load_multi_layer_data(
            args.data_dir,
            layer_step=args.layer_step,
            max_layers=args.max_layers,
            max_samples=args.max_samples
        )
    
    print(f"Loaded {len(layer_indices)} layers: {layer_indices}")
    for l in layer_indices[:3]:
        print(f"  Layer {l}: {layer_data[l].shape}")
    
    # Compute spectral features
    print(f"\nComputing FFT spectrum (bins={args.freq_bins}, window={args.window})...")
    spectral_features, metadata = compute_spectral_features(
        layer_data, layer_indices,
        freq_bins=args.freq_bins,
        window=args.window,
        combine_layers=args.combine_layers
    )
    print(f"Spectral features shape: {spectral_features.shape}")
    print(f"Metadata: {metadata}")
    
    # Plot spectral power distribution
    plot_spectral_power(spectral_features, layer_indices, out_dir / "spectral_power.png")
    
    # Build time-lagged pairs
    print(f"\nBuilding time-lagged pairs (lag={args.lag})...")
    x0, x1 = build_spectral_pairs(spectral_features, args.lag)
    print(f"Training pairs: {x0.shape[0]}")
    
    # Train VAMPnet
    print(f"\nTraining VAMPnet (latent_dim={args.latent_dim}, epochs={args.epochs})...")
    input_dim = x0.shape[1]
    
    model, loss_history, mean, std = train_vampnet(
        x0, x1,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    
    # Compute Koopman eigenvalues
    print("\nEstimating Koopman eigenvalues...")
    model.eval()
    with torch.no_grad():
        x0_norm = (x0 - mean) / std
        x1_norm = (x1 - mean) / std
        z0 = model(torch.tensor(x0_norm, dtype=torch.float32, device=device)).cpu().numpy()
        z1 = model(torch.tensor(x1_norm, dtype=torch.float32, device=device)).cpu().numpy()
    
    eigvals, eigvecs = estimate_koopman(z0, z1)
    
    # Find slow modes
    mags = np.abs(eigvals)
    slow_idx = int(np.argmin(np.abs(1.0 - mags)))
    print(f"Slowest mode: λ={eigvals[slow_idx]:.4f} (|λ|={mags[slow_idx]:.4f})")
    
    # Save results
    print(f"\nSaving results to {out_dir}...")
    
    np.savez(
        out_dir / "spectral_vampnet_outputs.npz",
        eigvals=eigvals,
        eigvecs=eigvecs,
        mean=mean,
        std=std,
        layer_indices=np.array(layer_indices),
        spectral_features=spectral_features,
        loss_history=np.array(loss_history),
    )
    
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "metadata": metadata,
    }, out_dir / "spectral_vampnet_model.pt")
    
    # Save metadata as JSON
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            **metadata,
            "layer_indices": layer_indices,
            "freq_bins": args.freq_bins,
            "lag": args.lag,
            "latent_dim": args.latent_dim,
            "slow_mode_eigval": complex(eigvals[slow_idx]),
            "slow_mode_magnitude": float(mags[slow_idx]),
        }, f, indent=2, default=str)
    
    # Plots
    plot_spectral_eigenvalues(
        {"Spectral VAMPnet": eigvals},
        out_dir / "spectral_vampnet_eigs.png"
    )
    plot_loss_curves({"VAMPnet": loss_history}, out_dir / "training_loss.png")
    
    print(f"\nDone! Results saved to {out_dir}")
    print(f"  - spectral_vampnet_outputs.npz")
    print(f"  - spectral_vampnet_model.pt")
    print(f"  - spectral_vampnet_eigs.png")
    print(f"  - spectral_power.png")
    print(f"  - training_loss.png")


if __name__ == "__main__":
    main()

