#!/usr/bin/env python3
"""
Time-lagged Independent Component Analysis (TICA) for spectral dynamics.

TICA is the linear limit of VAMPnet - it finds slow modes via generalized
eigenvalue decomposition without neural network training.

This script supports two modes:
1. **Spectral TICA**: Apply TICA to FFT-transformed activations (default)
2. **Raw TICA**: Apply TICA directly to raw hidden state activations

Example:
# Spectral TICA (on FFT features)
python train_tica.py --data-dir raw_captures/prompt_000 \
    --layer-step 5 --n-components 8 --mode spectral

# Raw TICA (on raw activations)
python train_tica.py --data-dir raw_captures/prompt_000 \
    --layer-step 5 --n-components 8 --mode raw

# Compare both modes
python train_tica.py --data-dir raw_captures/prompt_000 \
    --layer-step 5 --n-components 8 --mode both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

# Non-interactive backend
matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TICA analysis on multi-layer activations (raw or spectral)")
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Directory containing hidden_layers.npz files")
    p.add_argument("--data-file", type=Path, default=None,
                   help="Alternative: single NPZ file with multi-layer data")
    p.add_argument("--mode", type=str, default="spectral",
                   choices=["spectral", "raw", "both"],
                   help="TICA mode: 'spectral' (FFT features), 'raw' (raw activations), or 'both'")
    p.add_argument("--layer-step", type=int, default=5,
                   help="Layer increment step (e.g., 5 means layers 0, 5, 10, ...)")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Maximum number of layers to use")
    p.add_argument("--freq-bins", type=int, default=32,
                   help="Number of frequency bins for FFT (spectral mode only)")
    p.add_argument("--lag", type=int, default=1,
                   help="Time lag for TICA")
    p.add_argument("--n-components", type=int, default=8,
                   help="Number of TICA components to extract")
    p.add_argument("--out-dir", type=Path, default=Path("tica_runs"),
                   help="Output directory")
    p.add_argument("--window", type=str, default="hann",
                   choices=["none", "hann", "hamming"],
                   help="Window function for FFT (spectral mode only)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Max samples to load")
    p.add_argument("--regularization", type=float, default=1e-6,
                   help="Regularization for matrix inversion")
    p.add_argument("--compare-vampnet", type=Path, default=None,
                   help="Path to VAMPnet results NPZ for comparison")
    return p.parse_args()


class TICA:
    """
    Time-lagged Independent Component Analysis.
    
    Finds slow modes by solving the generalized eigenvalue problem:
        C_01 @ v = λ @ C_00 @ v
    where C_00 is the instantaneous covariance and C_01 is the time-lagged covariance.
    """
    
    def __init__(self, n_components: int = 4, lag: int = 1, regularization: float = 1e-6):
        self.n_components = n_components
        self.lag = lag
        self.regularization = regularization
        
        # Fitted parameters
        self.mean_ = None
        self.eigvals_ = None
        self.eigvecs_ = None
        self.timescales_ = None
        self.C00_ = None
        self.C01_ = None
        
    def fit(self, X: np.ndarray) -> "TICA":
        """
        Fit TICA model on time series data.
        
        Args:
            X: (N, T, D) array of trajectories or (T, D) single trajectory
        """
        if X.ndim == 2:
            X = X[None, ...]  # Add batch dim
        
        N, T, D = X.shape
        
        if T <= self.lag:
            raise ValueError(f"Sequence length {T} must exceed lag {self.lag}")
        
        # Build time-lagged pairs
        x0_list, x1_list = [], []
        for seq in X:
            x0_list.append(seq[:-self.lag])
            x1_list.append(seq[self.lag:])
        
        x0 = np.concatenate(x0_list, axis=0)  # (M, D)
        x1 = np.concatenate(x1_list, axis=0)  # (M, D)
        
        # Center data
        self.mean_ = x0.mean(axis=0)
        x0c = x0 - self.mean_
        x1c = x1 - self.mean_
        
        M = x0c.shape[0]
        
        # Covariance matrices
        self.C00_ = (x0c.T @ x0c) / (M - 1)
        self.C01_ = (x0c.T @ x1c) / (M - 1)
        C11 = (x1c.T @ x1c) / (M - 1)
        
        # Symmetrize time-lagged covariance for stability
        C01_sym = 0.5 * (self.C01_ + self.C01_.T)
        
        # Add regularization
        reg = self.regularization * np.eye(D)
        
        # Solve generalized eigenvalue problem: C01_sym @ v = λ @ C00 @ v
        try:
            eigvals, eigvecs = eigh(C01_sym, self.C00_ + reg)
        except np.linalg.LinAlgError:
            # Fallback: standard eigenvalue problem on whitened data
            U, s, Vt = np.linalg.svd(self.C00_ + reg, full_matrices=False)
            s_inv = np.where(s > 1e-10, 1.0 / np.sqrt(s), 0)
            C00_inv_sqrt = U @ np.diag(s_inv) @ Vt
            C_whitened = C00_inv_sqrt @ C01_sym @ C00_inv_sqrt.T
            eigvals, eigvecs_whitened = np.linalg.eigh(C_whitened)
            eigvecs = C00_inv_sqrt.T @ eigvecs_whitened
        
        # Sort by eigenvalue magnitude (descending = slowest modes first)
        idx = np.argsort(np.abs(eigvals))[::-1]
        self.eigvals_ = eigvals[idx][:self.n_components]
        self.eigvecs_ = eigvecs[:, idx][:, :self.n_components]
        
        # Compute implied timescales: τ = -lag / ln(|λ|)
        self.timescales_ = np.where(
            np.abs(self.eigvals_) > 1e-10,
            -self.lag / np.log(np.abs(self.eigvals_) + 1e-10),
            np.inf
        )
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto TICA components.
        
        Args:
            X: (N, T, D) or (T, D) or (M, D) data
        
        Returns:
            Projected data with shape (..., n_components)
        """
        original_shape = X.shape
        if X.ndim == 3:
            N, T, D = X.shape
            X_flat = X.reshape(-1, D)
        else:
            X_flat = X
        
        X_centered = X_flat - self.mean_
        Z = X_centered @ self.eigvecs_
        
        if len(original_shape) == 3:
            return Z.reshape(N, T, self.n_components)
        return Z
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def score(self, X: np.ndarray = None) -> float:
        """
        Compute VAMP-2 score (sum of squared eigenvalues).
        This is equivalent to what VAMPnet optimizes.
        """
        return float(np.sum(self.eigvals_[:self.n_components] ** 2))
    
    def kinetic_variance(self) -> float:
        """Compute kinetic variance (trace of C01)."""
        if self.C01_ is None:
            return 0.0
        return float(np.trace(self.C01_))


def token_fft(activations: np.ndarray, window: str = "hann", n_bins: Optional[int] = None) -> np.ndarray:
    """Compute token-axis FFT magnitude spectrum."""
    T, D = activations.shape
    
    if window == "hann":
        win = np.hanning(T)[:, None]
        activations = activations * win
    elif window == "hamming":
        win = np.hamming(T)[:, None]
        activations = activations * win
    
    h_hat = np.fft.rfft(activations, axis=0)
    magnitude = np.abs(h_hat)
    
    if n_bins is not None and n_bins < magnitude.shape[0]:
        magnitude = magnitude[:n_bins, :]
    
    return magnitude


def load_multi_layer_data(
    data_dir: Path,
    layer_step: int = 5,
    max_layers: Optional[int] = None,
    max_samples: Optional[int] = None
) -> Tuple[Dict[int, np.ndarray], List[int]]:
    """Load hidden states from multiple layers."""
    layer_data: Dict[int, List[np.ndarray]] = {}
    
    run_dirs = sorted(data_dir.glob("run_*"))
    if not run_dirs:
        npz_files = sorted(data_dir.glob("hidden_layers*.npz"))
        run_dirs = [f.parent for f in npz_files] if npz_files else []
    
    if not run_dirs:
        raise ValueError(f"No run directories found in {data_dir}")
    
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
        selected_layers = [l for l in available_layers if l % layer_step == 0]
        if max_layers:
            selected_layers = selected_layers[:max_layers]
        
        for layer_idx in selected_layers:
            key = f"layer_{layer_idx}"
            if key in data:
                hs = data[key].astype(np.float32)
                if hs.ndim == 3 and hs.shape[0] == 1:
                    hs = hs[0]
                elif hs.ndim != 2:
                    continue
                
                if layer_idx not in layer_data:
                    layer_data[layer_idx] = []
                layer_data[layer_idx].append(hs)
        
        sample_count += 1
    
    result = {}
    for layer_idx, arrays in layer_data.items():
        min_len = min(a.shape[0] for a in arrays)
        aligned = [a[:min_len] for a in arrays]
        result[layer_idx] = np.stack(aligned, axis=0)
    
    layer_indices = sorted(result.keys())
    return result, layer_indices


def compute_spectral_features(
    layer_data: Dict[int, np.ndarray],
    layer_indices: List[int],
    freq_bins: int = 32,
    window: str = "hann"
) -> Tuple[np.ndarray, Dict]:
    """Compute FFT spectrum for each layer and stack."""
    spectral_per_layer = {}
    
    for layer_idx in layer_indices:
        data = layer_data[layer_idx]
        N, T, D = data.shape
        
        spectra = []
        for i in range(N):
            mag = token_fft(data[i], window=window, n_bins=freq_bins)
            spectra.append(mag)
        
        spectral_per_layer[layer_idx] = np.stack(spectra, axis=0)
    
    N, F, D = spectral_per_layer[layer_indices[0]].shape
    
    # Stack layers: (N, n_layers * F, D)
    stacked = np.stack([spectral_per_layer[l] for l in layer_indices], axis=1)
    reshaped = stacked.reshape(N, len(layer_indices) * F, D)
    
    metadata = {
        "n_samples": N,
        "freq_bins": F,
        "hidden_dim": D,
        "layer_indices": layer_indices,
        "n_layers": len(layer_indices),
        "seq_len": len(layer_indices) * F,
        "input_type": "spectral",
    }
    
    return reshaped, metadata


def compute_raw_features(
    layer_data: Dict[int, np.ndarray],
    layer_indices: List[int],
) -> Tuple[np.ndarray, Dict]:
    """Stack raw activations from multiple layers."""
    # Stack layers along sequence dimension: (N, n_layers * T, D)
    first_layer = layer_indices[0]
    N, T, D = layer_data[first_layer].shape
    
    stacked = np.stack([layer_data[l] for l in layer_indices], axis=1)  # (N, n_layers, T, D)
    reshaped = stacked.reshape(N, len(layer_indices) * T, D)
    
    metadata = {
        "n_samples": N,
        "seq_len_per_layer": T,
        "hidden_dim": D,
        "layer_indices": layer_indices,
        "n_layers": len(layer_indices),
        "seq_len": len(layer_indices) * T,
        "input_type": "raw",
    }
    
    return reshaped, metadata


def plot_tica_eigenvalues(eigvals: np.ndarray, out_path: Path, title: str = "TICA Eigenvalues"):
    """Plot TICA eigenvalues on unit circle."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Complex plane (eigenvalues are real for symmetric TICA, but plot anyway)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    
    colors = np.arange(len(eigvals))
    ax1.scatter(eigvals.real, np.zeros_like(eigvals), c=colors, cmap="viridis",
                s=100, edgecolor="black", linewidth=0.5, zorder=5)
    
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.set_xlabel("Eigenvalue")
    ax1.set_ylabel("(Imaginary)")
    ax1.set_title(f"{title} - Complex Plane")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    
    # Bar chart of eigenvalue magnitudes
    ax2.bar(range(len(eigvals)), np.abs(eigvals), color='steelblue', edgecolor='black')
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='|λ|=1')
    ax2.set_xlabel("Component")
    ax2.set_ylabel("|λ|")
    ax2.set_title(f"{title} - Magnitudes")
    ax2.legend()
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_timescales(timescales: np.ndarray, out_path: Path, title: str = "TICA Implied Timescales"):
    """Plot implied timescales."""
    plt.figure(figsize=(8, 5))
    
    # Filter infinite timescales for plotting
    finite_ts = np.where(np.isinf(timescales), np.nan, timescales)
    
    plt.bar(range(len(timescales)), finite_ts, color='coral', edgecolor='black')
    plt.xlabel("TICA Component")
    plt.ylabel("Implied Timescale (tokens)")
    plt.title(title)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comparison(
    tica_eigvals: np.ndarray,
    vampnet_eigvals: Optional[np.ndarray],
    out_path: Path,
    tica_label: str = "TICA"
):
    """Compare TICA and VAMPnet eigenvalues."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    
    # TICA (on real axis)
    ax.scatter(tica_eigvals.real, np.zeros_like(tica_eigvals), 
               c='blue', s=100, label=tica_label, marker='o', edgecolor='black', zorder=5)
    
    # VAMPnet
    if vampnet_eigvals is not None:
        ax.scatter(vampnet_eigvals.real, vampnet_eigvals.imag,
                   c='red', s=100, label='VAMPnet', marker='^', edgecolor='black', zorder=5)
    
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(f"{tica_label} vs VAMPnet Eigenvalues")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.legend()
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_spectral_vs_raw_tica(
    spectral_eigvals: np.ndarray,
    raw_eigvals: np.ndarray,
    out_path: Path
):
    """Compare spectral and raw TICA eigenvalues."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Complex plane comparison
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)
    
    ax1.scatter(spectral_eigvals.real, np.zeros_like(spectral_eigvals),
               c='#2ecc71', s=100, label='Spectral TICA', marker='o', edgecolor='black', zorder=5)
    ax1.scatter(raw_eigvals.real, np.zeros_like(raw_eigvals) + 0.05,  # Slight offset for visibility
               c='#e74c3c', s=100, label='Raw TICA', marker='s', edgecolor='black', zorder=5)
    
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.set_xlabel("Eigenvalue")
    ax1.set_ylabel("(Offset for visibility)")
    ax1.set_title("Eigenvalue Comparison")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-0.5, 0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Bar comparison
    x = np.arange(len(spectral_eigvals))
    width = 0.35
    
    ax2.bar(x - width/2, np.abs(spectral_eigvals), width, label='Spectral TICA', color='#2ecc71', edgecolor='black')
    ax2.bar(x + width/2, np.abs(raw_eigvals), width, label='Raw TICA', color='#e74c3c', edgecolor='black')
    
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='|λ|=1')
    ax2.set_xlabel("Component")
    ax2.set_ylabel("|λ|")
    ax2.set_title("Eigenvalue Magnitudes")
    ax2.legend()
    ax2.grid(True, alpha=0.2, axis='y')
    
    plt.suptitle("Spectral vs Raw TICA Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_tica_projection(
    Z: np.ndarray,
    out_path: Path,
    title: str = "TICA Projection",
    max_points: int = 5000
):
    """Plot first two TICA components."""
    if Z.ndim == 3:
        Z = Z.reshape(-1, Z.shape[-1])
    
    if len(Z) > max_points:
        idx = np.random.choice(len(Z), max_points, replace=False)
        Z = Z[idx]
    
    if Z.shape[1] < 2:
        return
    
    plt.figure(figsize=(8, 8))
    plt.scatter(Z[:, 0], Z[:, 1], c=np.arange(len(Z)), cmap='viridis',
                s=5, alpha=0.5)
    plt.colorbar(label='Sample index')
    plt.xlabel("TICA 1")
    plt.ylabel("TICA 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_tica_analysis(
    features: np.ndarray,
    metadata: Dict,
    n_components: int,
    lag: int,
    regularization: float,
    out_dir: Path,
    prefix: str,
    vampnet_eigvals: Optional[np.ndarray] = None
) -> Dict:
    """Run TICA analysis and save results."""
    mode = metadata.get("input_type", "unknown")
    print(f"\n{'='*50}")
    print(f"Running {prefix} TICA (mode={mode})")
    print(f"{'='*50}")
    print(f"Features shape: {features.shape}")
    
    # Fit TICA
    print(f"Fitting TICA (n_components={n_components}, lag={lag})...")
    tica = TICA(
        n_components=n_components,
        lag=lag,
        regularization=regularization
    )
    Z = tica.fit_transform(features)
    
    print(f"Eigenvalues: {tica.eigvals_}")
    print(f"Timescales: {tica.timescales_}")
    print(f"VAMP-2 score: {tica.score():.4f}")
    
    # Find slowest mode
    finite_mask = np.isfinite(tica.timescales_)
    if np.any(finite_mask):
        slow_idx = int(np.argmax(np.abs(tica.timescales_[finite_mask])))
    else:
        slow_idx = 0
    print(f"Slowest mode: λ={tica.eigvals_[slow_idx]:.4f}, τ={tica.timescales_[slow_idx]:.2f}")
    
    # Save results
    npz_name = f"{prefix}_tica_outputs.npz"
    np.savez(
        out_dir / npz_name,
        eigvals=tica.eigvals_,
        eigvecs=tica.eigvecs_,
        timescales=tica.timescales_,
        mean=tica.mean_,
        projection=Z,
        layer_indices=np.array(metadata.get("layer_indices", [])),
        vamp2_score=tica.score(),
        input_type=mode,
    )
    
    meta_name = f"{prefix}_metadata.json"
    with open(out_dir / meta_name, "w") as f:
        json.dump({
            **metadata,
            "mode": mode,
            "lag": lag,
            "n_components": n_components,
            "vamp2_score": float(tica.score()),
            "kinetic_variance": float(tica.kinetic_variance()),
            "eigvals": tica.eigvals_.tolist(),
            "timescales": [float(t) if np.isfinite(t) else "inf" for t in tica.timescales_],
        }, f, indent=2)
    
    # Plots
    title_suffix = "Spectral" if mode == "spectral" else "Raw"
    plot_tica_eigenvalues(
        tica.eigvals_, 
        out_dir / f"{prefix}_eigenvalues.png",
        title=f"{title_suffix} TICA Eigenvalues"
    )
    plot_timescales(
        tica.timescales_,
        out_dir / f"{prefix}_timescales.png",
        title=f"{title_suffix} TICA Implied Timescales"
    )
    plot_tica_projection(
        Z,
        out_dir / f"{prefix}_projection.png",
        title=f"{title_suffix} TICA Projection"
    )
    
    if vampnet_eigvals is not None:
        plot_comparison(
            tica.eigvals_, vampnet_eigvals,
            out_dir / f"{prefix}_vs_vampnet.png",
            tica_label=f"{title_suffix} TICA"
        )
    
    return {
        "eigvals": tica.eigvals_,
        "timescales": tica.timescales_,
        "vamp2_score": tica.score(),
        "projection": Z,
        "metadata": metadata,
    }


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {args.data_dir}...")
    
    # Load data
    if args.data_file is not None:
        data = np.load(args.data_file)
        layer_data = {}
        for key in data.keys():
            if key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                if layer_idx % args.layer_step == 0:
                    hs = data[key].astype(np.float32)
                    if hs.ndim == 2:
                        hs = hs[None, ...]
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
    
    # Load VAMPnet results for comparison
    vampnet_eigvals = None
    if args.compare_vampnet is not None and args.compare_vampnet.exists():
        vampnet_data = np.load(args.compare_vampnet)
        vampnet_eigvals = vampnet_data.get("eigvals")
        print(f"\nLoaded VAMPnet eigenvalues for comparison")
    
    results = {}
    
    # Run spectral TICA
    if args.mode in ["spectral", "both"]:
        print(f"\nComputing FFT spectrum (bins={args.freq_bins}, window={args.window})...")
        spectral_features, spectral_metadata = compute_spectral_features(
            layer_data, layer_indices,
            freq_bins=args.freq_bins,
            window=args.window
        )
        
        results["spectral"] = run_tica_analysis(
            spectral_features, spectral_metadata,
            n_components=args.n_components,
            lag=args.lag,
            regularization=args.regularization,
            out_dir=out_dir,
            prefix="spectral",
            vampnet_eigvals=vampnet_eigvals
        )
    
    # Run raw TICA
    if args.mode in ["raw", "both"]:
        print(f"\nPreparing raw features...")
        raw_features, raw_metadata = compute_raw_features(layer_data, layer_indices)
        
        results["raw"] = run_tica_analysis(
            raw_features, raw_metadata,
            n_components=args.n_components,
            lag=args.lag,
            regularization=args.regularization,
            out_dir=out_dir,
            prefix="raw",
            vampnet_eigvals=vampnet_eigvals
        )
    
    # Compare spectral vs raw if both were run
    if args.mode == "both" and "spectral" in results and "raw" in results:
        print("\nGenerating spectral vs raw comparison plots...")
        plot_spectral_vs_raw_tica(
            results["spectral"]["eigvals"],
            results["raw"]["eigvals"],
            out_dir / "spectral_vs_raw_comparison.png"
        )
        
        # Summary comparison
        summary = {
            "spectral": {
                "vamp2_score": results["spectral"]["vamp2_score"],
                "eigvals": results["spectral"]["eigvals"].tolist(),
            },
            "raw": {
                "vamp2_score": results["raw"]["vamp2_score"],
                "eigvals": results["raw"]["eigvals"].tolist(),
            },
            "comparison": {
                "vamp2_ratio": results["spectral"]["vamp2_score"] / (results["raw"]["vamp2_score"] + 1e-10),
                "winner": "spectral" if results["spectral"]["vamp2_score"] > results["raw"]["vamp2_score"] else "raw"
            }
        }
        
        with open(out_dir / "comparison_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*50}")
        print("COMPARISON SUMMARY")
        print(f"{'='*50}")
        print(f"Spectral TICA VAMP-2: {results['spectral']['vamp2_score']:.4f}")
        print(f"Raw TICA VAMP-2: {results['raw']['vamp2_score']:.4f}")
        print(f"Winner: {summary['comparison']['winner'].upper()}")
    
    print(f"\n✓ Done! Results saved to {out_dir}")
    print("Output files:")
    for f in sorted(out_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
