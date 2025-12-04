#!/usr/bin/env python3
"""
Comprehensive analysis and visualization tool for spectral dynamics.

This script provides:
1. Side-by-side comparison of VAMPnet and TICA results (spectral and raw)
2. Per-layer spectral analysis and statistics
3. Slow mode visualization and stability analysis
4. Cross-method eigenvalue comparison
5. Publication-quality figures

Example:
python analyze_spectral_dynamics.py \
    --vampnet-dir spectral_vampnet_runs \
    --tica-dir tica_runs \
    --out-dir analysis_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Non-interactive backend
matplotlib.use("Agg")

# Style configuration for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze and visualize spectral dynamics results")
    p.add_argument("--vampnet-dir", type=Path, default=None,
                   help="Directory with spectral VAMPnet results")
    p.add_argument("--tica-dir", type=Path, default=None,
                   help="Directory with TICA results (spectral and/or raw)")
    p.add_argument("--raw-vampnet", type=Path, default=None,
                   help="Directory with original (raw) VAMPnet results")
    p.add_argument("--out-dir", type=Path, default=Path("analysis_results"),
                   help="Output directory for analysis")
    p.add_argument("--report-format", type=str, default="both",
                   choices=["markdown", "json", "both"],
                   help="Format for analysis report")
    p.add_argument("--dpi", type=int, default=200,
                   help="DPI for saved figures")
    return p.parse_args()


def load_results(
    vampnet_dir: Optional[Path] = None,
    tica_dir: Optional[Path] = None,
    raw_vampnet_dir: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """Load results from VAMPnet and TICA runs."""
    results = {}
    
    # Spectral VAMPnet
    if vampnet_dir and vampnet_dir.exists():
        npz_path = vampnet_dir / "spectral_vampnet_outputs.npz"
        meta_path = vampnet_dir / "metadata.json"
        
        if npz_path.exists():
            data = dict(np.load(npz_path, allow_pickle=True))
            results["spectral_vampnet"] = {
                "eigvals": data.get("eigvals"),
                "eigvecs": data.get("eigvecs"),
                "loss_history": data.get("loss_history"),
                "spectral_features": data.get("spectral_features"),
                "layer_indices": data.get("layer_indices"),
                "type": "vampnet",
                "input_type": "spectral",
            }
            if meta_path.exists():
                with open(meta_path) as f:
                    results["spectral_vampnet"]["metadata"] = json.load(f)
    
    # TICA results (both spectral and raw)
    if tica_dir and tica_dir.exists():
        # Spectral TICA
        spectral_npz = tica_dir / "spectral_tica_outputs.npz"
        spectral_meta = tica_dir / "spectral_metadata.json"
        
        if spectral_npz.exists():
            data = dict(np.load(spectral_npz, allow_pickle=True))
            results["spectral_tica"] = {
                "eigvals": data.get("eigvals"),
                "eigvecs": data.get("eigvecs"),
                "timescales": data.get("timescales"),
                "projection": data.get("projection"),
                "vamp2_score": float(data.get("vamp2_score", 0)),
                "type": "tica",
                "input_type": "spectral",
            }
            if spectral_meta.exists():
                with open(spectral_meta) as f:
                    results["spectral_tica"]["metadata"] = json.load(f)
        
        # Raw TICA
        raw_npz = tica_dir / "raw_tica_outputs.npz"
        raw_meta = tica_dir / "raw_metadata.json"
        
        if raw_npz.exists():
            data = dict(np.load(raw_npz, allow_pickle=True))
            results["raw_tica"] = {
                "eigvals": data.get("eigvals"),
                "eigvecs": data.get("eigvecs"),
                "timescales": data.get("timescales"),
                "projection": data.get("projection"),
                "vamp2_score": float(data.get("vamp2_score", 0)),
                "type": "tica",
                "input_type": "raw",
            }
            if raw_meta.exists():
                with open(raw_meta) as f:
                    results["raw_tica"]["metadata"] = json.load(f)
        
        # Legacy: single TICA output (backwards compatibility)
        legacy_npz = tica_dir / "tica_outputs.npz"
        if legacy_npz.exists() and "spectral_tica" not in results:
            data = dict(np.load(legacy_npz, allow_pickle=True))
            results["spectral_tica"] = {
                "eigvals": data.get("eigvals"),
                "eigvecs": data.get("eigvecs"),
                "timescales": data.get("timescales"),
                "projection": data.get("projection"),
                "vamp2_score": float(data.get("vamp2_score", 0)),
                "type": "tica",
                "input_type": str(data.get("input_type", "spectral")),
            }
    
    # Raw VAMPnet (on raw activations)
    if raw_vampnet_dir and raw_vampnet_dir.exists():
        npz_path = raw_vampnet_dir / "vampnet_outputs.npz"
        
        if npz_path.exists():
            data = dict(np.load(npz_path, allow_pickle=True))
            results["raw_vampnet"] = {
                "eigvals": data.get("eigvals"),
                "eigvecs": data.get("eigvecs"),
                "type": "vampnet",
                "input_type": "raw",
            }
    
    return results


def compute_stability_metrics(eigvals: np.ndarray) -> Dict[str, float]:
    """Compute stability metrics from eigenvalues."""
    mags = np.abs(eigvals)
    
    return {
        "max_magnitude": float(np.max(mags)),
        "min_magnitude": float(np.min(mags)),
        "mean_magnitude": float(np.mean(mags)),
        "std_magnitude": float(np.std(mags)),
        "n_stable": int(np.sum(mags <= 1.0)),
        "n_unstable": int(np.sum(mags > 1.0)),
        "spectral_radius": float(np.max(mags)),
        "stability_margin": float(1.0 - np.max(mags)),
    }


def compute_slow_mode_metrics(eigvals: np.ndarray, lag: int = 1) -> Dict[str, Any]:
    """Compute metrics for slow modes."""
    mags = np.abs(eigvals)
    
    # Implied timescales: τ = -lag / ln(|λ|)
    with np.errstate(divide='ignore', invalid='ignore'):
        timescales = np.where(
            mags > 1e-10,
            -lag / np.log(mags + 1e-10),
            np.inf
        )
    
    # Find slowest mode (closest to |λ|=1)
    slow_idx = int(np.argmin(np.abs(1.0 - mags)))
    
    return {
        "slowest_eigval": complex(eigvals[slow_idx]),
        "slowest_magnitude": float(mags[slow_idx]),
        "slowest_timescale": float(timescales[slow_idx]) if np.isfinite(timescales[slow_idx]) else None,
        "timescales": timescales.tolist(),
        "n_slow_modes": int(np.sum(mags > 0.9)),  # Modes with |λ| > 0.9
    }


def plot_eigenvalue_comparison(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
    dpi: int = 200
) -> None:
    """Create comprehensive eigenvalue comparison plot."""
    n_methods = sum(1 for k in results if results[k].get("eigvals") is not None)
    
    if n_methods == 0:
        print("No eigenvalue data to plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: All eigenvalues on complex plane
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1, label='Unit circle')
    
    # Define markers and colors for different methods
    markers = {
        'spectral_vampnet': 'o', 
        'spectral_tica': 's', 
        'raw_vampnet': '^',
        'raw_tica': 'D',
    }
    colors = {
        'spectral_vampnet': '#2ecc71',  # Green
        'spectral_tica': '#3498db',      # Blue
        'raw_vampnet': '#e74c3c',        # Red
        'raw_tica': '#9b59b6',           # Purple
    }
    
    for name, data in results.items():
        eigvals = data.get("eigvals")
        if eigvals is not None:
            eigvals = np.array(eigvals)
            label = name.replace('_', ' ').title()
            input_type = data.get("input_type", "")
            if input_type:
                label = f"{label} ({input_type})"
            
            ax1.scatter(eigvals.real, eigvals.imag,
                       marker=markers.get(name, 'o'),
                       c=colors.get(name, 'gray'),
                       s=80, label=label,
                       edgecolor='black', linewidth=0.5, alpha=0.8)
    
    ax1.axhline(0, color="gray", linewidth=0.5)
    ax1.axvline(0, color="gray", linewidth=0.5)
    ax1.set_xlabel("Re(λ)")
    ax1.set_ylabel("Im(λ)")
    ax1.set_title("Eigenvalues on Complex Plane")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.2)
    
    # Plot 2: Eigenvalue magnitudes bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    
    bar_width = 0.2
    x_offset = 0
    max_n_eigvals = max(len(np.array(d.get("eigvals", []))) for d in results.values())
    
    for name, data in results.items():
        eigvals = data.get("eigvals")
        if eigvals is not None:
            eigvals = np.array(eigvals)
            mags = np.abs(eigvals)
            x = np.arange(len(mags)) + x_offset
            label = name.replace('_', ' ').title()
            ax2.bar(x, mags, bar_width, label=label,
                   color=colors.get(name, 'gray'), edgecolor='black', alpha=0.8)
            x_offset += bar_width
    
    ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Stability boundary')
    ax2.set_xlabel("Eigenvalue Index")
    ax2.set_ylabel("|λ|")
    ax2.set_title("Eigenvalue Magnitudes")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2, axis='y')
    
    # Plot 3: VAMP-2 scores comparison
    ax3 = fig.add_subplot(gs[1, 0])
    
    vamp_scores = []
    method_names = []
    method_colors = []
    
    for name, data in results.items():
        eigvals = data.get("eigvals")
        if eigvals is not None:
            # Compute VAMP-2 score as sum of squared eigenvalues
            score = data.get("vamp2_score")
            if score is None:
                score = float(np.sum(np.array(eigvals) ** 2))
            vamp_scores.append(score)
            method_names.append(name.replace('_', ' ').title())
            method_colors.append(colors.get(name, 'gray'))
    
    if vamp_scores:
        x = np.arange(len(method_names))
        bars = ax3.bar(x, vamp_scores, color=method_colors, edgecolor='black')
        ax3.set_xticks(x)
        ax3.set_xticklabels(method_names, rotation=15, ha='right')
        ax3.set_ylabel("VAMP-2 Score")
        ax3.set_title("VAMP-2 Score Comparison (Higher = Better)")
        ax3.grid(True, alpha=0.2, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, vamp_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Timescales (for TICA methods)
    ax4 = fig.add_subplot(gs[1, 1])
    
    has_timescales = False
    for name, data in results.items():
        timescales = data.get("timescales")
        eigvals = data.get("eigvals")
        
        if timescales is not None:
            ts = np.array(timescales)
        elif eigvals is not None:
            # Compute timescales from eigenvalues
            mags = np.abs(np.array(eigvals))
            ts = np.where(mags > 1e-10, -1.0 / np.log(mags + 1e-10), np.inf)
        else:
            continue
        
        finite_ts = np.where(np.isinf(ts), np.nan, ts)
        
        if not np.all(np.isnan(finite_ts)):
            ax4.plot(range(len(finite_ts)), finite_ts, 
                    marker=markers.get(name, 'o'),
                    color=colors.get(name, 'gray'),
                    label=name.replace('_', ' ').title(),
                    linewidth=2, markersize=8)
            has_timescales = True
    
    if has_timescales:
        ax4.set_xlabel("Mode Index")
        ax4.set_ylabel("Implied Timescale (tokens)")
        ax4.set_title("Implied Timescales")
        ax4.set_yscale('log')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.2)
    else:
        ax4.text(0.5, 0.5, "No timescale data available", 
                ha='center', va='center', transform=ax4.transAxes)
    
    plt.suptitle("Spectral Dynamics Analysis: Method Comparison", fontsize=14, fontweight='bold')
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved eigenvalue comparison to {out_path}")


def plot_spectral_vs_raw_summary(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
    dpi: int = 200
) -> None:
    """Plot summary comparing spectral vs raw input methods."""
    spectral_methods = {k: v for k, v in results.items() if v.get("input_type") == "spectral"}
    raw_methods = {k: v for k, v in results.items() if v.get("input_type") == "raw"}
    
    if not spectral_methods or not raw_methods:
        print("Need both spectral and raw methods for comparison")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Colors
    spectral_color = '#3498db'
    raw_color = '#e74c3c'
    
    # Plot 1: VAMP-2 scores
    ax1 = axes[0]
    labels = []
    spectral_scores = []
    raw_scores = []
    
    for method_type in ['vampnet', 'tica']:
        spectral_key = f'spectral_{method_type}'
        raw_key = f'raw_{method_type}'
        
        if spectral_key in spectral_methods and raw_key in raw_methods:
            labels.append(method_type.upper())
            
            s_data = spectral_methods[spectral_key]
            r_data = raw_methods[raw_key]
            
            s_score = s_data.get('vamp2_score')
            if s_score is None and s_data.get('eigvals') is not None:
                s_score = float(np.sum(np.array(s_data['eigvals']) ** 2))
            
            r_score = r_data.get('vamp2_score')
            if r_score is None and r_data.get('eigvals') is not None:
                r_score = float(np.sum(np.array(r_data['eigvals']) ** 2))
            
            spectral_scores.append(s_score or 0)
            raw_scores.append(r_score or 0)
    
    if labels:
        x = np.arange(len(labels))
        width = 0.35
        ax1.bar(x - width/2, spectral_scores, width, label='Spectral (FFT)', color=spectral_color, edgecolor='black')
        ax1.bar(x + width/2, raw_scores, width, label='Raw', color=raw_color, edgecolor='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel('VAMP-2 Score')
        ax1.set_title('VAMP-2 Score by Input Type')
        ax1.legend()
        ax1.grid(True, alpha=0.2, axis='y')
    
    # Plot 2: Spectral radius comparison
    ax2 = axes[1]
    labels = []
    spectral_radii_s = []
    spectral_radii_r = []
    
    for method_type in ['vampnet', 'tica']:
        spectral_key = f'spectral_{method_type}'
        raw_key = f'raw_{method_type}'
        
        if spectral_key in spectral_methods and raw_key in raw_methods:
            labels.append(method_type.upper())
            
            s_eigvals = spectral_methods[spectral_key].get('eigvals')
            r_eigvals = raw_methods[raw_key].get('eigvals')
            
            spectral_radii_s.append(float(np.max(np.abs(s_eigvals))) if s_eigvals is not None else 0)
            spectral_radii_r.append(float(np.max(np.abs(r_eigvals))) if r_eigvals is not None else 0)
    
    if labels:
        x = np.arange(len(labels))
        ax2.bar(x - width/2, spectral_radii_s, width, label='Spectral (FFT)', color=spectral_color, edgecolor='black')
        ax2.bar(x + width/2, spectral_radii_r, width, label='Raw', color=raw_color, edgecolor='black')
        ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Stability')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Spectral Radius')
        ax2.set_title('Spectral Radius by Input Type')
        ax2.legend()
        ax2.grid(True, alpha=0.2, axis='y')
    
    # Plot 3: Method preference summary
    ax3 = axes[2]
    
    comparisons = []
    winners = []
    
    for method_type in ['vampnet', 'tica']:
        spectral_key = f'spectral_{method_type}'
        raw_key = f'raw_{method_type}'
        
        if spectral_key in spectral_methods and raw_key in raw_methods:
            s_data = spectral_methods[spectral_key]
            r_data = raw_methods[raw_key]
            
            s_score = s_data.get('vamp2_score')
            if s_score is None and s_data.get('eigvals') is not None:
                s_score = float(np.sum(np.array(s_data['eigvals']) ** 2))
            
            r_score = r_data.get('vamp2_score')
            if r_score is None and r_data.get('eigvals') is not None:
                r_score = float(np.sum(np.array(r_data['eigvals']) ** 2))
            
            comparisons.append(method_type.upper())
            if s_score and r_score:
                winners.append('Spectral' if s_score > r_score else 'Raw')
            else:
                winners.append('N/A')
    
    if comparisons:
        spectral_wins = winners.count('Spectral')
        raw_wins = winners.count('Raw')
        
        ax3.pie([spectral_wins, raw_wins], 
               labels=['Spectral Wins', 'Raw Wins'],
               colors=[spectral_color, raw_color],
               autopct='%1.0f%%',
               startangle=90)
        ax3.set_title('Winner by VAMP-2 Score')
    
    plt.suptitle('Spectral vs Raw Input Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved spectral vs raw comparison to {out_path}")


def plot_training_comparison(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
    dpi: int = 200
) -> None:
    """Plot training curves for VAMPnet methods."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    has_data = False
    colors = {'spectral_vampnet': '#2ecc71', 'raw_vampnet': '#e74c3c'}
    
    for name, data in results.items():
        loss_history = data.get("loss_history")
        if loss_history is not None:
            loss_history = np.array(loss_history)
            ax.plot(loss_history, label=name.replace('_', ' ').title(),
                   color=colors.get(name, 'gray'), linewidth=2)
            has_data = True
    
    if has_data:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (Negative VAMP-2 Score)")
        ax.set_title("VAMPnet Training Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved training comparison to {out_path}")
    
    plt.close()


def plot_spectral_heatmap(
    spectral_features: np.ndarray,
    layer_indices: np.ndarray,
    out_path: Path,
    dpi: int = 200
) -> None:
    """Plot heatmap of spectral power across layers and frequencies."""
    if spectral_features is None:
        return
    
    # Average over samples and hidden dim: (N, seq, D) -> (seq,)
    mean_power = np.mean(np.abs(spectral_features) ** 2, axis=(0, 2))
    
    # Reshape to (n_layers, freq_bins) if possible
    n_layers = len(layer_indices)
    if len(mean_power) % n_layers == 0:
        freq_bins = len(mean_power) // n_layers
        power_matrix = mean_power.reshape(n_layers, freq_bins)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(power_matrix, aspect='auto', cmap='viridis',
                      origin='lower')
        
        ax.set_xlabel("Frequency Bin")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([str(l) for l in layer_indices])
        ax.set_title("Spectral Power Distribution Across Layers")
        
        plt.colorbar(im, ax=ax, label='Power')
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved spectral heatmap to {out_path}")
        plt.close()


def plot_projection_comparison(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
    dpi: int = 200,
    max_points: int = 3000
) -> None:
    """Plot 2D projections from different methods."""
    projections = {}
    
    for name, data in results.items():
        proj = data.get("projection")
        if proj is not None:
            proj = np.array(proj)
            if proj.ndim == 3:
                proj = proj.reshape(-1, proj.shape[-1])
            if proj.shape[1] >= 2:
                projections[name.replace('_', ' ').title()] = proj[:, :2]
    
    if not projections:
        print("No projection data available")
        return
    
    n_projs = len(projections)
    cols = min(3, n_projs)
    rows = (n_projs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_projs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_projs > 1 else [axes]
    
    colors_map = {
        'Spectral Vampnet': '#2ecc71',
        'Spectral Tica': '#3498db',
        'Raw Vampnet': '#e74c3c',
        'Raw Tica': '#9b59b6',
    }
    
    for i, (name, proj) in enumerate(projections.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        if len(proj) > max_points:
            idx = np.random.choice(len(proj), max_points, replace=False)
            proj = proj[idx]
        
        color = colors_map.get(name, 'gray')
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)),
                       cmap='viridis', s=3, alpha=0.5)
        ax.set_xlabel(f"Component 1")
        ax.set_ylabel(f"Component 2")
        ax.set_title(f"{name} Projection")
        plt.colorbar(sc, ax=ax, label='Sample index')
        ax.grid(True, alpha=0.2)
    
    # Hide unused axes
    for i in range(len(projections), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle("Latent Space Projections", fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved projection comparison to {out_path}")
    plt.close()


def generate_report(
    results: Dict[str, Dict[str, Any]],
    out_dir: Path,
    format: str = "both"
) -> None:
    """Generate analysis report."""
    
    report_data = {
        "methods": [],
        "comparison": {},
        "spectral_vs_raw": {},
    }
    
    for name, data in results.items():
        eigvals = data.get("eigvals")
        if eigvals is None:
            continue
        
        eigvals = np.array(eigvals)
        stability = compute_stability_metrics(eigvals)
        slow_modes = compute_slow_mode_metrics(eigvals)
        
        method_report = {
            "name": name,
            "type": data.get("type", "unknown"),
            "input_type": data.get("input_type", "unknown"),
            "n_eigenvalues": len(eigvals),
            "stability": stability,
            "slow_modes": {
                "slowest_magnitude": slow_modes["slowest_magnitude"],
                "n_slow_modes": slow_modes["n_slow_modes"],
            },
        }
        
        if "metadata" in data:
            method_report["metadata"] = data["metadata"]
        
        vamp2 = data.get("vamp2_score")
        if vamp2 is None:
            vamp2 = float(np.sum(eigvals ** 2))
        method_report["vamp2_score"] = vamp2
        
        report_data["methods"].append(method_report)
    
    # Comparison metrics
    if len(report_data["methods"]) >= 2:
        eigval_sets = {m["name"]: np.array(results[m["name"]]["eigvals"]) 
                      for m in report_data["methods"]}
        
        # Compare stability
        stabilities = {n: compute_stability_metrics(e) for n, e in eigval_sets.items()}
        report_data["comparison"]["stability_ranking"] = sorted(
            stabilities.keys(),
            key=lambda x: stabilities[x]["spectral_radius"]
        )
        
        # VAMP-2 ranking
        vamp_scores = {m["name"]: m["vamp2_score"] for m in report_data["methods"]}
        report_data["comparison"]["vamp2_ranking"] = sorted(
            vamp_scores.keys(),
            key=lambda x: vamp_scores[x],
            reverse=True
        )
    
    # Spectral vs Raw comparison
    spectral_methods = [m for m in report_data["methods"] if m["input_type"] == "spectral"]
    raw_methods = [m for m in report_data["methods"] if m["input_type"] == "raw"]
    
    if spectral_methods and raw_methods:
        report_data["spectral_vs_raw"] = {
            "spectral_avg_vamp2": np.mean([m["vamp2_score"] for m in spectral_methods]),
            "raw_avg_vamp2": np.mean([m["vamp2_score"] for m in raw_methods]),
            "recommendation": "spectral" if np.mean([m["vamp2_score"] for m in spectral_methods]) > 
                                           np.mean([m["vamp2_score"] for m in raw_methods]) else "raw"
        }
    
    # Save reports
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if format in ["json", "both"]:
        with open(out_dir / "analysis_report.json", "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        print(f"Saved JSON report to {out_dir / 'analysis_report.json'}")
    
    if format in ["markdown", "both"]:
        md_lines = [
            "# Spectral Dynamics Analysis Report",
            "",
            "## Summary",
            "",
            f"**Methods Analyzed**: {len(report_data['methods'])}",
            "",
            "| Method | Type | Input | VAMP-2 Score | Spectral Radius | Stable Modes |",
            "|--------|------|-------|--------------|-----------------|--------------|",
        ]
        
        for method in report_data["methods"]:
            md_lines.append(
                f"| {method['name']} | {method['type']} | {method['input_type']} | "
                f"{method['vamp2_score']:.4f} | {method['stability']['spectral_radius']:.4f} | "
                f"{method['stability']['n_stable']}/{method['n_eigenvalues']} |"
            )
        
        md_lines.extend(["", "## Method Details", ""])
        
        for method in report_data["methods"]:
            md_lines.extend([
                f"### {method['name'].replace('_', ' ').title()}",
                "",
                f"- **Type**: {method['type']}",
                f"- **Input**: {method['input_type']}",
                f"- **Number of eigenvalues**: {method['n_eigenvalues']}",
                f"- **Spectral radius**: {method['stability']['spectral_radius']:.4f}",
                f"- **Mean |λ|**: {method['stability']['mean_magnitude']:.4f}",
                f"- **Stable modes**: {method['stability']['n_stable']}/{method['n_eigenvalues']}",
                f"- **VAMP-2 Score**: {method['vamp2_score']:.4f}",
                "",
            ])
        
        if "vamp2_ranking" in report_data.get("comparison", {}):
            md_lines.extend([
                "## Rankings",
                "",
                "### VAMP-2 Score (highest first)",
                "",
            ])
            for i, name in enumerate(report_data["comparison"]["vamp2_ranking"], 1):
                md_lines.append(f"{i}. {name.replace('_', ' ').title()}")
            
            md_lines.extend([
                "",
                "### Stability (most stable first)",
                "",
            ])
            for i, name in enumerate(report_data["comparison"]["stability_ranking"], 1):
                md_lines.append(f"{i}. {name.replace('_', ' ').title()}")
        
        if report_data.get("spectral_vs_raw"):
            md_lines.extend([
                "",
                "## Spectral vs Raw Comparison",
                "",
                f"- **Spectral (FFT) average VAMP-2**: {report_data['spectral_vs_raw']['spectral_avg_vamp2']:.4f}",
                f"- **Raw average VAMP-2**: {report_data['spectral_vs_raw']['raw_avg_vamp2']:.4f}",
                f"- **Recommendation**: Use **{report_data['spectral_vs_raw']['recommendation'].upper()}** features",
                "",
            ])
        
        with open(out_dir / "analysis_report.md", "w") as f:
            f.write("\n".join(md_lines))
        print(f"Saved Markdown report to {out_dir / 'analysis_report.md'}")


def main():
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading results...")
    results = load_results(
        vampnet_dir=args.vampnet_dir,
        tica_dir=args.tica_dir,
        raw_vampnet_dir=args.raw_vampnet
    )
    
    if not results:
        print("No results found! Please specify at least one of:")
        print("  --vampnet-dir, --tica-dir, or --raw-vampnet")
        return
    
    print(f"Loaded results for: {list(results.keys())}")
    for name, data in results.items():
        print(f"  - {name}: type={data.get('type')}, input={data.get('input_type')}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    
    plot_eigenvalue_comparison(results, out_dir / "eigenvalue_comparison.png", args.dpi)
    plot_training_comparison(results, out_dir / "training_comparison.png", args.dpi)
    plot_projection_comparison(results, out_dir / "projection_comparison.png", args.dpi)
    
    # Spectral vs raw comparison
    spectral_methods = {k: v for k, v in results.items() if v.get("input_type") == "spectral"}
    raw_methods = {k: v for k, v in results.items() if v.get("input_type") == "raw"}
    if spectral_methods and raw_methods:
        plot_spectral_vs_raw_summary(results, out_dir / "spectral_vs_raw_summary.png", args.dpi)
    
    # Spectral heatmap if available
    if "spectral_vampnet" in results:
        spectral = results["spectral_vampnet"]
        if spectral.get("spectral_features") is not None and spectral.get("layer_indices") is not None:
            plot_spectral_heatmap(
                spectral["spectral_features"],
                spectral["layer_indices"],
                out_dir / "spectral_heatmap.png",
                args.dpi
            )
    
    # Generate report
    print("\nGenerating analysis report...")
    generate_report(results, out_dir, args.report_format)
    
    print(f"\n✓ Analysis complete! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
