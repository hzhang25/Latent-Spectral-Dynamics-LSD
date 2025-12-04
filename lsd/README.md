# LSD: Lightweight Spectral Dynamics Toolkit

**Standalone scripts for spectral analysis of LLM hidden state dynamics**

This folder contains a self-contained toolkit for capturing, transforming, and analyzing latent dynamics in large language models using spectral methods. Unlike the main `src/` framework (which implements a full 3-axis decomposition), these scripts focus on **practical, minimal-dependency workflows** for rapid experimentation.

---

## Overview

```
lsd/
├── extract_activations.py     # NEW: Multi-dataset activation extraction
├── simple_lsd.py              # Core: FFT + DMD analysis
├── capture_dataset_raw.py     # Data collection utility
├── train_vampnet.py           # VAMPnet on raw activations
├── train_spectral_vampnet.py  # VAMPnet on FFT-transformed data
├── train_tica.py              # TICA (spectral and raw modes)
├── analyze_spectral_dynamics.py  # Visualization & comparison
├── processed_data/            # Input: Deception detection datasets (JSON)
├── data/                      # Output: Extracted activations (NPZ)
│   ├── instruction_pairs/     #   - AILiar honest/deceptive pairs
│   ├── roleplaying/           #   - Role-play scenarios
│   ├── sandbagging/           #   - Sandbagging detection
│   └── alpaca/                #   - Ground truth control
├── activations_and_dmd.npz    # Example output data
├── raw_captures.tar.gz        # Archived capture data
└── README.md                  # This file
```

---

## Quick Start

```bash
# 1. Extract activations from deception datasets (NEW)
python extract_activations.py \
    --data-dir processed_data \
    --out-dir data \
    --datasets instruction_pairs roleplaying alpaca \
    --layer-step 5

# 2. Run spectral VAMPnet on extracted data
python train_spectral_vampnet.py --data-dir data/instruction_pairs/0_honest --layer-step 5

# 3. Run TICA (both spectral and raw modes)
python train_tica.py --data-dir data/instruction_pairs/0_honest --layer-step 5 --mode both

# 4. Compare all methods
python analyze_spectral_dynamics.py --vampnet-dir spectral_vampnet_runs --tica-dir tica_runs
```

---

## Scripts

### 0. `extract_activations.py` — Multi-Dataset Activation Extraction (NEW)

Extracts hidden state activations from LLaMA 3.3 70B for four deception detection datasets. Each sample is saved in its own subfolder with the prompt metadata preserved.

```bash
# Extract all datasets (outputs to lsd/data/)
python extract_activations.py \
    --model-id meta-llama/Llama-3.3-70B-Instruct \
    --data-dir processed_data \
    --out-dir data \
    --datasets instruction_pairs roleplaying sandbagging alpaca \
    --layer-step 5

# Extract specific dataset with custom settings
python extract_activations.py \
    --data-dir processed_data \
    --out-dir data \
    --datasets instruction_pairs \
    --layer-step 10 \
    --max-samples 20 \
    --resume  # Skip already processed samples
```

**Supported Datasets:**

| Dataset | Source | Examples | Labels |
|---------|--------|----------|--------|
| `instruction_pairs` | AILiar paper | 54 | honest/deceptive pairs |
| `roleplaying` | Custom scenarios | 742 | honest/deceptive pairs |
| `sandbagging` | Anthropic Sabotage Evals | Variable | sandbag/control |
| `alpaca` | HuggingFace (tatsu-lab/alpaca) | Up to 10k | honest (ground truth) |

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-id` | `meta-llama/Llama-3.3-70B-Instruct` | HuggingFace model |
| `--data-dir` | `processed_data` | Input data directory |
| `--out-dir` | `data` | Output directory |
| `--datasets` | instruction_pairs, roleplaying, alpaca | Datasets to process |
| `--layer-step` | 5 | Layer sampling (0, 5, 10, ...) |
| `--max-samples` | None | Limit samples per dataset |
| `--alpaca-max` | 500 | Max Alpaca samples |
| `--resume` | False | Skip existing samples |

**Output Structure:**

```
lsd/data/
├── extraction_metadata.json    # Model & extraction settings
├── instruction_pairs/
│   ├── prompts.json            # All prompts metadata
│   ├── 0_honest/
│   │   ├── hidden_layers.npz   # Keys: layer_0, layer_5, ...
│   │   └── prompt.json         # Individual prompt info
│   ├── 0_deceptive/
│   │   ├── hidden_layers.npz
│   │   └── prompt.json
│   └── ...
├── roleplaying/
│   ├── prompts.json
│   └── ...
├── sandbagging/
│   ├── prompts.json
│   └── ...
└── alpaca/
    ├── prompts.json
    └── ...
```

---

### 1. `simple_lsd.py` — Minimal Spectral Analysis Demo

The foundational script for latent spectral dynamics. Performs:
- **Token-axis FFT**: Power spectral density over sequence positions
- **Dynamic Mode Decomposition (DMD)**: Koopman eigenvalue analysis

```bash
# Basic usage (default: LLaMA 70B)
python simple_lsd.py --prompts "Describe a storm over the ocean."

# With custom model and generation
python simple_lsd.py \
    --model-id meta-llama/Llama-3.3-70B-Instruct \
    --generate \
    --max-new-tokens 64 \
    --save-npz \
    --out-dir outputs/

# Analyze specific layer with dataset
python simple_lsd.py \
    --dataset-json prompts.json \
    --layer 40 \
    --save-hidden-layers \
    --hidden-layer-count 10
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--model-id` | LLaMA-3.3-70B | HuggingFace model path |
| `--layer` | -1 (last) | Hidden layer to analyze |
| `--generate` | False | Generate text and analyze output |
| `--save-npz` | False | Save activations and DMD results |
| `--save-hidden-layers` | False | Save multiple layer activations |
| `--rank` | None | DMD truncation rank |

**Outputs:**
- `spectrum.png` — Token-axis power spectral density
- `dmd_eigs.png` — DMD eigenvalues on complex plane
- `activations_and_dmd.npz` — Raw data (if `--save-npz`)

---

### 2. `capture_dataset_raw.py` — Bulk Activation Collection

Captures raw hidden states and attention maps for systematic experiments.

```bash
python capture_dataset_raw.py \
    --dataset-json /path/to/prompts.json \
    --model-id meta-llama/Llama-3.3-70B-Instruct \
    --runs-per-prompt 5 \
    --layer-count 10 \
    --out-dir outputs/raw_captures
```

**Output Structure:**
```
raw_captures/
├── prompt_000/
│   ├── prompt.txt           # Original prompt text
│   ├── run_01/
│   │   ├── hidden_layers.npz  # Keys: layer_0, layer_1, ...
│   │   ├── attentions.npz     # Keys: layer_0, layer_1, ...
│   │   └── inputs.npz         # input_ids, attention_mask
│   ├── run_02/
│   └── ...
├── prompt_001/
└── ...
```

---

### 3. `train_vampnet.py` — VAMPnet on Raw Activations

Trains a Variational Approach for Markov Processes network to learn slow dynamical modes directly from raw hidden states.

```bash
python train_vampnet.py \
    --data activations_and_dmd.npz \
    --lag 1 \
    --latent-dim 4 \
    --epochs 10 \
    --device cuda \
    --out-dir vampnet_runs
```

**Method:**
1. Forms time-lagged pairs `(x_t, x_{t+lag})`
2. Optimizes VAMP-2 score via neural network
3. Estimates Koopman eigenvalues from learned projection

**Outputs:**
- `vampnet_outputs.npz` — Eigenvalues, eigenvectors, normalization params
- `vampnet_model.pt` — Trained PyTorch model
- `vampnet_eigs.png` — Eigenvalue plot

---

### 4. `train_spectral_vampnet.py` — VAMPnet on FFT Spectrum

Trains VAMPnet on **FFT-transformed** activations from multiple layers.

```bash
python train_spectral_vampnet.py \
    --data-dir raw_captures/prompt_000 \
    --layer-step 5 \
    --freq-bins 32 \
    --latent-dim 8 \
    --epochs 20 \
    --device cuda \
    --out-dir spectral_vampnet_runs
```

**Key Features:**
- Multi-layer support with configurable step (`--layer-step 5` → layers 0, 5, 10, ...)
- FFT preprocessing with windowing (`--window hann|hamming|none`)
- Frequency truncation for memory efficiency (`--freq-bins`)
- Option to concatenate layer spectra (`--combine-layers`)

**Outputs:**
- `spectral_vampnet_outputs.npz` — Eigenvalues, spectral features, loss history
- `spectral_vampnet_model.pt` — Trained model with metadata
- `spectral_vampnet_eigs.png` — Eigenvalue plot
- `spectral_power.png` — Power distribution across layers/frequencies
- `training_loss.png` — Convergence curve

---

### 5. `train_tica.py` — Time-lagged ICA (Spectral & Raw Modes)

Implements TICA as a fast, deterministic alternative to VAMPnet. **Supports both spectral (FFT) and raw activation inputs.**

```bash
# Spectral TICA (default) - on FFT-transformed features
python train_tica.py \
    --data-dir raw_captures/prompt_000 \
    --layer-step 5 \
    --n-components 8 \
    --mode spectral

# Raw TICA - directly on raw activations
python train_tica.py \
    --data-dir raw_captures/prompt_000 \
    --layer-step 5 \
    --n-components 8 \
    --mode raw

# Compare both modes in one run
python train_tica.py \
    --data-dir raw_captures/prompt_000 \
    --layer-step 5 \
    --n-components 8 \
    --mode both \
    --compare-vampnet spectral_vampnet_runs/spectral_vampnet_outputs.npz
```

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | spectral | `spectral` (FFT), `raw`, or `both` |
| `--layer-step` | 5 | Layer increment (0, 5, 10, ...) |
| `--freq-bins` | 32 | FFT frequency bins (spectral mode) |
| `--n-components` | 8 | TICA components to extract |
| `--lag` | 1 | Time lag for covariance |
| `--compare-vampnet` | None | Path to VAMPnet results for comparison |

**Method:**
- Solves generalized eigenvalue problem: `C₀₁ v = λ C₀₀ v`
- Computes implied timescales: `τ = -lag / ln(|λ|)`
- No neural network training (fast, deterministic)

**Outputs (per mode):**
- `{spectral,raw}_tica_outputs.npz` — Eigenvalues, eigenvectors, timescales
- `{spectral,raw}_eigenvalues.png` — Eigenvalues on complex plane
- `{spectral,raw}_timescales.png` — Implied timescales plot
- `{spectral,raw}_projection.png` — 2D latent projection
- `spectral_vs_raw_comparison.png` — Side-by-side comparison (if `--mode both`)
- `comparison_summary.json` — VAMP-2 scores and winner

---

### 6. `analyze_spectral_dynamics.py` — Visualization & Comparison

Comprehensive analysis tool for comparing all methods.

```bash
python analyze_spectral_dynamics.py \
    --vampnet-dir spectral_vampnet_runs \
    --tica-dir tica_runs \
    --raw-vampnet vampnet_runs \
    --out-dir analysis_results \
    --report-format both
```

**Features:**
- Compares up to 4 methods: Spectral VAMPnet, Raw VAMPnet, Spectral TICA, Raw TICA
- VAMP-2 score ranking
- Stability analysis (spectral radius, stable/unstable mode counts)
- Implied timescale comparison
- Spectral vs Raw input type summary
- Latent space projection visualizations

**Outputs:**
- `eigenvalue_comparison.png` — All methods on same complex plane
- `spectral_vs_raw_summary.png` — Comparison of input types
- `training_comparison.png` — Loss curves (VAMPnet methods)
- `projection_comparison.png` — 2D projections from all methods
- `spectral_heatmap.png` — Power across layers and frequencies
- `analysis_report.md` — Markdown summary with tables
- `analysis_report.json` — Structured data for programmatic use

---

## Complete Pipeline

```bash
# Step 1: Capture raw activations
python capture_dataset_raw.py \
    --dataset-json my_prompts.json \
    --runs-per-prompt 5 \
    --layer-count 20 \
    --out-dir raw_captures

# Step 2: Quick spectral analysis
python simple_lsd.py \
    --dataset-json my_prompts.json \
    --save-npz \
    --out-dir quick_analysis

# Step 3: Train VAMPnet on FFT spectrum
python train_spectral_vampnet.py \
    --data-dir raw_captures/prompt_000 \
    --layer-step 5 \
    --out-dir spectral_vampnet_runs

# Step 4: Run TICA (spectral and raw)
python train_tica.py \
    --data-dir raw_captures/prompt_000 \
    --layer-step 5 \
    --mode both \
    --compare-vampnet spectral_vampnet_runs/spectral_vampnet_outputs.npz \
    --out-dir tica_runs

# Step 5: Comprehensive analysis
python analyze_spectral_dynamics.py \
    --vampnet-dir spectral_vampnet_runs \
    --tica-dir tica_runs \
    --out-dir analysis_results
```

---

## Method Comparison

### By Method Type

| Method | Input | Type | Speed | Nonlinearity | Training |
|--------|-------|------|-------|--------------|----------|
| **DMD** | Raw | Linear | Fast | None | None |
| **Raw VAMPnet** | Raw | Neural | Medium | Yes | SGD |
| **Spectral VAMPnet** | FFT | Neural | Medium | Yes | SGD |
| **Raw TICA** | Raw | Linear | Fast | None | Eigendecomp |
| **Spectral TICA** | FFT | Linear | Fast | None | Eigendecomp |

### Spectral vs Raw Input

| Aspect | Spectral (FFT) | Raw |
|--------|----------------|-----|
| **Preprocessing** | FFT magnitude spectrum | None (direct) |
| **Information** | Frequency content | Full temporal |
| **Dimensionality** | Reduced (freq bins) | Full sequence |
| **Best for** | Periodic patterns | Local dynamics |
| **Computation** | FFT overhead | Direct |

### When to Use Each

| Scenario | Recommended Method |
|----------|-------------------|
| Quick baseline | DMD (`simple_lsd.py`) |
| Single-layer analysis | Raw VAMPnet |
| Multi-layer frequency patterns | Spectral VAMPnet |
| Fast linear baseline | Spectral TICA |
| Validate nonlinear learning | Compare VAMPnet vs TICA |
| Compare input representations | Run `--mode both` with TICA |

---

## Data Formats

### Input: `hidden_layers.npz`
```python
{
    "layer_0": np.ndarray,   # (1, seq_len, hidden_dim) or (seq_len, hidden_dim)
    "layer_5": np.ndarray,
    "layer_10": np.ndarray,
    ...
}
```

### Output: `*_outputs.npz`
```python
# VAMPnet outputs
{
    "eigvals": np.ndarray,      # Complex eigenvalues
    "eigvecs": np.ndarray,      # Corresponding eigenvectors
    "loss_history": np.ndarray, # Training loss curve
    "spectral_features": np.ndarray,  # (for spectral VAMPnet)
}

# TICA outputs
{
    "eigvals": np.ndarray,      # Real eigenvalues
    "eigvecs": np.ndarray,      # Eigenvectors
    "timescales": np.ndarray,   # Implied timescales
    "projection": np.ndarray,   # Projected data
    "vamp2_score": float,       # VAMP-2 score
    "input_type": str,          # "spectral" or "raw"
}
```

---

## Dependencies

**Core requirements:**
```
numpy>=1.21.0
matplotlib>=3.5.0
torch>=1.12.0
scipy>=1.9.0
transformers>=4.30.0
```

Install with:
```bash
pip install -r requirements.txt
```

**Optional (for large models):**
```bash
# Faster tokenization
pip install tokenizers

# Multi-GPU / memory optimization
pip install accelerate

# 4-bit/8-bit quantization (reduce VRAM)
pip install bitsandbytes
```

---

## Relationship to Main Framework

| Feature | Main Framework (`src/`) | This Toolkit (`lsd/`) |
|---------|------------------------|----------------------|
| **Spectral Axes** | 3 (Fourier + Laplacian + Koopman) | 2 (Fourier + Koopman) |
| **Graph Laplacian** | ✅ Attention graph decomposition | ❌ Not implemented |
| **Joint Expansion** | ✅ Combined spectral basis | ❌ Separate analyses |
| **Memory Optimization** | 4-bit quantization | Full precision |
| **Target Models** | Mistral-7B (8GB VRAM) | LLaMA-70B (multi-GPU) |
| **VAMPnet** | ❌ | ✅ (Spectral + Raw) |
| **TICA** | ❌ | ✅ (Spectral + Raw) |
| **Input Comparison** | N/A | ✅ Spectral vs Raw |

The `/lsd` toolkit is designed for **rapid prototyping** on powerful hardware, while the main framework targets **memory-efficient deployment** on consumer GPUs.

---

## Example Output

After running the complete pipeline, you'll have:

```
analysis_results/
├── eigenvalue_comparison.png      # 4-panel comparison of all methods
├── spectral_vs_raw_summary.png    # Spectral vs raw input comparison
├── training_comparison.png        # VAMPnet training curves
├── projection_comparison.png      # 2D latent projections
├── spectral_heatmap.png          # Power distribution heatmap
├── analysis_report.md            # Human-readable summary
└── analysis_report.json          # Programmatic data

tica_runs/
├── spectral_tica_outputs.npz     # Spectral TICA results
├── spectral_eigenvalues.png
├── spectral_timescales.png
├── spectral_projection.png
├── raw_tica_outputs.npz          # Raw TICA results
├── raw_eigenvalues.png
├── raw_timescales.png
├── raw_projection.png
├── spectral_vs_raw_comparison.png
└── comparison_summary.json
```

---

## Citation

If you use this toolkit, please cite:

```bibtex
@software{lsd_toolkit,
  title={LSD: Lightweight Spectral Dynamics Toolkit},
  year={2024},
  note={Part of Latent Spectral Dynamics project}
}
```

---

## License

MIT License — see parent directory for details.
