#!/usr/bin/env python3
"""
Minimal latent spectral dynamics demo using a LLaMA 70B model.

Pipeline:
1) Load a LLaMA model (default: meta-llama/Llama-2-70b-hf) with output_hidden_states=True.
2) Grab one hidden layer's activations for a batch of prompts.
3) Compute a simple FFT-based spectrum over the token dimension.
4) Run a lightweight Dynamic Mode Decomposition (DMD) on the same activations.
5) Save spectrum/DMD visualizations and optionally raw arrays.

The goal is clarity and minimal dependencies—no quantization or FlashAttention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Force a non-interactive backend for headless use.
matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal LLaMA latent spectral dynamics demo")
    parser.add_argument(
        "--model-id",
        default="/home/qinanh/eagle/models/Llama-3.3-70B-Instruct",
        help="Hugging Face model id for a 70B LLaMA checkpoint.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Prompts to feed the model. If omitted, sample prompts are used.",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=None,
        help="JSON dataset file with entries containing 'input_str' (overrides --prompts/--prompt-file).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional text file with one prompt per line.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Hidden layer index to analyze (PyTorch index; -1 grabs the final layer).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per prompt after padding/truncation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate (used when --generate is enabled).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for plots and optional NPZ dumps.",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Save hidden states, spectrum, and DMD outputs as NPZ for reuse.",
    )
    parser.add_argument(
        "--save-attentions",
        action="store_true",
        help="Save per-layer attention maps from the forward pass.",
    )
    parser.add_argument(
        "--save-hidden-layers",
        action="store_true",
        help="Save hidden states from multiple layers (default: evenly sample 10 layers).",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="*",
        default=None,
        help="Explicit hidden layer indices to save when --save-hidden-layers is enabled.",
    )
    parser.add_argument(
        "--hidden-layer-count",
        type=int,
        default=10,
        help="Number of layers to sample evenly across the stack if --hidden-layers is not provided.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype. float16 is lightest; float32 is safer but heavier.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed to transformers (e.g., 'auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate continuations for prompts and analyze hidden states on generated sequences.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use sampling during generation (otherwise greedy).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter for generation.",
    )
    parser.add_argument(
        "--analyze-generated-only",
        action="store_true",
        help="When generating, restrict analysis mask to only generated tokens (exclude prompt tokens).",
    )
    parser.add_argument(
        "--record-model-output",
        action="store_true",
        help="Save raw model outputs (logits + ids) for inspection alongside DMD artifacts.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Optional DMD truncation rank; defaults to full rank (min(m-1, d)).",
    )
    return parser.parse_args()


def load_dataset_prompts(dataset_path: Path) -> List[str]:
    data = json.loads(dataset_path.read_text())
    prompts: List[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        text = item.get("input_str")
        if text:
            prompts.append(text)
            continue
        if "input_messages" in item:
            joined = " ".join(msg.get("content", "") for msg in item.get("input_messages", []) if isinstance(msg, dict))
            if joined.strip():
                prompts.append(joined)
    if not prompts:
        raise ValueError(f"No usable prompts found in dataset {dataset_path}")
    return prompts


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.dataset_json:
        return load_dataset_prompts(args.dataset_json)
    if args.prompts:
        return list(args.prompts)
    if args.prompt_file and args.prompt_file.exists():
        return [line.strip() for line in args.prompt_file.read_text().splitlines() if line.strip()]
    # Fallback prompts keep things lightweight while still producing structure in activations.
    return [
        "Describe a stable physical process that shows periodic behavior over time.",
        "Summarize a recent scientific paper about reinforcement learning for robotics.",
        "Write a short, vivid paragraph about a storm forming over the ocean.",
    ]


def resolve_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(model_id: str, dtype: str, device_map: str):
    torch_dtype = resolve_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        output_hidden_states=True,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return tokenizer, model


def select_hidden_layers(
    model, explicit_layers: Optional[Sequence[int]], sample_count: int
) -> List[int]:
    if explicit_layers:
        return sorted(set(int(x) for x in explicit_layers))
    if not sample_count or sample_count <= 0:
        return []
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if not num_layers:
        return []
    sample_count = min(sample_count, num_layers)
    sampled = np.linspace(1, num_layers, num=sample_count, dtype=int).tolist()
    return sorted(set(int(x) for x in sampled))


def detach_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def run_model_forward(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    need_attentions: bool,
):
    with torch.no_grad():
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=need_attentions,
        )


def collect_hidden_states(
    tokenizer,
    model,
    prompts: Sequence[str],
    layer: int,
    max_tokens: int,
    sample_layers: Sequence[int] | None = None,
    need_attentions: bool = False,
    trim_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    encoded = {k: v.to(model.device) for k, v in encoded.items()}
    outputs = run_model_forward(model, encoded["input_ids"], encoded["attention_mask"], need_attentions)
    hidden_states = detach_to_numpy(outputs.hidden_states[layer])
    logits = detach_to_numpy(outputs.logits)
    attention_mask = encoded["attention_mask"].detach().cpu().numpy()
    mask_for_trim = trim_mask if trim_mask is not None else attention_mask

    sampled_hidden = prepare_sampled_hidden(outputs.hidden_states, sample_layers or [], mask_for_trim)
    attention_maps = attention_dict_from_outputs(outputs.attentions) if need_attentions else {}
    return hidden_states, attention_mask, logits, encoded["input_ids"].detach().cpu().numpy(), sampled_hidden, attention_maps


def forward_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int,
    sample_layers: Sequence[int] | None = None,
    need_attentions: bool = False,
    trim_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    outputs = run_model_forward(model, input_ids, attention_mask, need_attentions)
    hidden_states = detach_to_numpy(outputs.hidden_states[layer])
    logits = detach_to_numpy(outputs.logits)
    mask_for_trim = trim_mask if trim_mask is not None else attention_mask.detach().cpu().numpy()
    sampled_hidden = prepare_sampled_hidden(
        outputs.hidden_states,
        sample_layers or [],
        mask_for_trim,
    )
    attention_maps = attention_dict_from_outputs(outputs.attentions) if need_attentions else {}
    return hidden_states, logits, sampled_hidden, attention_maps


def generate_sequences(
    tokenizer,
    model,
    prompts: Sequence[str],
    max_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], np.ndarray]:
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_tokens,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    gen_out = model.generate(**inputs, **gen_kwargs)
    sequences = gen_out.sequences  # (batch, seq_len_generated)
    # Attention mask for generated sequences (pad_token_id padding)
    attn_mask = (sequences != tokenizer.pad_token_id).long()
    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    prompt_lens = inputs["attention_mask"].sum(dim=1).detach().cpu().numpy()
    return sequences, attn_mask, texts, prompt_lens


def trim_and_stack(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # Align all sequences to a common length so FFT/DMD operate cleanly.
    valid_lengths = [int(mask.sum()) for mask in attention_mask]
    target_len = min(valid_lengths)
    trimmed = []
    for hs, mask in zip(hidden_states, attention_mask):
        active_tokens = hs[mask.astype(bool)]
        trimmed.append(active_tokens[:target_len])
    return np.stack(trimmed, axis=0)


def prepare_sampled_hidden(
    hidden_states: Sequence[torch.Tensor] | Sequence[np.ndarray],
    layers: Sequence[int],
    attention_mask: np.ndarray,
) -> dict:
    if not layers:
        return {}
    sampled = {}
    total_layers = len(hidden_states)
    for idx in sorted(set(layers)):
        if idx < -total_layers or idx >= total_layers:
            print(f"[warn] Requested hidden layer {idx} is outside range [-{total_layers}, {total_layers - 1}], skipping.")
            continue
        hs = hidden_states[idx]
        hs_np = detach_to_numpy(hs) if isinstance(hs, torch.Tensor) else hs
        sampled[f"hidden_layer_{idx}"] = trim_and_stack(hs_np, attention_mask)
    return sampled


def attention_dict_from_outputs(attentions: Optional[Sequence[torch.Tensor]]) -> dict:
    if attentions is None:
        return {}
    attn_dict = {}
    for i, attn in enumerate(attentions):
        attn_dict[f"attn_layer_{i}"] = detach_to_numpy(attn)
    return attn_dict


def token_fft(activations: np.ndarray, window: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Token-axis FFT matching original implementation (src/transforms/fourier.py).

    Args:
        activations: (T, d)
        window: Optional window function ("hann", "hamming")
    """
    T, _ = activations.shape
    if window == "hann":
        activations = activations * np.hanning(T)[:, None]
    elif window == "hamming":
        activations = activations * np.hamming(T)[:, None]
    h_hat = np.fft.rfft(activations, axis=0)
    freqs = np.fft.rfftfreq(T)
    return h_hat, freqs


def compute_token_psd(h_hat: np.ndarray) -> np.ndarray:
    """
    PSD_ℓ(ω) = (1/d) Σ_j |h_hat[ω, j]|² — identical to original.
    """
    power = np.abs(h_hat) ** 2
    return np.mean(power, axis=1)


def compute_mean_spectrum(aligned_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply token FFT + PSD per sample, then average over batch.
    Returns frequencies and marginal power spectrum.
    """
    psds = []
    freqs = None
    for sample in aligned_states:
        h_hat, freqs = token_fft(sample)
        psds.append(compute_token_psd(h_hat))
    mean_psd = np.stack(psds, axis=0).mean(axis=0)
    return freqs, mean_psd


def _robust_svd(matrix: np.ndarray):
    """Robust SVD (float64 + jitter) mirroring original DMD util."""
    matrix = np.asarray(matrix, dtype=np.float64)
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        scale = max(np.max(np.abs(matrix)), 1.0)
        jitter = (np.arange(matrix.size, dtype=np.float64).reshape(matrix.shape) - (matrix.size / 2))
        jitter = jitter / np.max(np.abs(jitter)) if np.max(np.abs(jitter)) > 0 else jitter
        matrix = matrix + (scale * 1e-8) * jitter
        return np.linalg.svd(matrix, full_matrices=False)


def run_dmd(aligned_states: np.ndarray, rank: int | None = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Exact DMD using the standard projected Koopman operator (column snapshots), algebraically
    equivalent to the original formulation:

    - Snapshots are token steps as columns: X = h[:, :-1], X' = h[:, 1:], shape (d, m-1).
    - SVD(X) = U Σ V*, with optional truncation rank r.
    - A_tilde = U^* X' V Σ^{-1}  (shape r x r).
    - eig(A_tilde) -> (Λ, W).
    - Modes Φ = X' V Σ^{-1} W  (shape d x r), same as original Vt.T projection.
    """
    token_mean = aligned_states.mean(axis=0).T.astype(np.float64)  # (hidden, seq)
    X = token_mean[:, :-1]       # (d, m-1)
    X_prime = token_mean[:, 1:]  # (d, m-1)

    U, s, Vt = _robust_svd(X)
    max_r = min(X.shape[0], X.shape[1], len(s))
    if rank is None or rank >= max_r:
        r = max_r
    else:
        r = rank
        U, s, Vt = U[:, :r], s[:r], Vt[:r, :]

    eps = np.finfo(s.dtype).eps
    safe_s = np.where(s > eps, s, eps)
    S_inv = np.diag(1.0 / safe_s)

    # Projected Koopman operator (r x r)
    A_tilde = U.T @ X_prime @ Vt.T @ S_inv
    eigvals, W = np.linalg.eig(A_tilde)
    # Full-state modes (d x r)
    modes = X_prime @ Vt.T @ S_inv @ W

    # Reconstruction error in data space: ||X' - U A_tilde U^* X||_F / ||X'||_F
    X_r = U.T @ X              # (r, m-1)
    X_prime_hat = U @ (A_tilde @ X_r)
    rel_err = np.linalg.norm(X_prime - X_prime_hat, ord="fro") / (np.linalg.norm(X_prime, ord="fro") + 1e-12)

    return eigvals, modes, rel_err


def plot_spectrum(freqs: np.ndarray, spectrum: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, spectrum, linewidth=2)
    plt.xlabel("Normalized frequency (1/token)")
    plt.ylabel("Power spectral density")
    plt.title("Hidden-state PSD (token axis)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_dmd_eigs(eigvals: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    colors = np.abs(np.log(np.abs(eigvals)))
    plt.scatter(eigvals.real, eigvals.imag, c=colors, cmap="viridis", s=50, edgecolor="black", linewidth=0.5)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel("Real")
    plt.ylabel("Imag")
    plt.title("DMD eigenvalues (growth/decay vs. oscillation)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_npz(out_path: Path, **arrays) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def main():
    args = parse_args()
    prompts = load_prompts(args)
    print(f"Using {len(prompts)} prompt(s). Example: {prompts[0]!r}")

    tokenizer, model = load_model_and_tokenizer(args.model_id, args.dtype, args.device_map)
    sampled_layers = select_hidden_layers(model, args.hidden_layers, args.hidden_layer_count) if args.save_hidden_layers else []
    if sampled_layers:
        print(f"Saving hidden states for layers: {sampled_layers}")
    if args.generate:
        sequences, attn_mask_t, gen_texts, prompt_lens = generate_sequences(
            tokenizer,
            model,
            prompts,
            max_tokens=args.max_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        attn_mask = attn_mask_t.detach().cpu().numpy()
        analysis_mask = attn_mask.copy()
        if args.analyze_generated_only:
            analysis_mask = np.zeros_like(attn_mask)
            for i, plen in enumerate(prompt_lens):
                analysis_mask[i, int(plen):] = attn_mask[i, int(plen):]
        hidden_states, logits, sampled_hidden, attention_maps = forward_hidden_states(
            model,
            sequences,
            attn_mask_t,
            args.layer,
            sample_layers=sampled_layers,
            need_attentions=args.save_attentions,
            trim_mask=analysis_mask,
        )
        input_ids = sequences.detach().cpu().numpy()
        decoded_texts = gen_texts
    else:
        hidden_states, attn_mask, logits, input_ids, sampled_hidden, attention_maps = collect_hidden_states(
            tokenizer,
            model,
            prompts,
            args.layer,
            args.max_tokens,
            sample_layers=sampled_layers,
            need_attentions=args.save_attentions,
        )
        analysis_mask = attn_mask
        decoded_texts = prompts

    aligned = trim_and_stack(hidden_states, analysis_mask)
    print(f"Aligned activations shape: {aligned.shape} (batch, seq, hidden)")

    freqs, spectrum = compute_mean_spectrum(aligned)
    eigvals, modes, dmd_rel_err = run_dmd(aligned, rank=args.rank)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spectrum_path = args.out_dir / "spectrum.png"
    dmd_path = args.out_dir / "dmd_eigs.png"
    plot_spectrum(freqs, spectrum, spectrum_path)
    plot_dmd_eigs(eigvals, dmd_path)

    if args.save_npz:
        npz_path = args.out_dir / "activations_and_dmd.npz"
        npz_payload = {
            "hidden_states": aligned,
            "freqs": freqs,
            "spectrum": spectrum,
            "eigvals": eigvals,
            "dmd_modes": modes,
            "dmd_rel_err": dmd_rel_err,
        }
        save_npz(npz_path, **npz_payload)
        print(f"Saved NPZ to {npz_path}")
    if sampled_hidden:
        sampled_path = args.out_dir / "sampled_hidden_layers.npz"
        save_npz(sampled_path, **sampled_hidden)
        print(f"Saved sampled hidden states to {sampled_path}")
    if attention_maps:
        attn_path = args.out_dir / "attentions.npz"
        save_npz(attn_path, **attention_maps)
        print(f"Saved attention maps to {attn_path}")
    if args.record_model_output:
        model_npz = args.out_dir / "model_outputs.npz"
        save_npz(
            model_npz,
            input_ids=input_ids,
            attention_mask=analysis_mask,
            logits=logits,
        )
        decoded_path = args.out_dir / "model_outputs.txt"
        decoded_path.write_text("\n\n".join(decoded_texts))
        print(f"Recorded raw model outputs to {model_npz} and decoded texts to {decoded_path}")

    dominant_freq = freqs[np.argmax(spectrum)]
    top_eig = eigvals[np.argmax(np.abs(eigvals))]
    print(f"Dominant spectral freq: {dominant_freq:.4f}")
    print(f"Top DMD eigenvalue: {top_eig.real:.4f} + {top_eig.imag:.4f}j")
    print(f"DMD relative reconstruction error ||X' - A X||_F / ||X'||_F: {dmd_rel_err:.3e}")
    print(f"Plots saved to: {spectrum_path} and {dmd_path}")


if __name__ == "__main__":
    main()
