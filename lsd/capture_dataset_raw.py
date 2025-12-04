#!/usr/bin/env python3
"""
Capture raw hidden activations and attention maps for each prompt in a dataset.

- Runs every prompt in the JSON dataset.
- Repeats each prompt N times (default 5).
- Saves the first `layer-count` transformer layer hidden states (post-embedding) and attention maps.
- No spectrum/DMD computation is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture raw hidden states and attentions for a dataset of prompts")
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("/home/qinanh/eagle/class/Latent-Spectral-Dynamics-Minimal/out_simple_4_many.json"),
        help="JSON file containing prompts (expects 'input_str' or 'input_messages').",
    )
    parser.add_argument(
        "--model-id",
        default="/home/qinanh/eagle/models/Llama-3.3-70B-Instruct",
        help="Hugging Face model id/path.",
    )
    parser.add_argument(
        "--runs-per-prompt",
        type=int,
        default=5,
        help="Number of times to run each prompt.",
    )
    parser.add_argument(
        "--layer-count",
        type=int,
        default=10,
        help="Number of early transformer layers to save (post-embedding).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Padding/truncation length for tokenization.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model dtype.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for model loading (e.g., auto/cuda/cpu).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/raw_captures"),
        help="Root directory for saved activations/attentions.",
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
            joined = " ".join(
                msg.get("content", "") for msg in item.get("input_messages", []) if isinstance(msg, dict)
            )
            if joined.strip():
                prompts.append(joined)
    if not prompts:
        raise ValueError(f"No usable prompts found in dataset {dataset_path}")
    return prompts


def resolve_dtype(name: str):
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def select_layers(num_hidden_layers: int, count: int) -> List[int]:
    return list(range(min(count, num_hidden_layers)))


def detach_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def main():
    args = parse_args()
    prompts = load_dataset_prompts(args.dataset_json)
    print(f"Loaded {len(prompts)} prompts from {args.dataset_json}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
        output_hidden_states=True,
        output_attentions=True,
    )
    model.eval()
    torch.set_grad_enabled(False)

    total_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 0) or 0
    layer_indices = select_layers(total_layers, args.layer_count)
    print(f"Capturing hidden layers (post-embedding): {layer_indices}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for p_idx, prompt in enumerate(prompts):
        prompt_dir = args.out_dir / f"prompt_{p_idx:03d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "prompt.txt").write_text(prompt)
        for run_idx in range(1, args.runs_per_prompt + 1):
            run_dir = prompt_dir / f"run_{run_idx:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            enc = tokenizer(
                [prompt],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_tokens,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            with torch.no_grad():
                outputs = model(
                    **enc,
                    output_hidden_states=True,
                    output_attentions=True,
                )

            # hidden_states[0] is embedding; layer i is hidden_states[i+1]
            hidden_payload = {}
            for layer in layer_indices:
                hs = outputs.hidden_states[layer + 1]
                hidden_payload[f"layer_{layer}"] = detach_to_numpy(hs)
            attn_payload = {}
            for layer in layer_indices:
                if layer < len(outputs.attentions):
                    attn_payload[f"layer_{layer}"] = detach_to_numpy(outputs.attentions[layer])

            np.savez_compressed(run_dir / "hidden_layers.npz", **hidden_payload)
            np.savez_compressed(run_dir / "attentions.npz", **attn_payload)
            np.savez_compressed(
                run_dir / "inputs.npz",
                input_ids=detach_to_numpy(enc["input_ids"]),
                attention_mask=detach_to_numpy(enc["attention_mask"]),
            )
            print(f"Saved prompt {p_idx} run {run_idx} to {run_dir}")


if __name__ == "__main__":
    main()
