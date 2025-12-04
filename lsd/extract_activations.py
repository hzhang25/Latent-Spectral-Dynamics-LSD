#!/usr/bin/env python3
"""
Extract hidden state activations from LLaMA 3.3 70B for multiple deception datasets.

Datasets:
1. instruction_pairs - AILiar honest/deceptive pairs (54 examples)
2. roleplaying - Role-play scenarios (742 examples)
3. sandbagging - Sandbagging detection prompts (requires WMDP/MMLU questions)
4. alpaca - Standard instruction-following (ground truth control)

Output Structure:
    lsd/data/
    ├── extraction_metadata.json
    ├── instruction_pairs/
    │   ├── prompts.json
    │   ├── 0_honest/
    │   │   ├── hidden_layers.npz
    │   │   └── prompt.json
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

Example:
python extract_activations.py \
    --model-id meta-llama/Llama-3.3-70B-Instruct \
    --data-dir processed_data \
    --out-dir data \
    --layer-step 5 \
    --datasets instruction_pairs roleplaying alpaca
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract activations from LLaMA 3.3 70B for deception datasets")
    p.add_argument("--model-id", type=str, 
                   default="meta-llama/Llama-3.3-70B-Instruct",
                   help="HuggingFace model ID")
    p.add_argument("--data-dir", type=Path, default=Path("processed_data"),
                   help="Directory containing processed JSON files")
    p.add_argument("--out-dir", type=Path, default=Path("data"),
                   help="Output directory for activations (default: data/)")
    p.add_argument("--datasets", nargs="+", 
                   default=["instruction_pairs", "roleplaying", "alpaca"],
                   choices=["instruction_pairs", "roleplaying", "sandbagging", "alpaca"],
                   help="Datasets to process")
    p.add_argument("--layer-step", type=int, default=5,
                   help="Layer step for saving (e.g., 5 means layers 0, 5, 10, ...)")
    p.add_argument("--max-layers", type=int, default=None,
                   help="Maximum number of layers to save (None = all)")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Maximum sequence length for tokenization")
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["float16", "bfloat16", "float32"],
                   help="Model dtype")
    p.add_argument("--device-map", type=str, default="auto",
                   help="Device map for model loading")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Maximum samples per dataset (None = all)")
    p.add_argument("--alpaca-max", type=int, default=500,
                   help="Maximum Alpaca samples to use")
    p.add_argument("--sandbagging-questions", type=Path, default=None,
                   help="Path to WMDP/MMLU questions JSON for sandbagging")
    p.add_argument("--resume", action="store_true",
                   help="Skip already processed samples")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed progress")
    return p.parse_args()


def resolve_dtype(name: str):
    """Convert dtype string to torch dtype."""
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(model_id: str, dtype: str, device_map: str):
    """Load LLaMA model and tokenizer."""
    print(f"Loading model: {model_id}")
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
    
    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", 80)
    print(f"Model loaded: {num_layers} layers, dtype={dtype}")
    
    return tokenizer, model, num_layers


def format_messages_to_prompt(
    tokenizer,
    messages: List[Dict[str, str]],
    assistant_prefix: str = ""
) -> str:
    """Format messages using LLaMA chat template."""
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Add assistant prefix if provided
    if assistant_prefix:
        prompt += assistant_prefix + " "
    
    return prompt


def extract_hidden_states(
    model,
    tokenizer,
    prompt: str,
    max_seq_len: int,
    layer_indices: List[int]
) -> Dict[str, np.ndarray]:
    """Extract hidden states from specified layers."""
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extract hidden states
    hidden_states = {}
    for layer_idx in layer_indices:
        if layer_idx < len(outputs.hidden_states):
            hs = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
            hidden_states[f"layer_{layer_idx}"] = hs.cpu().float().numpy()
    
    return hidden_states


def load_instruction_pairs(data_dir: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load instruction pairs dataset."""
    file_path = data_dir / "instruction_pairs_inference.json"
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return []
    
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    
    samples = data.get("data", [])
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Loaded {len(samples)} instruction pairs samples")
    return samples


def load_roleplaying(data_dir: Path, max_samples: Optional[int] = None) -> List[Dict]:
    """Load roleplaying dataset."""
    file_path = data_dir / "roleplaying_inference.json"
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return []
    
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    
    samples = data.get("data", [])
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"Loaded {len(samples)} roleplaying samples")
    return samples


def load_sandbagging(
    data_dir: Path, 
    questions_path: Optional[Path] = None,
    max_samples: Optional[int] = None
) -> List[Dict]:
    """Load sandbagging dataset with WMDP/MMLU questions."""
    prompts_path = data_dir / "sandbagging_v2_prompts.json"
    if not prompts_path.exists():
        print(f"Warning: {prompts_path} not found")
        return []
    
    with open(prompts_path, encoding="utf-8") as f:
        prompts_data = json.load(f)
    
    system_template = prompts_data["prompts"]["system_prompt_template"]
    user_template = prompts_data["prompts"]["user_prompt_template"]
    skill_desc = prompts_data["prompts"]["skill_descriptions"].get("wmdp", "")
    
    # If no questions file, generate placeholder samples
    if questions_path is None or not questions_path.exists():
        print("Warning: No sandbagging questions file provided. Using placeholder.")
        # Create a few placeholder samples
        placeholder_questions = [
            {
                "question": "What is the chemical formula for water?\nA) H2O\nB) CO2\nC) NaCl\nD) O2",
                "answer": "A",
                "id": "placeholder_0"
            },
            {
                "question": "What programming language is commonly used for machine learning?\nA) HTML\nB) Python\nC) CSS\nD) SQL",
                "answer": "B",
                "id": "placeholder_1"
            }
        ]
        questions = placeholder_questions
    else:
        with open(questions_path, encoding="utf-8") as f:
            questions = json.load(f)
    
    if max_samples:
        questions = questions[:max_samples]
    
    samples = []
    for i, q in enumerate(questions):
        q_text = q.get("question", q.get("text", ""))
        q_id = q.get("id", str(i))
        
        # Sandbagging variant (deceptive - should sandbag)
        system_content = system_template.format(skill=skill_desc)
        user_content = user_template.format(question=q_text)
        
        samples.append({
            "id": f"{q_id}_sandbagging",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            "assistant_prefix": "",
            "label": "deceptive",
            "metadata": {"variant": "sandbagging", "original_id": q_id}
        })
        
        # Control variant (honest - normal answering)
        control_system = "You are a helpful AI assistant. Answer questions accurately and to the best of your ability."
        samples.append({
            "id": f"{q_id}_control",
            "messages": [
                {"role": "system", "content": control_system},
                {"role": "user", "content": user_content}
            ],
            "assistant_prefix": "",
            "label": "honest",
            "metadata": {"variant": "control", "original_id": q_id}
        })
    
    print(f"Loaded {len(samples)} sandbagging samples ({len(questions)} questions × 2 variants)")
    return samples


def load_alpaca(max_samples: int = 500) -> List[Dict]:
    """Load Alpaca dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: 'datasets' library not installed. Skipping Alpaca.")
        print("Install with: pip install datasets")
        return []
    
    print(f"Loading Alpaca dataset (max {max_samples} samples)...")
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except Exception as e:
        print(f"Warning: Failed to load Alpaca dataset: {e}")
        return []
    
    samples = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break
        
        # Build instruction with optional input
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        
        if input_text:
            user_content = f"{instruction}\n\nInput: {input_text}"
        else:
            user_content = instruction
        
        samples.append({
            "id": f"alpaca_{i}",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content}
            ],
            "assistant_prefix": "",
            "label": "honest",  # Alpaca is ground truth (no deception)
            "metadata": {
                "variant": "control",
                "expected_output": item.get("output", "")
            }
        })
    
    print(f"Loaded {len(samples)} Alpaca samples")
    return samples


def process_dataset(
    dataset_name: str,
    samples: List[Dict],
    tokenizer,
    model,
    layer_indices: List[int],
    out_dir: Path,
    max_seq_len: int,
    resume: bool = False,
    verbose: bool = False
):
    """Process a dataset and save activations."""
    dataset_dir = out_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Save prompts metadata
    prompts_file = dataset_dir / "prompts.json"
    prompts_meta = {
        "dataset": dataset_name,
        "num_samples": len(samples),
        "layer_indices": layer_indices,
        "samples": []
    }
    
    print(f"\nProcessing {dataset_name}: {len(samples)} samples")
    
    for i, sample in enumerate(samples):
        sample_id = sample.get("id", str(i))
        sample_dir = dataset_dir / sample_id
        
        # Check if already processed
        if resume and sample_dir.exists() and (sample_dir / "hidden_layers.npz").exists():
            if verbose:
                print(f"  Skipping {sample_id} (already exists)")
            prompts_meta["samples"].append({
                "id": sample_id,
                "label": sample.get("label"),
                "dir": str(sample_dir.relative_to(out_dir))
            })
            continue
        
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Format prompt
        messages = sample.get("messages", [])
        assistant_prefix = sample.get("assistant_prefix", "")
        prompt = format_messages_to_prompt(tokenizer, messages, assistant_prefix)
        
        # Extract hidden states
        try:
            hidden_states = extract_hidden_states(
                model, tokenizer, prompt, max_seq_len, layer_indices
            )
            
            # Save hidden states
            np.savez_compressed(sample_dir / "hidden_layers.npz", **hidden_states)
            
            # Save prompt info
            prompt_info = {
                "id": sample_id,
                "label": sample.get("label"),
                "messages": messages,
                "assistant_prefix": assistant_prefix,
                "formatted_prompt": prompt,
                "metadata": sample.get("metadata", {})
            }
            with open(sample_dir / "prompt.json", "w", encoding="utf-8") as f:
                json.dump(prompt_info, f, indent=2, ensure_ascii=False)
            
            prompts_meta["samples"].append({
                "id": sample_id,
                "label": sample.get("label"),
                "dir": str(sample_dir.relative_to(out_dir))
            })
            
            if verbose or (i + 1) % 10 == 0:
                print(f"  [{i+1}/{len(samples)}] Processed {sample_id}")
                
        except Exception as e:
            print(f"  Error processing {sample_id}: {e}")
            continue
    
    # Save prompts metadata
    with open(prompts_file, "w", encoding="utf-8") as f:
        json.dump(prompts_meta, f, indent=2, ensure_ascii=False)
    
    print(f"  Saved {len(prompts_meta['samples'])} samples to {dataset_dir}")


def main():
    args = parse_args()
    
    # Setup output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    tokenizer, model, num_layers = load_model_and_tokenizer(
        args.model_id, args.dtype, args.device_map
    )
    
    # Determine layer indices
    if args.max_layers:
        max_layer = min(args.max_layers * args.layer_step, num_layers)
    else:
        max_layer = num_layers
    
    layer_indices = list(range(0, max_layer, args.layer_step))
    print(f"Extracting layers: {layer_indices}")
    
    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        if dataset_name == "instruction_pairs":
            samples = load_instruction_pairs(args.data_dir, args.max_samples)
        elif dataset_name == "roleplaying":
            samples = load_roleplaying(args.data_dir, args.max_samples)
        elif dataset_name == "sandbagging":
            samples = load_sandbagging(
                args.data_dir, 
                args.sandbagging_questions,
                args.max_samples
            )
        elif dataset_name == "alpaca":
            samples = load_alpaca(args.alpaca_max)
        else:
            print(f"Unknown dataset: {dataset_name}")
            continue
        
        if not samples:
            print(f"No samples found for {dataset_name}")
            continue
        
        process_dataset(
            dataset_name=dataset_name,
            samples=samples,
            tokenizer=tokenizer,
            model=model,
            layer_indices=layer_indices,
            out_dir=args.out_dir,
            max_seq_len=args.max_seq_len,
            resume=args.resume,
            verbose=args.verbose
        )
    
    # Save extraction metadata
    meta = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "layer_indices": layer_indices,
        "layer_step": args.layer_step,
        "max_seq_len": args.max_seq_len,
        "datasets": args.datasets,
        "num_model_layers": num_layers,
    }
    with open(args.out_dir / "extraction_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Extraction complete!")
    print(f"  Output directory: {args.out_dir}")
    print(f"  Datasets processed: {args.datasets}")
    print(f"  Layers extracted: {layer_indices}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

