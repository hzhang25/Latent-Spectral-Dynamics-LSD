# Processed Data for Prompt Reuse

This folder contains extracted datasets in JSON format for easy reuse in prompt experiments.

## Datasets

### 1. Instruction Pairs (AILiar)

**Source:** "How to Catch an AI Liar" paper ([arXiv:2309.15840](https://arxiv.org/abs/2309.15840))

| File | Description | Examples |
|------|-------------|----------|
| `instruction_pairs_raw.json` | Raw data with all fields | 27 pairs |
| `instruction_pairs_processed.json` | Processed dialogue format | 54 (27 honest + 27 deceptive) |
| `instruction_pairs_inference.json` | **Inference-ready** (no response) | 54 prompts |

**Format (processed):**
```json
{
  "id": "0_honest",
  "pair_id": 0,
  "variant": "honest",
  "system_instruction": "You are a salesperson AI...",
  "user_message": "Human: Is this T-shirt made of cotton?",
  "assistant_response": "AI: I can assure you...",
  "label": "honest"
}
```

### 2. Role Playing

**Source:** Custom scenarios from deception-detection repo

| File | Description | Examples |
|------|-------------|----------|
| `roleplaying_raw.json` | Raw scenarios with all fields | 371 scenarios |
| `roleplaying_processed.json` | Processed dialogue format | 742 (371 honest + 371 deceptive) |
| `roleplaying_inference.json` | **Inference-ready** (no response) | 742 prompts |

**Format (processed):**
```json
{
  "id": "0_honest",
  "pair_id": 0,
  "variant": "honest",
  "dialogue": [
    {"role": "system", "content": "You are a student who..."},
    {"role": "user", "content": "Teacher: Why haven't you..."},
    {"role": "assistant", "content": "Student: I haven't submitted..."}
  ],
  "label": "honest",
  "scenario": "You are a student who..."
}
```

### 3. Sandbagging V2

**Source:** Adapted from Anthropic's Sabotage Evals + WMDP/MMLU datasets

| File | Description |
|------|-------------|
| `sandbagging_v2_prompts.json` | System/user prompt templates |

**Prompt Template:**
- System prompt instructs model to sandbag on specific skills
- Uses step-by-step reasoning with hidden steps
- Labels: "deceptive" (should sandbag) vs "honest" (control)

### 4. Alpaca Control

**Source:** [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) from HuggingFace

| File | Description | Examples |
|------|-------------|----------|
| `alpaca_control.json` | Standard instruction-following | Up to 10,000 |

---

## 🚀 Running Inference with LLaMA 3.3 70B

The `*_inference.json` files are formatted for direct model completion—**the assistant response is NOT included**, allowing the model to generate its own response.

### Inference-Ready Format

```json
{
  "id": "0_honest",
  "messages": [
    {"role": "system", "content": "You are a salesperson AI..."},
    {"role": "user", "content": "Human: Is this T-shirt made of cotton?"}
  ],
  "assistant_prefix": "AI:",
  "label": "honest",
  "metadata": {...}
}
```

The `assistant_prefix` field contains any required prefix for the response (e.g., "Student:", "AI:").

---

### Option 1: vLLM (Recommended for High Throughput)

```python
from vllm import LLM, SamplingParams
import json

# Load inference data
with open('instruction_pairs_inference.json') as f:
    data = json.load(f)

# Initialize model
llm = LLM(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,  # Adjust based on GPU count
    max_model_len=4096,
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    stop=["Human:", "\n\n"]  # Adjust stop tokens as needed
)

# Prepare prompts
prompts = []
for item in data['data']:
    # Format as LLaMA 3 chat template
    messages = item['messages']
    prompt = llm.get_tokenizer().apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # Add assistant prefix if present
    if item.get('assistant_prefix'):
        prompt += item['assistant_prefix'] + " "
    prompts.append(prompt)

# Generate completions
outputs = llm.generate(prompts, sampling_params)

# Process results
results = []
for item, output in zip(data['data'], outputs):
    results.append({
        "id": item['id'],
        "label": item['label'],
        "prompt_messages": item['messages'],
        "generated_response": output.outputs[0].text,
    })

# Save results
with open('inference_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

### Option 2: Hugging Face Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Load model
model_id = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Load inference data
with open('instruction_pairs_inference.json') as f:
    data = json.load(f)

def generate_response(messages, assistant_prefix=""):
    """Generate a response for the given messages."""
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Add assistant prefix if provided
    if assistant_prefix:
        prompt += assistant_prefix + " "
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Run inference
results = []
for item in data['data']:
    response = generate_response(
        item['messages'],
        item.get('assistant_prefix', '')
    )
    results.append({
        "id": item['id'],
        "label": item['label'],
        "generated_response": response,
    })

# Save results
with open('inference_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

### Option 3: Together AI / OpenRouter API

```python
import json
from openai import OpenAI

# Initialize client (Together AI example)
client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.together.xyz/v1"
)

# Load inference data
with open('instruction_pairs_inference.json') as f:
    data = json.load(f)

def generate_response(messages, assistant_prefix=""):
    """Generate response via API."""
    # If there's an assistant prefix, add it as a partial assistant message
    api_messages = messages.copy()
    if assistant_prefix:
        api_messages.append({
            "role": "assistant",
            "content": assistant_prefix
        })
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=api_messages,
        max_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )
    
    return response.choices[0].message.content

# Run inference
results = []
for item in data['data']:
    response = generate_response(
        item['messages'],
        item.get('assistant_prefix', '')
    )
    results.append({
        "id": item['id'],
        "label": item['label'],
        "generated_response": response,
    })

# Save results
with open('inference_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

### Option 4: Ollama (Local)

```python
import json
import ollama

# Load inference data
with open('instruction_pairs_inference.json') as f:
    data = json.load(f)

def generate_response(messages, assistant_prefix=""):
    """Generate response using Ollama."""
    response = ollama.chat(
        model='llama3.3:70b',
        messages=messages,
        options={
            'temperature': 0.7,
            'top_p': 0.9,
            'num_predict': 512,
        }
    )
    
    result = response['message']['content']
    if assistant_prefix and not result.startswith(assistant_prefix):
        result = assistant_prefix + " " + result
    
    return result

# Run inference
results = []
for item in data['data']:
    response = generate_response(
        item['messages'],
        item.get('assistant_prefix', '')
    )
    results.append({
        "id": item['id'],
        "label": item['label'],
        "generated_response": response,
    })
```

---

## 📝 Converting Processed Data to Inference Format

If you need to convert the processed datasets to inference-ready format:

```python
import json

def convert_to_inference_format(input_file, output_file):
    """Convert processed dataset to inference-ready format."""
    with open(input_file) as f:
        data = json.load(f)
    
    inference_data = {
        "dataset_info": {
            **data.get('dataset_info', {}),
            "format": "inference-ready (no assistant response)"
        },
        "num_examples": data['num_examples'],
        "data": []
    }
    
    for item in data['data']:
        # Handle instruction_pairs format
        if 'system_instruction' in item:
            inference_item = {
                "id": item['id'],
                "messages": [
                    {"role": "system", "content": item['system_instruction']},
                    {"role": "user", "content": item['user_message']}
                ],
                "assistant_prefix": item['user_message'].split(':')[0].replace('Human', '').strip() + ":" if ':' in item.get('assistant_response', '') else "",
                "label": item['label'],
                "metadata": {
                    "pair_id": item.get('pair_id'),
                    "variant": item.get('variant'),
                    "expected_response": item.get('assistant_response')  # For evaluation
                }
            }
        # Handle roleplaying format
        elif 'dialogue' in item:
            messages = [m for m in item['dialogue'] if m['role'] != 'assistant']
            assistant_msg = next((m for m in item['dialogue'] if m['role'] == 'assistant'), None)
            prefix = ""
            if assistant_msg and ':' in assistant_msg['content']:
                prefix = assistant_msg['content'].split(':')[0] + ":"
            
            inference_item = {
                "id": item['id'],
                "messages": messages,
                "assistant_prefix": prefix,
                "label": item['label'],
                "metadata": {
                    "pair_id": item.get('pair_id'),
                    "variant": item.get('variant'),
                    "scenario": item.get('scenario'),
                    "expected_response": assistant_msg['content'] if assistant_msg else None
                }
            }
        else:
            continue
            
        inference_data['data'].append(inference_item)
    
    with open(output_file, 'w') as f:
        json.dump(inference_data, f, indent=2)
    
    print(f"Converted {len(inference_data['data'])} examples to {output_file}")

# Convert datasets
convert_to_inference_format('instruction_pairs_processed.json', 'instruction_pairs_inference.json')
convert_to_inference_format('roleplaying_processed.json', 'roleplaying_inference.json')
```

---

## 🔍 Evaluating Generated Responses

After running inference, you can evaluate generated responses against expected behavior:

```python
import json

def evaluate_responses(inference_results_file, expected_labels=True):
    """Basic evaluation of generated responses."""
    with open(inference_results_file) as f:
        results = json.load(f)
    
    stats = {
        "total": len(results),
        "by_label": {}
    }
    
    for result in results:
        label = result.get('label', 'unknown')
        if label not in stats['by_label']:
            stats['by_label'][label] = {"count": 0, "examples": []}
        stats['by_label'][label]['count'] += 1
        
        # Store first few examples
        if len(stats['by_label'][label]['examples']) < 3:
            stats['by_label'][label]['examples'].append({
                "id": result['id'],
                "response_preview": result['generated_response'][:200] + "..."
            })
    
    return stats

# Example usage
stats = evaluate_responses('inference_results.json')
print(json.dumps(stats, indent=2))
```

---

## Re-extraction

To re-extract datasets, run:

```bash
cd deception-detection
python scripts/extract_datasets.py

# Options:
#   --skip-hf          Skip HuggingFace datasets (faster)
#   --alpaca-max N     Limit Alpaca examples (default: 10000)
#   --sandbagging-max N  Limit sandbagging examples (default: 1000)
#   --output-dir PATH  Custom output directory
#   --inference        Generate inference-ready files
```

---

## Labels

All datasets use consistent labeling:
- `"honest"` - Truthful/non-deceptive behavior
- `"deceptive"` - Deceptive/misleading behavior
- `"ambiguous"` - Unclear cases
- `"skip"` - Invalid examples

---

## Hardware Requirements for LLaMA 3.3 70B

| Method | Min VRAM | Notes |
|--------|----------|-------|
| vLLM (FP16) | ~140GB | 4x A100 80GB recommended |
| vLLM (AWQ/GPTQ) | ~40GB | 2x A100 40GB or 1x A100 80GB |
| Transformers (4-bit) | ~40GB | Use `load_in_4bit=True` |
| Ollama | ~40GB | Automatic quantization |
| API | N/A | No local hardware needed |

For lower VRAM setups, consider using quantized models or API endpoints.
