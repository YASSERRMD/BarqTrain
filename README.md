# BarqTrain: High-Performance LLM Fine-Tuning Accelerator

**BarqTrain** is a high-performance fine-tuning library designed to bypass the memory and compute bottlenecks of standard Hugging Face/PyTorch training. By combining **Rust** for GIL-free data pipelines, **CUDA C++** for custom fused kernels, and **Python** for the user-facing API, BarqTrain achieves massive speedups and memory reductions.

## Features

- **Fused RMSNorm**: Single-kernel normalization reducing HBM bandwidth by 3-4x
- **Chunked Cross-Entropy**: Avoids logit materialization, saving up to 60% VRAM for large vocabularies
- **FlashAttention Integration**: `patch_model(...)` selects `flash_attention_2` when available and falls back to PyTorch SDPA otherwise
- **Fused LoRA**: Single-pass GEMM combining base weights and LoRA adapters
- **Rust Data Pipeline**: Native causal-LM sequence packing with zero GIL contention
- **Paged KV Cache**: CUDA-backed paged cache append path that avoids `torch.cat` cache growth during generation
- **Paged Optimizer Support**: Switch between `AdamW`, `PagedAdamW32bit`, and `PagedAdamW8bit`

## Current Status

BarqTrain is already a useful native acceleration layer, but it is not yet a full native memory-management stack for LLM serving.

| Area | Native Status Today | Primary Benefit Today | Biggest Missing Piece |
|------|----------------------|-----------------------|-----------------------|
| RMSNorm | CUDA kernel shipped | lower kernel overhead | deeper fusion into larger blocks |
| Cross-entropy | CUDA chunked loss shipped | lower training memory and better training throughput | more fused projection-plus-loss work |
| Data path | Rust packing shipped | lower Python overhead and less padding waste | padding-free end-to-end training path |
| Attention | backend selection shipped | faster attention when FlashAttention is available | deeper native attention fusion |
| Inference memory | phase 1+2 shipped | resident/peak accounting plus paged KV-cache generation | quantized/offloaded KV-cache and page-table compaction |
| Optimizer memory | wrapper-level | optional training-memory savings | native optimizer-state control |

## Research-Backed Roadmap

The next memory-focused work is tracked as a phased native implementation plan in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

The roadmap is based on the most relevant public work for this problem space:

- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [KIVI KV-cache quantization](https://arxiv.org/abs/2402.02750)
- [Cut Cross Entropy](https://arxiv.org/abs/2411.09009)
- [Padding-Free Transformer](https://huggingface.co/blog/mayank-mishra/padding-free-transformer)
- [PyTorch activation checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)

## Installation

### Google Colab (NVIDIA GPU) 🚀

BarqTrain requires a native Rust build during install. The Colab flow below installs
the Rust toolchain first and fails immediately if `barqtrain_rs` does not build.

**Step 1 — Select a GPU runtime**

`Runtime → Change runtime type → Hardware accelerator → T4 / A100 / L4`

**Step 2 — Clone & install**

```python
# Cell 1 – clone, install, and verify (run once per Colab session)
!git clone -b codex/phase_24-colab-branch-fix https://github.com/YASSERRMD/BarqTrain.git
%cd BarqTrain
from pathlib import Path
from urllib.request import urlopen
import os
if not Path(os.path.expanduser("~/.cargo/bin/cargo")).exists():
    Path("/tmp/rustup-init.sh").write_text(urlopen("https://sh.rustup.rs").read().decode("utf-8"))
    !sh /tmp/rustup-init.sh -y
import os
os.environ["PATH"] = f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}"
!python -m pip install --upgrade pip setuptools wheel setuptools-rust
!python -m pip install ninja packaging datasets accelerate peft trl
!python -m pip install -e . --no-build-isolation

# Colab doesn't always reload .pth files in a running session —
# this sys.path line makes the import work immediately.
import sys, importlib, importlib.util
sys.path.insert(0, '/content/BarqTrain/python')
importlib.invalidate_caches()
assert importlib.util.find_spec("barqtrain_rs"), "barqtrain_rs did not build"
import barqtrain._ffi as ffi
assert ffi.load_rust_backend() is not None, "barqtrain_rs did not load"

import barqtrain
from barqtrain import patch_model
print(f"BarqTrain {barqtrain.__version__} loaded ✓")
```

**Step 3 — Restart the Colab runtime/session**

After the native Rust/CUDA build, restart the runtime before training or benchmarking.

`Runtime -> Restart session`

**Step 4 — Verify native runtime loading after restart**

```python
%cd /content/BarqTrain
import sys, importlib, importlib.util
sys.path.insert(0, "/content/BarqTrain/python")
importlib.invalidate_caches()

import barqtrain
import barqtrain._ffi as ffi

print("barqtrain_rs spec:", bool(importlib.util.find_spec("barqtrain_rs")))
print("barqtrain_cuda spec:", bool(importlib.util.find_spec("barqtrain_cuda")))
print("rust runtime load:", ffi.load_rust_backend() is not None)
print("cuda runtime load:", ffi.load_cuda_backend() is not None)

assert ffi.load_rust_backend() is not None
assert ffi.load_cuda_backend() is not None
print(f"BarqTrain {barqtrain.__version__} runtime verification: OK")
```

**Step 5 — (Optional) Compile CUDA kernels for maximum performance**

The CUDA kernels give the biggest speedups. Compilation takes ~2 min on Colab.

```python
# Cell 3 – compile CUDA kernels (T4 / A100 / L4 / V100 all supported)
!BARQTRAIN_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation

# Verify the CUDA extension loaded
import importlib.util
assert importlib.util.find_spec("barqtrain_cuda"), "barqtrain_cuda did not build"
import barqtrain._ffi as ffi
assert ffi.load_cuda_backend() is not None, "barqtrain_cuda did not load"
print("CUDA extension loaded ✓")
```

**Step 4 — Full Colab example: fine-tune Llama-3 on a custom dataset**

```python
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from barqtrain import patch_model

# Load model (bfloat16 to fit in Colab VRAM)
model_id = "meta-llama/Meta-Llama-3-8B"   # swap for any HF model
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Causal LMs need a pad token; use EOS if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Apply BarqTrain fused kernels (RMSNorm, FlashAttention, chunked CE)
patch_model(model)
print("BarqTrain patches applied ✓")

# Load and tokenize dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    # labels must be provided — for causal LM they equal input_ids.
    # DataCollatorForLanguageModeling will shift them internally.
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
tokenized.set_format("torch")

# DataCollator for causal LM — sets padding positions in labels to -100
# so they are ignored in the loss, and shifts labels by one position.
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Train
args = TrainingArguments(
    output_dir="./barqtrain-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator,
)
trainer.train()
```

> **Tip:** On a free Colab T4 (16 GB VRAM), use `per_device_train_batch_size=1`
> and `gradient_accumulation_steps=8`. On A100 (40 GB), you can use batch size 4-8.

**Verify GPU is active and VRAM usage:**

```python
!nvidia-smi
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

---

### Ready-to-use Colab notebooks (in this repo)

- Training + Inference (`examples/barqtrain_training_inference_colab.ipynb`):
  <a href="https://colab.research.google.com/github/YASSERRMD/BarqTrain/blob/codex/phase_24-colab-branch-fix/examples/barqtrain_training_inference_colab.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- Benchmark Comparison (`examples/barqtrain_benchmark_comparison_colab.ipynb`):
  <a href="https://colab.research.google.com/github/YASSERRMD/BarqTrain/blob/codex/phase_24-colab-branch-fix/examples/barqtrain_benchmark_comparison_colab.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  Both notebooks now make the post-build restart explicit and include a runtime verification step for `barqtrain_rs` and `barqtrain_cuda`.

### Local (from source)

```bash
# Clone the repository
git clone -b codex/phase_24-colab-branch-fix https://github.com/YASSERRMD/BarqTrain.git
cd BarqTrain

# Install Python package and native Rust build dependencies
curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"
python -m pip install --upgrade pip setuptools wheel setuptools-rust
python -m pip install -e . --no-build-isolation

# Verify the Rust extension was built
python - <<'PY'
import importlib.util
assert importlib.util.find_spec("barqtrain_rs"), "barqtrain_rs did not build"
PY

# Optional: build CUDA kernels (requires NVIDIA GPU + CUDA toolkit)
BARQTRAIN_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation

# Headless/Docker CUDA build without a visible GPU:
BARQTRAIN_CUDA_ARCH_LIST=7.5 BARQTRAIN_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation
```

### Training Helpers

BarqTrain exposes thin helpers for the optimized training path:

- `patch_model(model)`: patches supported RMSNorm layers, configures the best attention backend available, routes compatible decoder-only training with labels through chunked loss, and wraps compatible CUDA generation calls to inject BarqTrain's paged KV cache automatically
- `patch_inference(model)`: inference-only patching path for decode benchmarks and low-memory generation experiments
- `PackedCausalLMDataCollator(...)`: uses the Rust packing backend for denser causal-LM batches
- `create_optimizer(...)`: selects `adamw`, `paged_adamw_32bit`, or `paged_adamw_8bit`
- `create_paged_kv_cache(...)`: explicitly create a paged KV cache when you want to control decode capacity yourself

```python
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from barqtrain import (
    PackedCausalLMDataCollator,
    create_optimizer,
    patch_model,
)

model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id)
patch_model(model)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:256]")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512, padding=False)

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
collator = PackedCausalLMDataCollator(
    max_length=512,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
dataloader = DataLoader(tokenized, batch_size=4, shuffle=True, collate_fn=collator)

optimizer = create_optimizer(
    model.parameters(),
    lr=1e-5,
    optimizer_name="paged_adamw_32bit",
)
```

### Inference Helpers

For explicit decode-cache control, you can create and pass a paged KV cache yourself:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from barqtrain import create_paged_kv_cache, patch_inference

model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
patch_inference(model)

inputs = tokenizer("Explain paged KV caches.", return_tensors="pt").to("cuda")
cache = create_paged_kv_cache(model, max_batch_size=1, max_cache_len=256, page_size=16)
outputs = model.generate(**inputs, max_new_tokens=64, past_key_values=cache)
```


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from barqtrain import patch_model

# Load your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Apply BarqTrain optimizations
patch_model(model)

# Train as usual - BarqTrain kernels are automatically used
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="./llama2-ft",
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()
```

## Performance

BarqTrain should be evaluated in two separate ways:

- **Training path**: chunked loss and packed data can reduce activation or loss-path pressure and improve throughput.
- **Inference path**: the current native stack now measures resident VRAM separately and ships a paged KV cache for decode-time cache growth, but total peak VRAM on short full-weight runs can still be dominated by model residency.

That distinction matters. A faster inference benchmark does not automatically mean lower total VRAM if model weights remain the largest memory bucket.

The current inference benchmark now runs two profiles:

1. `throughput_short`: short prompt + short decode
2. `memory_long`: long prompt + longer decode to stress KV-cache growth

The current inference benchmark should be read using these numbers together:

1. `resident_vram_mb`: memory already committed before decode
2. `generation_overhead_mb`: memory added by decode-time work
3. `paged_kv_cache`: whether BarqTrain's paged cache path was actually active
4. `last_token_logits_only`: whether decode-token logits-only generation was requested

The roadmap in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) therefore prioritizes:

1. native memory accounting
2. paged KV-cache
3. quantized KV-cache
4. fused projection-plus-loss improvements
5. padding-free packed training
6. activation-memory control

## Architecture

```
barqtrain/
├── python/              # Python package
│   └── barqtrain/
│       ├── __init__.py
│       ├── patch_models.py   # HF model patching
│       ├── kv_cache.py       # Paged KV-cache integration
│       ├── ops.py            # Custom ops wrappers
│       ├── lora.py           # LoRA implementations
│       └── data.py           # Rust data pipeline wrappers
├── csrc/                # CUDA kernels
│   ├── src/bindings.cpp      # PyBind11 bindings
│   └── kernels/
│       ├── rmsnorm.cu        # Fused RMSNorm
│       ├── flash_attention.cu # FlashAttention + RoPE
│       ├── chunked_cross_entropy.cu # Chunked CE
│       ├── paged_kv_cache.cu # Paged KV-cache append kernel
│       └── lora.cu           # Fused LoRA GEMM
├── rust/                # Rust data pipeline
│   └── src/lib.rs       # PyO3 bindings
└── tests/               # Numerical parity tests
```

## Supported Models

- Frontier/general LLMs: Llama (1, 2, 3, 4) - RMSNorm fused patch support
- Enterprise/production: Qwen family (Qwen2, Qwen2-MoE, Qwen3), IBM Granite, AI21 Jamba - RMSNorm fused patch support
- Reasoning/open frontier: DeepSeek family (V2, V3), Mistral/Mixtral - RMSNorm fused patch support
- Efficient edge models: Microsoft Phi family (Phi-3, Phi-4 Multimodal), Liquid LFM2/LFM2.5 (including 1.2B) - RMSNorm fused patch support
- Research/open science: OLMo family (OLMo2, OLMoE), Google Gemma family (Gemma, Gemma2, Gemma3) - RMSNorm fused patch support
- Any HF model with RMSNorm - Partial support

## Development

### Running Tests

```bash
# Run numerical parity tests
pytest tests/test_rmsnorm.py -v

# Run benchmarks
python -m barqtrain.benchmarks.baseline --model tinyllama --steps 100

# Benchmark Rust packing + paged optimizer
python -m barqtrain.benchmarks.baseline \
  --model tinyllama \
  --steps 100 \
  --use-packing \
  --optimizer paged_adamw_32bit
```

### Building Documentation

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Build docs (if available)
cd docs && make html
```

## Citation

If you use BarqTrain in your research, please cite:

```bibtex
@software{barqtrain2024,
  title={BarqTrain: High-Performance LLM Fine-Tuning Accelerator},
  author={BarqTrain Contributors},
  year={2024},
  url={https://github.com/YASSERRMD/BarqTrain}
}
```

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by [Unsloth](https://github.com/unslothai/unsloth) and [Liger Kernel](https://github.com/linkedin/Liger-Kernel)
- Uses [PyTorch](https://pytorch.org/), [CUDA](https://developer.nvidia.com/cuda-toolkit), [Rust](https://www.rust-lang.org/), and [PyO3](https://pyo3.rs/)

## Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## Disclaimer

This project is under active development. APIs and implementations may change between versions.
