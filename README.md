# BarqTrain: High-Performance LLM Fine-Tuning Accelerator

**BarqTrain** is a high-performance fine-tuning library designed to bypass the memory and compute bottlenecks of standard Hugging Face/PyTorch training. By combining **Rust** for GIL-free data pipelines, **CUDA C++** for custom fused kernels, and **Python** for the user-facing API, BarqTrain achieves massive speedups and memory reductions.

## Features

- **Fused RMSNorm**: Single-kernel normalization reducing HBM bandwidth by 3-4x
- **Chunked Cross-Entropy**: Avoids logit materialization, saving up to 60% VRAM for large vocabularies
- **FlashAttention Integration**: `patch_model(...)` selects `flash_attention_2` when available and falls back to PyTorch SDPA otherwise
- **Fused LoRA**: Single-pass GEMM combining base weights and LoRA adapters
- **Rust Data Pipeline**: Native causal-LM sequence packing with zero GIL contention
- **Paged Optimizer Support**: Switch between `AdamW`, `PagedAdamW32bit`, and `PagedAdamW8bit`

## Installation

### Google Colab (NVIDIA GPU) 🚀

BarqTrain requires a native Rust build during install. The Colab flow below installs
the Rust toolchain first and fails immediately if `barqtrain_rs` does not build.

**Step 1 — Select a GPU runtime**

`Runtime → Change runtime type → Hardware accelerator → T4 / A100 / L4`

**Step 2 — Clone & install**

```python
# Cell 1 – clone, install, and verify (run once per Colab session)
!git clone https://github.com/YASSERRMD/BarqTrain.git
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

> **⚠️ Note:** If you restart the Colab runtime, just re-run this cell.

**Step 3 — (Optional) Compile CUDA kernels for maximum performance**

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
  <a href="https://colab.research.google.com/github/YASSERRMD/BarqTrain/blob/main/examples/barqtrain_training_inference_colab.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- Benchmark Comparison (`examples/barqtrain_benchmark_comparison_colab.ipynb`):
  <a href="https://colab.research.google.com/github/YASSERRMD/BarqTrain/blob/main/examples/barqtrain_benchmark_comparison_colab.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  Both notebooks fail fast if `barqtrain_rs` or `barqtrain_cuda` do not load. The benchmark notebook compares only baseline HF, compiled BarqTrain CUDA, and full-weight Unsloth.

### Local (from source)

```bash
# Clone the repository
git clone https://github.com/YASSERRMD/BarqTrain.git
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
```

### Training Helpers

BarqTrain exposes thin helpers for the optimized training path:

- `patch_model(model)`: patches supported RMSNorm layers, configures the best attention backend available, and routes compatible decoder-only training with labels through chunked loss
- `PackedCausalLMDataCollator(...)`: uses the Rust packing backend for denser causal-LM batches
- `create_optimizer(...)`: selects `adamw`, `paged_adamw_32bit`, or `paged_adamw_8bit`

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

BarqTrain achieves significant improvements over standard Hugging Face/PyTorch training:

| Optimization | Memory Savings | Speed Improvement |
|--------------|----------------|-------------------|
| Chunked Cross-Entropy | 40-60% VRAM | 1.2-1.5x throughput |
| Fused RMSNorm | 5-10% VRAM | 1.1-1.3x per layer |
| FlashAttention / SDPA backend selection | 20-30% VRAM | 1.5-2.0x attention |
| Fused LoRA | Minimal | 1.1-1.2x adapter compute |
| Rust sequence packing | Lower padding waste | 1.3-2.0x data loading |
| Paged optimizer | Lower optimizer-state pressure | Model and workload dependent |

## Architecture

```
barqtrain/
├── python/              # Python package
│   └── barqtrain/
│       ├── __init__.py
│       ├── patch_models.py   # HF model patching
│       ├── ops.py            # Custom ops wrappers
│       ├── lora.py           # LoRA implementations
│       └── data.py           # Rust data pipeline wrappers
├── csrc/                # CUDA kernels
│   ├── src/bindings.cpp      # PyBind11 bindings
│   └── kernels/
│       ├── rmsnorm.cu        # Fused RMSNorm
│       ├── flash_attention.cu # FlashAttention + RoPE
│       ├── chunked_cross_entropy.cu # Chunked CE
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
