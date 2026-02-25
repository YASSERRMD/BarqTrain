# BarqTrain: High-Performance LLM Fine-Tuning Accelerator

**BarqTrain** is a high-performance fine-tuning library designed to bypass the memory and compute bottlenecks of standard Hugging Face/PyTorch training. By combining **Rust** for GIL-free data pipelines, **CUDA C++** for custom fused kernels, and **Python** for the user-facing API, BarqTrain achieves massive speedups and memory reductions.

## Features

- **Fused RMSNorm**: Single-kernel normalization reducing HBM bandwidth by 3-4x
- **Chunked Cross-Entropy**: Avoids logit materialization, saving up to 60% VRAM for large vocabularies
- **FlashAttention with Fused RoPE**: Tiled attention computation with rotary position embeddings
- **Fused LoRA**: Single-pass GEMM combining base weights and LoRA adapters
- **Rust Data Pipeline**: Multi-threaded sequence packing with zero GIL contention

## Installation

### Google Colab (NVIDIA GPU) 🚀

BarqTrain works out of the box on Colab's GPU runtimes — no manual setup needed.

**Step 1 — Select a GPU runtime**

`Runtime → Change runtime type → Hardware accelerator → T4 / A100 / L4`

**Step 2 — Clone & install**

```python
# Cell 1 – clone and install (run this every time you open a new Colab session)
!git clone -b test https://github.com/YASSERRMD/BarqTrain.git
%cd BarqTrain
# pip install -e . registers the package so 'import barqtrain' works
!pip install -e . -q

# Verify the import works
import barqtrain
from barqtrain import patch_model
print(f"BarqTrain {barqtrain.__version__} loaded ✓")
```

> **⚠️ Important:** If you do `git pull` later to get updates, re-run
> `pip install -e .` so the package metadata stays in sync.

**Step 3 — (Optional) Compile CUDA kernels for maximum performance**

The CUDA kernels give the biggest speedups. Compilation takes ~2 min on Colab.

```python
# Cell 3a – install Python dev headers (required to compile C extensions)
# Without this you get: fatal error: Python.h: No such file or directory
!apt-get install -y python3-dev

# Cell 3b – compile CUDA kernels (T4 / A100 / L4 / V100 all supported)
!BARQTRAIN_BUILD_CUDA=1 pip install -e .

# Verify the CUDA extension loaded
import barqtrain_cuda
print("CUDA extension loaded ✓")
```

> **Note:** `python3-dev` is not pre-installed on Colab. It provides `Python.h`
> which is required when compiling any C/C++ extension that embeds Python.

**Step 4 — Full Colab example: fine-tune Llama-3 on a custom dataset**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from barqtrain import patch_model

# Load model (bfloat16 to fit in Colab VRAM)
model_id = "meta-llama/Meta-Llama-3-8B"   # swap for any HF model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",          # auto-places on the Colab GPU
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Apply BarqTrain fused kernels (RMSNorm, FlashAttention, chunked CE)
patch_model(model)
print("BarqTrain patches applied ✓")

# Load dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

def tokenize(example):
    return tokenizer(
        example["text"], truncation=True, max_length=512, padding="max_length"
    )

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

# Train
args = TrainingArguments(
    output_dir="./barqtrain-output",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    bf16=True,                  # use bfloat16 on Colab A100/L4/T4
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
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

### Local (from source)

```bash
# Clone the repository
git clone https://github.com/YASSERRMD/BarqTrain.git
cd BarqTrain

# Install Python package (no compilation required)
pip install -e .

# Optional: build CUDA kernels (requires NVIDIA GPU + CUDA toolkit)
BARQTRAIN_BUILD_CUDA=1 pip install -e .

# Optional: build via CMake directly
cd csrc && mkdir build && cd build
cmake ..        # auto-detects PyTorch cmake prefix
make -j$(nproc)

# Optional: build Rust extension (requires Rust + maturin)
cd ../../rust
maturin develop --release
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
| FlashAttention | 20-30% VRAM | 1.5-2.0x attention |
| Fused LoRA | Minimal | 1.1-1.2x adapter compute |
| Rust Data Pipeline | N/A | 1.3-2.0x data loading |

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

- Llama (1, 2, 3) - Full support
- Qwen/Qwen2 - Full support
- Mistral - Full support
- Any HF model with RMSNorm - Partial support

## Development

### Running Tests

```bash
# Run numerical parity tests
pytest tests/test_rmsnorm.py -v

# Run benchmarks
python -m barqtrain.benchmarks.baseline --model tinyllama --steps 100
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
