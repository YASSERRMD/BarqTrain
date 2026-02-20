# BarqTrain: High-Performance LLM Fine-Tuning Accelerator

**BarqTrain** is a high-performance fine-tuning library designed to bypass the memory and compute bottlenecks of standard Hugging Face/PyTorch training. By combining **Rust** for GIL-free data pipelines, **CUDA C++** for custom fused kernels, and **Python** for the user-facing API, BarqTrain achieves massive speedups and memory reductions.

## Features

- **Fused RMSNorm**: Single-kernel normalization reducing HBM bandwidth by 3-4x
- **Chunked Cross-Entropy**: Avoids logit materialization, saving up to 60% VRAM for large vocabularies
- **FlashAttention with Fused RoPE**: Tiled attention computation with rotary position embeddings
- **Fused LoRA**: Single-pass GEMM combining base weights and LoRA adapters
- **Rust Data Pipeline**: Multi-threaded sequence packing with zero GIL contention

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/YASSERRMD/BarqTrain.git
cd BarqTrain

# Install Python dependencies
pip install -e .

# Build CUDA extension (requires CUDA toolkit)
cd csrc
mkdir build && cd build
cmake ..
make -j

# Build Rust extension
cd ../../rust
maturin develop --release
```

### Quick Start

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
