# BarqTrain: High-Performance LLM Fine-Tuning Accelerator

**BarqTrain** is a high-performance fine-tuning library designed to bypass the memory and compute bottlenecks of standard Hugging Face/PyTorch training. By combining **Rust** for GIL-free data pipelines, **CUDA C++** for custom fused kernels, and **Python** for the user-facing API, BarqTrain achieves massive speedups and memory reductions similar to Unsloth and Liger Kernel.

---

## 1. Core Technical Bottlenecks & BarqTrain Solutions

Standard fine-tuning is memory-bandwidth-bound. BarqTrain solves this by attacking 6 specific inefficiencies:

1. **Attention Memory & Bandwidth:** Standard attention materializes O(N^2) matrices in HBM. 
   * **Solution:** Implement a custom CUDA C++ fused FlashAttention-3 kernel (using CUTLASS) with a tiled backward pass that avoids FP32 atomics.
2. **Cross-Entropy Loss Logit Materialization:** Large vocabularies (e.g., 128K for Llama 3) cause massive OOMs by materializing the `[batch × seq_len × vocab_size]` tensor.
   * **Solution:** Build a chunked CUDA kernel fusing `nn.Linear → log_softmax → nll_loss` that processes the vocab dimension in chunks without materializing the full logit tensor (saves up to 60% VRAM).
3. **RMSNorm / LayerNorm Overhead:** Standard norms launch 3-4 separate kernels, wasting HBM bandwidth.
   * **Solution:** A single fused CUDA C kernel that reads input once, computes mean/variance/normalization in shared memory, and writes output once.
4. **RoPE (Rotary Positional Embeddings):** Standard RoPE does extra HBM reads/writes of Q and K.
   * **Solution:** Fuse RoPE rotation directly into the FlashAttention QK computation in C++/CUDA.
5. **LoRA Adapter Inefficiency:** Standard PEFT applies LoRA via separate matmuls and additions.
   * **Solution:** Fused LoRA C/CUDA kernel performing `xW + x(AB)` in a single batched GEMM step.
6. **Data Pipeline CPU Stalls:** Tokenization and packing in Python starve the GPU due to the GIL.
   * **Solution:** A PyO3-bound Rust pipeline that multi-threads tokenization and sequence packing, filling a lock-free prefetch queue.

---

## 2. Repository Architecture

Use a monorepo structure to keep the multi-language stack organized:

```text
barqtrain/
├── python/                  # Python package (HF patching + trainer)
│   └── barqtrain/
│       ├── __init__.py
│       ├── patch_models.py  # Monkey-patches HF models with custom ops
│       └── benchmarks/      # Perf tracking scripts
├── csrc/                    # CUDA/C++ Torch Extensions
│   ├── src/                 # PyBind11 bindings
│   ├── kernels/             # .cu files (FlashAttn, RMSNorm, ChunkedCE)
│   └── CMakeLists.txt
├── rust/                    # Rust core for data pipelines
│   ├── Cargo.toml
│   └── src/                 # PyO3 bindings, Rayon parallel packers
├── tests/                   # Parity tests vs standard PyTorch
├── pyproject.toml           # Build config (Maturin + Setuptools)
└── README.md
```

---

## 3. Git Workflow & Best Practices

To maintain a clean, professional open-source history, BarqTrain enforces strict Git identity and **Atomic Commits**.

### Git Identity Setup
Ensure your commits are properly attributed to your developer profile:
```bash
# Set globally for your machine
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# OR set locally just for the BarqTrain repo
git config user.name "Your Name"
git config user.email "you@example.com"

# Verify
git config --list
```

### Atomic & Conventional Commits
An **atomic commit** contains exactly *one logical change*. Do not mix kernel development with documentation or unrelated bug fixes.

We follow the **Conventional Commits** specification (`type(scope): message`):
* `feat(scope): ...` — New feature or kernel.
* `fix(scope): ...` — Bug fix or numerics correction.
* `perf(scope): ...` — Performance improvements.
* `refactor(scope): ...` — Code restructuring.
* `test(scope): ...` — Adding tests.
* `docs(scope): ...` — Documentation changes.

*Examples:*
* `feat(cuda/rmsnorm): implement fused forward pass`
* `fix(rust/packer): resolve index out of bounds in bin-packing`
* `perf(cuda/xent): chunk vocab dimension to reduce VRAM by 40%`

---

## 4. Phase-by-Phase Implementation Plan

### Phase 0: Bootstrap & Infrastructure (Day 1)
**Goal:** Repo builds, imports, and runs a trivial "hello op".
* Initialize Git repo, `.gitignore`, and basic folder structure.
* Set up `pyproject.toml` to build both the C++ extension and the Rust PyO3 module.
* **Commits:**
  * `chore(repo): bootstrap BarqTrain skeleton`
  * `build(python): configure pyproject for maturin and cmake`

### Phase 1: Benchmark Harness (Day 2-3)
**Goal:** Create a reproducible baseline to measure your future optimizations against.
* Write a script that fine-tunes a tiny Llama/Qwen model on standard Hugging Face/PEFT.
* Track tokens/sec, step time, and peak VRAM.
* **Commits:**
  * `feat(bench): add tokens/sec and VRAM tracking harness`
  * `docs(bench): record baseline HF performance metrics`

### Phase 2: Rust Data Pipeline (Day 4-7)
**Goal:** Eliminate Python GIL GPU starvation.
* Implement parallel sequence packing (bin-packing algorithms) in Rust using `rayon`.
* Expose it to Python using `PyO3`.
* Add a prefetch queue to feed the dataloader.
* **Commits:**
  * `feat(rust): initialize PyO3 crate`
  * `feat(rust/pack): add multi-threaded sequence bin-packing`
  * `perf(python/data): integrate Rust prefetch queue into trainer`

### Phase 3: Fused RMSNorm (Week 2)
**Goal:** Your first CUDA kernel win. Easy to implement, runs at every layer.
* Write `fused_rmsnorm.cu` (one read, shared memory reduction, one write).
* Wrap via `torch::extension` and `pybind11`.
* Write a PyTorch autograd wrapper and test exact parity against `torch.nn.RMSNorm`.
* **Commits:**
  * `feat(cuda/rmsnorm): add fused rmsnorm forward and backward`
  * `test(cuda/rmsnorm): add bf16 numerical parity tests`
  * `feat(patch): add injection script to replace HF norms`

### Phase 4: Chunked Cross-Entropy Loss (Week 2-3) **[HIGHEST ROI]**
**Goal:** Massive VRAM reduction by avoiding logit materialization.
* Write a custom kernel that fuses the final linear projection, log_softmax, and NLL loss.
* Process the `vocab_size` dimension in SRAM chunks.
* Compute the gradient w.r.t the hidden states directly.
* **Commits:**
  * `feat(cuda/xent): implement chunked linear and cross-entropy forward`
  * `feat(cuda/xent): implement backward pass for dHidden`
  * `perf(train): route trainer loss to use fused CE`

### Phase 5: FlashAttention & Fused RoPE (Week 4-5)
**Goal:** Speed up sequence processing and remove RoPE memory overhead.
* Port or implement FlashAttention-2/3 kernels using CUTLASS.
* Fuse Rotary Positional Embeddings directly into the Q and K load phase inside the attention kernel.
* **Commits:**
  * `feat(cuda/attn): integrate flash attention forward/backward`
  * `feat(cuda/rope): fuse rope calculation into qk load phase`
  * `test(cuda/attn): verify attention parity with causal masking`

### Phase 6: Fused LoRA GEMM (Week 6)
**Goal:** Drop the adapter addition overhead.
* Implement a kernel to compute `xW_base + x(A@B)` in a single step for the forward pass during inference/validation, and optimize the backward gradient paths for frozen base weights.
* **Commits:**
  * `feat(cuda/lora): implement fused lora gemm forward`
  * `perf(cuda/lora): optimize backward paths for frozen base weights`

### Phase 7: Packaging & Release
**Goal:** Make BarqTrain installable via `pip`.
* Finalize wheel generation.
* Write usage instructions (e.g., `from barqtrain import patch_model; patch_model(model)`).
* **Commits:**
  * `build(ci): add GitHub actions for wheel building`
  * `docs(readme): add installation and quickstart guide`

---

## 5. Implementation Code Patterns

### Pattern A: Rust to Python (PyO3)
```rust
// rust/src/lib.rs
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn pack_sequences(sequences: Vec<Vec<u32>>, max_len: usize) -> Vec<Vec<u32>> {
    sequences.par_iter()
        .map(|seq| /* packing logic */ seq.clone())
        .collect()
}

#[pymodule]
fn barqtrain_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pack_sequences, m)?)?;
    Ok(())
}
```

### Pattern B: CUDA to PyTorch (PyBind11)
```cpp
// csrc/src/bindings.cpp
#include <torch/extension.h>

// Forward declaration of CUDA launcher
torch::Tensor fused_rmsnorm_cuda(torch::Tensor input, torch::Tensor weight, float eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rmsnorm", &fused_rmsnorm_cuda, "BarqTrain Fused RMSNorm (CUDA)");
}
```

### Pattern C: Python Monkey-Patching
```python
# python/barqtrain/patch_models.py
import torch
import barqtrain_cuda as _C

class BarqRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        out, rms = _C.fused_rmsnorm(x, w, eps)
        ctx.save_for_backward(x, w, rms)
        return out

def patch_llama(model):
    import transformers
    for module in model.modules():
        if isinstance(module, transformers.models.llama.modeling_llama.LlamaRMSNorm):
            # Swap standard norm with BarqTrain fused norm
            module.forward = lambda x: BarqRMSNorm.apply(x, module.weight, module.variance_epsilon)
```
