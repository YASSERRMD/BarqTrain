# BarqTrain

BarqTrain is a native acceleration layer for decoder-only LLM training and inference. It uses Rust for orchestration and data-path utilities, CUDA/C++ for hot kernels and memory hooks, and Python as the control plane. The project focuses on measurable improvements in throughput, VRAM efficiency, and decode behavior without changing standard Hugging Face training and generation flows.

## Features

- **Fused RMSNorm**: Single-kernel normalization plus Phase 8 residual-add, attention-projection, and MLP-projection block-fusion helpers with call-site auditing
- **Fused Vocab Projection + Chunked Loss**: compatible decoder-only training uses the fused LM-head projection/loss path, while decode can request last-token-only logits
- **FlashAttention Integration**: `patch_model(...)` keeps FlashAttention as an optional fast path while Phase 9 adds native decode dispatch over contiguous, paged, and quantized KV layouts
- **Fused LoRA**: Single-pass GEMM combining base weights and LoRA adapters, plus PEFT-style module replacement for adapter-only training paths
- **Rust Data Pipeline**: Native causal-LM sequence packing with zero GIL contention
- **Padding-Free Packed Training Metadata**: Rust emits `cu_seqlens`, block offsets, position IDs, sequence IDs, document IDs, and loss masks for packed batches
- **Activation Checkpointing Presets**: `max_throughput`, `balanced`, and `max_memory_saving` presets wrap attention/MLP hot paths without changing the public training loop
- **Native Optimizer-State Layouts**: BarqTrain-managed AdamW now supports `full`, `compact`, and `paged` state modes with explicit byte accounting
- **Native Memory Accounting**: Rust/CUDA benchmark reporting splits resident model memory, KV-cache memory, decode scratch memory, training peak VRAM, and inference peak VRAM
- **Paged and Quantized KV Cache**: CUDA-backed allocator, page table, gather/scatter path, recycler/free-list management, and quantized older pages with a recent fp residual window
- **Paged Optimizer Support**: Switch between `AdamW`, `PagedAdamW32bit`, and `PagedAdamW8bit`

## Current Status

Phases 1 through 10 of the current implementation plan are now shipped. The remaining work is follow-on integration, broader model-family coverage, and future compaction/offload work rather than missing core kernel paths.

| Area | Shipped Today | Benefit Today | Roadmap Next |
|------|---------------|---------------|--------------|
| RMSNorm | Phase 8 shipped | residual-add, attention-projection, and MLP-projection fusion helpers plus call-site auditing | broader model-family integration of the fused blocks |
| Cross-entropy and decode projection | Phase 4 shipped | fused LM-head projection/loss path plus last-token decode specialization | deeper vocab/head fusion across more model families |
| Data path | Phase 5 shipped | Rust packing plus padding-free training metadata, masked packed-loss consumption, and padded fallback benchmarking | activation-memory control presets and broader model patching |
| Attention | Phase 9 shipped | native decode dispatch with RoPE, KV materialization, last-token specialization, and FlashAttention/SDPA fallback routing | broader model-family integration and future serving-side compaction work |
| Inference memory accounting | Phase 1 shipped | resident/KV/decode bucket reporting plus last-token decode cleanup | offloaded cache modes and serving-side compaction accounting |
| KV cache implementation | Phase 2 shipped | paged allocator, page tables, gather/scatter reads, recycler/free-list management, and contiguous fallback | future compaction/offload |
| Quantized KV cache | Phase 3 shipped | older pages stored in int8 with a recent fp residual window plus quality/memory tradeoff reporting | compaction/offload |
| Activation memory control | Phase 6 shipped | checkpoint presets for attention/MLP hot paths plus stability/VRAM benchmark reporting | native optimizer-state control |
| Optimizer memory | Phase 7 shipped | native `barqtrain_adamw`, `barqtrain_adamw_compact`, and `barqtrain_adamw_paged` modes with explicit state accounting | compaction/offload and broader training-loop ownership |
| Adapter training | Phase 10 shipped | fused LoRA training path, PEFT-style module replacement, backward cleanup, and benchmark reporting for dense and packed training | broader model-family coverage and tighter mixed-precision integration |

## Research References And Follow-On Work

The current plan and remaining follow-on work are tracked in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

The roadmap is based on the most relevant public work for this problem space:

- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [KIVI KV-cache quantization](https://arxiv.org/abs/2402.02750)
- [Cut Cross Entropy](https://arxiv.org/abs/2411.09009)
- [Padding-Free Transformer](https://huggingface.co/blog/mayank-mishra/padding-free-transformer)
- [PyTorch activation checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)

## Installation

### Google Colab (NVIDIA GPU)

BarqTrain requires a native Rust build during install. The Colab flow below installs
the Rust toolchain first and fails immediately if `barqtrain_rs` does not build.

**Step 1: Select a GPU runtime**

`Runtime -> Change runtime type -> Hardware accelerator -> T4 / A100 / L4`

**Step 2: Clone and install**

```python
# Cell 1 - clone, install, and verify (run once per Colab session)
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

# Colab does not always reload .pth files in a running session.
# this sys.path line makes the import work immediately.
import sys, importlib, importlib.util
sys.path.insert(0, '/content/BarqTrain/python')
importlib.invalidate_caches()
assert importlib.util.find_spec("barqtrain_rs"), "barqtrain_rs did not build"
import barqtrain._ffi as ffi
assert ffi.load_rust_backend() is not None, "barqtrain_rs did not load"

import barqtrain
from barqtrain import patch_model
print(f"BarqTrain {barqtrain.__version__} loaded.")
```

**Step 3: Restart the Colab runtime**

After the native Rust/CUDA build, restart the runtime before training or benchmarking.

`Runtime -> Restart session`

**Step 4: Verify native runtime loading after restart**

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

**Step 5: Optionally compile CUDA kernels for maximum performance**

The CUDA kernels give the biggest speedups. Compilation takes ~2 min on Colab.

```python
# Cell 3 - compile CUDA kernels (T4 / A100 / L4 / V100 all supported)
!BARQTRAIN_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation

# Verify the CUDA extension loaded
import importlib.util
assert importlib.util.find_spec("barqtrain_cuda"), "barqtrain_cuda did not build"
import barqtrain._ffi as ffi
assert ffi.load_cuda_backend() is not None, "barqtrain_cuda did not load"
print("CUDA extension loaded.")
```

**Step 6: Fine-tune a model with BarqTrain patches and optional fused LoRA**

```python
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from barqtrain import patch_lora_modules, patch_model

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

# Apply BarqTrain fused kernels (RMSNorm, attention dispatch, chunked CE)
patch_model(model)
patch_lora_modules(model, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"), rank=8, alpha=16.0)
for name, param in model.named_parameters():
    if "lora_" not in name:
        param.requires_grad = False
print("BarqTrain patches applied.")

# Load and tokenize dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

def tokenize(example):
    out = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    # Labels must be provided. For causal LM they equal input_ids.
    # DataCollatorForLanguageModeling will shift them internally.
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
tokenized.set_format("torch")

# DataCollator for causal LM. Padding positions in labels are set to -100.
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

### Ready-to-use Colab notebooks

- Training + Inference (`examples/barqtrain_training_inference_colab.ipynb`):
  <a href="https://colab.research.google.com/github/YASSERRMD/BarqTrain/blob/main/examples/barqtrain_training_inference_colab.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
- Benchmark Comparison (`examples/barqtrain_benchmark_comparison_colab.ipynb`):
  <a href="https://colab.research.google.com/github/YASSERRMD/BarqTrain/blob/main/examples/barqtrain_benchmark_comparison_colab.ipynb" target="_blank" rel="noopener noreferrer"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  Both notebooks now make the post-build restart explicit and include a runtime verification step for `barqtrain_rs` and `barqtrain_cuda`.

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

# Headless/Docker CUDA build without a visible GPU:
BARQTRAIN_CUDA_ARCH_LIST=7.5 BARQTRAIN_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation
```

### Training Helpers

BarqTrain exposes thin helpers for the optimized training path:

- `patch_model(model)`: patches supported RMSNorm layers, configures the best attention backend available, routes compatible decoder-only training with labels through the fused LM-head projection/loss path, and wraps compatible CUDA generation calls to inject BarqTrain's native KV cache automatically (`paged`, `contiguous`, `paged_quantized`, or `auto`)
- `patch_inference(model)`: inference-only patching path for decode benchmarks and low-memory generation experiments
- `patch_lora_modules(model, ...)`: replaces compatible projection modules with fused LoRA adapters while keeping a PEFT-style Python control plane
- `PackedCausalLMDataCollator(...)`: uses the Rust packing backend for denser causal-LM batches
- `PaddingFreeCausalLMDataCollator(...)`: emits packed blocks plus `cu_seqlens`, block offsets, document IDs, and loss masks for padding-free training experiments
- `pack_for_padding_free_causal_lm(...)`: direct Rust-backed packing helper for jagged metadata emission
- `apply_activation_checkpointing(model, preset=...)`: wraps compatible attention/MLP modules with explicit `max_throughput`, `balanced`, or `max_memory_saving` presets
- `create_optimizer(...)`: selects `adamw`, BarqTrain-native AdamW state modes, or the existing bitsandbytes paged variants
- `optimizer_state_bytes(optimizer)`: reports native optimizer-state memory separately from model and activation memory
- `create_kv_cache(...)`: explicitly create a contiguous, paged, or paged-quantized KV cache
- `create_contiguous_kv_cache(...)`: explicitly create the contiguous fallback cache
- `create_paged_kv_cache(...)`: explicitly create a paged KV cache when you want to control decode capacity yourself
- `create_quantized_paged_kv_cache(...)`: explicitly create a paged KV cache that quantizes older pages and keeps a recent fp residual window

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

For explicit decode-cache control, you can create and pass any shipped cache layout yourself:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from barqtrain import create_kv_cache, patch_inference

model_id = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
patch_inference(model)

inputs = tokenizer("Explain paged KV caches.", return_tensors="pt").to("cuda")
cache = create_kv_cache(model, max_batch_size=1, max_cache_len=256, page_size=16, mode="paged_quantized")
outputs = model.generate(**inputs, max_new_tokens=64, past_key_values=cache)
```

Runtime selection is also available via `BARQTRAIN_KV_CACHE_MODE=auto|paged|contiguous|paged_quantized`. The quantized path uses `BARQTRAIN_QUANTIZED_KV_RESIDUAL_TOKENS` to keep a recent fp16/bf16 window while older pages are stored in int8. `BARQTRAIN_LAST_TOKEN_LOGITS_ONLY=1` keeps generation on the last-token projection path by default, while `BARQTRAIN_LAST_TOKEN_LOGITS_ONLY=0` forces the full-logits fallback. Detailed native memory bucket collection remains gated behind `BARQTRAIN_DETAILED_PROFILING=1` so release-path overhead stays minimal.


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
- **Inference path**: Phase 1 reports memory buckets separately, Phase 2 compares paged versus contiguous KV-cache behavior, Phase 3 extends that to quantized older pages with explicit quality/memory tradeoffs, and Phase 4 measures fused LM-head projection/loss versus the full-logits fallback.
- **Packed training path**: Phase 5 compares padded versus packed training at matched effective-token counts and tracks the document-masked packed mode separately.
- **Activation-memory path**: Phase 6 compares checkpoint presets on the same training state and reports VRAM, tokens/sec, step time, and loss-curve stability.
- **Optimizer-state path**: Phase 7 compares native optimizer layouts against AdamW and reports state bytes, throughput, and loss deltas.
- **RMSNorm block-fusion path**: Phase 8 compares separated versus fused residual/norm/projection execution and reports approximate memory-traffic reduction alongside parity.
- **Attention dispatch path**: Phase 9 compares FlashAttention, native decode dispatch, and SDPA across prefill, decode, and long-context serving workloads.
- **Adapter-training path**: Phase 10 compares a PEFT-style LoRA baseline against BarqTrain fused LoRA on dense and packed chunked-loss training workloads.

Shipped benchmark reporting now includes these memory buckets explicitly:

1. `resident_model_mb`
2. `kv_cache_mb`
3. `temporary_decode_buffers_mb`
4. `training_peak_vram_mb`
5. `inference_peak_vram_mb`

The shipped Phase 1 decode matrix covers:

1. `short_prompt_long_decode`
2. `long_prompt_short_decode`
3. batch sizes `1`, `4`, and `8`

The Phase 1 decode report also records:

1. `paged_kv_cache`
2. `last_token_logits_only`

The shipped Phase 2 KV benchmark suite compares `paged` and `contiguous` layouts on:

1. `long_prompt_generation`
2. `multi_request_serving`
3. `fixed_vram_batch_growth`

Each Phase 2 KV report entry records:

1. `scenario_name`
2. `cache_layout`
3. `oom_rate`
4. `fragmentation_ratio`
5. `resident_vram_mb`
6. `peak_vram_mb`
7. the same bucketed `memory` breakdown used by Phase 1

The shipped Phase 3 quantized KV benchmark suite compares `contiguous`, `paged`, and `paged_quantized` layouts on:

1. `memory_savings_vs_latency`
2. `long_context_generation_quality`
3. `throughput_per_gb`

Each Phase 3 quantized KV report entry records:

1. `throughput_per_gb`
2. `memory_savings_vs_contiguous_percent`
3. `latency_vs_contiguous_percent`
4. `generation_match_ratio`
5. `perplexity`
6. `reference_perplexity`
7. the same resident, peak, and bucketed memory fields used by earlier suites

The shipped Phase 4 fused projection benchmark suite compares `baseline` and `fused` projection modes on:

1. `vocab_heavy_long_decode`
2. `vocab_heavy_long_context`

Each Phase 4 projection report entry records:

1. `projection_mode`
2. `training_step_time_seconds`
3. `decode_tokens_per_second`
4. `training_peak_vram_mb`
5. `inference_peak_vram_mb`
6. `loss_delta_vs_baseline`
7. `generation_match_ratio`
8. `last_token_logits_only`
9. the same bucketed `memory` breakdown used by earlier suites

The shipped Phase 5 packed training benchmark suite compares `padded` and `packed` execution on:

1. `matched_effective_tokens`
2. `document_masked_training`

Each Phase 5 packed training report entry records:

1. `packing_mode`
2. `effective_tokens`
3. `step_time_seconds`
4. `effective_tokens_per_second`
5. `throughput_at_matched_effective_tokens`
6. `peak_vram_mb`
7. `loss_delta_vs_padded`
8. the same bucketed `memory` breakdown used by earlier suites

The shipped Phase 6 activation-checkpoint benchmark suite compares these presets:

1. `max_throughput`
2. `balanced`
3. `max_memory_saving`

Each Phase 6 checkpoint report entry records:

1. `preset_name`
2. `total_steps`
3. `total_tokens`
4. `tokens_per_second`
5. `avg_step_time_seconds`
6. `peak_vram_mb`
7. `loss_stddev`
8. `loss_delta_vs_max_throughput`
9. the same bucketed `memory` breakdown used by earlier suites

The shipped Phase 7 optimizer benchmark suite compares these modes:

1. `adamw`
2. `barqtrain_adamw`
3. `barqtrain_adamw_compact`
4. `barqtrain_adamw_paged`

Each Phase 7 optimizer report entry records:

1. `optimizer_name`
2. `total_steps`
3. `total_tokens`
4. `tokens_per_second`
5. `avg_step_time_seconds`
6. `optimizer_state_mb`
7. `loss_delta_vs_adamw`
8. the same bucketed `memory` breakdown used by earlier suites

The shipped Phase 8 RMSNorm fusion benchmark suite compares `separated` and `fused` execution on:

1. `residual_add_rmsnorm`
2. `attention_input_projection`
3. `mlp_input_projection`

Each Phase 8 RMSNorm fusion report entry records:

1. `fusion_mode`
2. `latency_seconds`
3. `effective_tokens_per_second`
4. `approximate_memory_traffic_mb`
5. `memory_traffic_reduction_percent`
6. `max_abs_error`

The shipped Phase 9 attention benchmark suite compares these backends across the scenario matrix:

1. `flash_attention_2`
2. `barqtrain_native_decode`
3. `sdpa`

The shipped Phase 9 scenario matrix covers:

1. `prefill_throughput`
2. `decode_throughput`
3. `long_context_serving`

Each Phase 9 attention report entry records:

1. `attention_backend`
2. `cache_layout`
3. `prefill_tokens_per_second`
4. `decode_tokens_per_second`
5. `memory_overhead_mb`
6. `max_abs_error`
7. `last_token_only`

The shipped Phase 10 fused LoRA benchmark suite compares these adapter modes:

1. `reference`
2. `barqtrain_fused`

The shipped Phase 10 scenario matrix covers:

1. `dense_chunked_loss`
2. `packed_chunked_loss`

Each Phase 10 fused LoRA report entry records:

1. `adapter_mode`
2. `effective_tokens`
3. `step_time_seconds`
4. `effective_tokens_per_second`
5. `peak_vram_mb`
6. `loss_delta_vs_reference`
7. the same bucketed `memory` breakdown used by earlier training suites

The remaining follow-on work in [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) now focuses on:

1. broader model-family integration for the shipped kernels and training paths
2. cache compaction and offload work for longer-context serving

Example Phase 1 report shape:

```json
{
  "training": {
    "memory": {
      "resident_model_mb": 0.0,
      "kv_cache_mb": 0.0,
      "temporary_decode_buffers_mb": 0.0,
      "training_peak_vram_mb": 0.0,
      "inference_peak_vram_mb": 0.0
    }
  },
  "inference_profiles": [
    {
      "profile_name": "short_prompt_long_decode",
      "batch_size": 1,
      "paged_kv_cache": false,
      "last_token_logits_only": true,
      "memory": {
        "resident_model_mb": 0.0,
        "kv_cache_mb": 0.0,
        "temporary_decode_buffers_mb": 0.0,
        "training_peak_vram_mb": 0.0,
        "inference_peak_vram_mb": 0.0
      }
    }
  ]
}
```

Example Phase 2 report shape:

```json
{
  "benchmark_suite": "phase2",
  "kv_cache_profiles": [
    {
      "scenario_name": "long_prompt_generation",
      "cache_layout": "paged",
      "batch_size": 1,
      "oom_rate": 0.0,
      "fragmentation_ratio": 0.0,
      "resident_vram_mb": 0.0,
      "peak_vram_mb": 0.0,
      "memory": {
        "resident_model_mb": 0.0,
        "kv_cache_mb": 0.0,
        "temporary_decode_buffers_mb": 0.0,
        "training_peak_vram_mb": 0.0,
        "inference_peak_vram_mb": 0.0
      }
    }
  ]
}
```

Example Phase 3 report shape:

```json
{
  "benchmark_suite": "phase3",
  "quantized_kv_profiles": [
    {
      "scenario_name": "memory_savings_vs_latency",
      "cache_layout": "paged_quantized",
      "throughput_per_gb": 0.0,
      "memory_savings_vs_contiguous_percent": 0.0,
      "latency_vs_contiguous_percent": 0.0,
      "generation_match_ratio": 1.0,
      "perplexity": 0.0,
      "reference_perplexity": 0.0,
      "memory": {
        "resident_model_mb": 0.0,
        "kv_cache_mb": 0.0,
        "temporary_decode_buffers_mb": 0.0,
        "training_peak_vram_mb": 0.0,
        "inference_peak_vram_mb": 0.0
      }
    }
  ]
}
```

Example Phase 4 report shape:

```json
{
  "benchmark_suite": "phase4",
  "projection_profiles": [
    {
      "scenario_name": "vocab_heavy_long_decode",
      "projection_mode": "fused",
      "training_step_time_seconds": 0.0,
      "decode_tokens_per_second": 0.0,
      "training_peak_vram_mb": 0.0,
      "inference_peak_vram_mb": 0.0,
      "loss_delta_vs_baseline": 0.0,
      "generation_match_ratio": 1.0,
      "last_token_logits_only": true,
      "memory": {
        "resident_model_mb": 0.0,
        "kv_cache_mb": 0.0,
        "temporary_decode_buffers_mb": 0.0,
        "training_peak_vram_mb": 0.0,
        "inference_peak_vram_mb": 0.0
      }
    }
  ]
}
```

Example Phase 5 report shape:

```json
{
  "benchmark_suite": "phase5",
  "packed_training_profiles": [
    {
      "scenario_name": "matched_effective_tokens",
      "packing_mode": "packed",
      "effective_tokens": 0,
      "step_time_seconds": 0.0,
      "effective_tokens_per_second": 0.0,
      "throughput_at_matched_effective_tokens": 0.0,
      "peak_vram_mb": 0.0,
      "loss_delta_vs_padded": 0.0,
      "memory": {
        "resident_model_mb": 0.0,
        "kv_cache_mb": 0.0,
        "temporary_decode_buffers_mb": 0.0,
        "training_peak_vram_mb": 0.0,
        "inference_peak_vram_mb": 0.0
      }
    }
  ]
}
```

Example Phase 6 report shape:

```json
{
  "benchmark_suite": "phase6",
  "checkpoint_profiles": [
    {
      "preset_name": "balanced",
      "total_steps": 3,
      "total_tokens": 0,
      "tokens_per_second": 0.0,
      "avg_step_time_seconds": 0.0,
      "peak_vram_mb": 0.0,
      "loss_stddev": 0.0,
      "loss_delta_vs_max_throughput": 0.0,
      "memory": {
        "resident_model_mb": 0.0,
        "kv_cache_mb": 0.0,
        "temporary_decode_buffers_mb": 0.0,
        "training_peak_vram_mb": 0.0,
        "inference_peak_vram_mb": 0.0
      }
    }
  ]
}
```

Example Phase 7 report shape:

```json
{
  "benchmark_suite": "phase7",
  "optimizer_profiles": [
    {
      "optimizer_name": "barqtrain_adamw_compact",
      "total_steps": 5,
      "total_tokens": 0,
      "tokens_per_second": 0.0,
      "avg_step_time_seconds": 0.0,
      "optimizer_state_mb": 0.0,
      "loss_delta_vs_adamw": 0.0,
      "memory": {
        "resident_model_mb": 0.0,
        "kv_cache_mb": 0.0,
        "temporary_decode_buffers_mb": 0.0,
        "training_peak_vram_mb": 0.0,
        "inference_peak_vram_mb": 0.0
      }
    }
  ]
}
```

Example Phase 8 report shape:

```json
{
  "benchmark_suite": "phase8",
  "rmsnorm_fusion_profiles": [
    {
      "scenario_name": "attention_input_projection",
      "fusion_mode": "fused",
      "latency_seconds": 0.0,
      "effective_tokens_per_second": 0.0,
      "approximate_memory_traffic_mb": 0.0,
      "memory_traffic_reduction_percent": 0.0,
      "max_abs_error": 0.0
    }
  ]
}
```

Example Phase 9 report shape:

```json
{
  "benchmark_suite": "phase9",
  "attention_profiles": [
    {
      "scenario_name": "decode_throughput",
      "attention_backend": "barqtrain_native_decode",
      "cache_layout": "paged",
      "prefill_tokens_per_second": 0.0,
      "decode_tokens_per_second": 0.0,
      "memory_overhead_mb": 0.0,
      "max_abs_error": 0.0,
      "last_token_only": true
    }
  ]
}
```

Example Phase 10 report shape:

```json
{
  "benchmark_suite": "phase10",
  "lora_profiles": [
    {
      "scenario_name": "packed_chunked_loss",
      "adapter_mode": "barqtrain_fused",
      "effective_tokens": 0,
      "step_time_seconds": 0.0,
      "effective_tokens_per_second": 0.0,
      "peak_vram_mb": 0.0,
      "loss_delta_vs_reference": 0.0,
      "memory": {
        "resident_model_mb": 0.0,
        "kv_cache_mb": 0.0,
        "temporary_decode_buffers_mb": 0.0,
        "training_peak_vram_mb": 0.0,
        "inference_peak_vram_mb": 0.0
      }
    }
  ]
}
```

## Architecture

```text
barqtrain/
|- python/barqtrain/__init__.py
|- python/barqtrain/patch_models.py
|- python/barqtrain/kv_cache.py
|- python/barqtrain/ops.py
|- python/barqtrain/lora.py
|- python/barqtrain/data.py
|- csrc/src/bindings.cpp
|- csrc/kernels/rmsnorm.cu
|- csrc/kernels/flash_attention.cu
|- csrc/kernels/chunked_cross_entropy.cu
|- csrc/kernels/paged_kv_cache.cu
|- csrc/kernels/lora.cu
|- rust/src/lib.rs
`- tests/
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
python -m barqtrain.benchmarks.baseline \
  --model tinyllama \
  --mode both \
  --steps 100 \
  --detailed-profiling

# Benchmark Rust packing + paged optimizer
python -m barqtrain.benchmarks.baseline \
  --model tinyllama \
  --mode training \
  --steps 100 \
  --use-packing \
  --optimizer paged_adamw_32bit

# Benchmark fused LoRA training modes
python -m barqtrain.benchmarks.baseline \
  --suite phase10 \
  --model tinyllama \
  --sequence_length 512 \
  --inference-batch-sizes 1,4,8
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
