# BarqTrain Research-Based Implementation Plan

BarqTrain already ships native CUDA kernels for RMSNorm and chunked cross-entropy, plus a Rust-backed data path for sequence packing. The current bottleneck is different: inference speed is improved, but peak VRAM during short-context full-weight generation is still dominated by model residency rather than BarqTrain's temporary buffers.

This plan turns the project into a native memory-and-throughput stack in phased, measurable steps.

## Objective

Build BarqTrain into a native accelerator that improves all three of the following at the same time:

- faster training throughput
- faster inference throughput
- lower memory use in both training and inference

## Current State

What BarqTrain already does in native code:

- CUDA fused RMSNorm
- CUDA chunked cross-entropy to avoid full logits materialization during training
- Rust sequence packing for the causal-LM data path

What is still missing for meaningful inference-memory reduction:

- native KV-cache management
- native KV-cache compression / quantization
- native decode-time logits optimization
- native padding-free attention path for packed training batches

## Why The Current Inference VRAM Barely Moves

In the current benchmark, full-weight model residency dominates GPU memory. That means total peak VRAM is mostly the model itself, not BarqTrain's execution buffers. As a result, CUDA kernel fusion can improve latency and throughput without changing total VRAM very much.

To materially reduce inference memory, BarqTrain needs native work in the generation path itself:

- paged KV-cache allocation
- KV-cache quantization or offload
- reduced decode-time temporary buffers

## Research Signals Guiding The Plan

The phases below are based on the most relevant public work for LLM training and serving:

- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)
- [KIVI KV-cache quantization](https://arxiv.org/abs/2402.02750)
- [Cut Cross Entropy](https://arxiv.org/abs/2411.09009)
- [Padding-Free Transformer](https://huggingface.co/blog/mayank-mishra/padding-free-transformer)
- [PyTorch activation checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

## Phase 1: Native Memory Accounting And Decode Cleanup

Goal:
Establish accurate memory attribution before adding more kernels.

Why first:
Without separating resident model memory from KV-cache and scratch buffers, the project will keep optimizing the wrong number.

Native work:

- CUDA-side instrumentation for:
  - resident model memory
  - KV-cache memory
  - temporary decode buffers
- decode path that only computes last-token logits during generation when full logits are not needed
- explicit separation of training-memory and inference-memory metrics in the benchmark harness

Expected impact:

- cleaner regression tracking
- lower decode-time scratch memory
- modest inference speed improvement

Validation:

- report resident VRAM, peak VRAM, and decode overhead separately
- verify no regression in generated text parity

## Phase 2: CUDA Paged KV Cache

Goal:
Reduce inference memory fragmentation and enable longer contexts without allocating monolithic contiguous KV buffers.

Research basis:
[PagedAttention / vLLM](https://arxiv.org/abs/2309.06180)

Native work:

- CUDA page allocator for KV blocks
- page table + gather/scatter kernels for attention reads
- block recycling on decode steps
- model patching hooks that route supported models through the paged cache path

Expected impact:

- real inference-memory reduction under long contexts and batch growth
- better serving stability
- improved decode throughput from reduced allocator pressure

Validation:

- compare contiguous vs paged cache under long prompts
- track OOM rate, fragmentation, and tokens/sec at fixed context length

## Phase 3: CUDA Quantized KV Cache

Goal:
Compress KV-cache memory after the paged allocator exists.

Research basis:
[KIVI](https://arxiv.org/abs/2402.02750)

Native work:

- quantized KV-cache storage in CUDA
- residual full-precision window plus compressed older pages
- dequant-on-read path integrated into attention kernels
- runtime selection between full, paged, and quantized KV-cache modes

Expected impact:

- significant inference-memory reduction for long contexts
- better throughput-per-GB for multi-request serving

Validation:

- perplexity / generation quality checks against fp16 or bf16 cache
- memory reduction vs latency tradeoff curves

## Phase 4: Fused Vocab Projection And Loss Path

Goal:
Reduce both training memory and decode-time temporary allocations around the LM head.

Research basis:
[Cut Cross Entropy](https://arxiv.org/abs/2411.09009) and [Liger Kernel](https://github.com/linkedin/Liger-Kernel)

Native work:

- extend the current chunked cross-entropy path into a more fully fused linear-plus-loss implementation
- add decode-time last-token projection specialization so generation does not materialize more logits than necessary
- reduce unnecessary fp32 promotion and temporary tensor retention in the native loss path

Expected impact:

- lower training activation pressure for large vocab models
- faster training step time
- less decode-time temporary memory

Validation:

- numerical parity against `torch.nn.functional.cross_entropy`
- peak VRAM and tokens/sec on vocab-heavy models

## Phase 5: Rust Packing Plus CUDA Padding-Free Training

Goal:
Stop paying for padded tokens during training.

Research basis:
[Padding-Free Transformer](https://huggingface.co/blog/mayank-mishra/padding-free-transformer)

Native work:

- Rust emits packed jagged metadata, offsets, and document boundaries
- CUDA attention and loss kernels consume packed sequences directly
- optional document-masked packed training mode for decoder-only fine-tuning

Expected impact:

- lower training memory from fewer padded activations
- higher effective tokens/sec
- better utilization on instruction-tuning datasets with variable sequence lengths

Validation:

- compare padded vs packed runs at identical effective tokens
- verify document-boundary masking correctness

## Phase 6: Training Activation Memory Control

Goal:
Reduce training-time activation memory so larger batches or longer sequences fit.

Research basis:
[PyTorch activation checkpointing techniques](https://pytorch.org/blog/activation-checkpointing-techniques/)

Native work:

- checkpoint-friendly native kernels and patched model wrappers
- selective checkpointing around attention and MLP hot paths
- training presets that combine chunked loss, packed batches, and checkpointing without breaking native kernels

Expected impact:

- materially lower training VRAM
- larger batch size or sequence length on the same GPU

Validation:

- benchmark VRAM and throughput with and without selective checkpointing
- ensure backward correctness and stable loss curves

## Phase 7: Native Optimizer-State Memory Work

Goal:
Reduce optimizer-state overhead without depending entirely on external wrappers.

Why later:
Optimizer memory matters for training, but the highest immediate ROI is still in loss, attention, and padding elimination.

Native work:

- evaluate a native paged optimizer-state layout for BarqTrain-managed training loops
- keep compatibility with bitsandbytes paths where native support is not ready
- expose explicit memory/performance tradeoffs rather than a single default

Expected impact:

- lower optimizer-state pressure on fine-tuning jobs
- cleaner BarqTrain-controlled training stack

Validation:

- optimizer-state memory accounting
- convergence checks against standard AdamW baselines

## Recommended Execution Order

The highest-ROI sequence is:

1. Phase 1: measure the right memory buckets
2. Phase 2: paged KV-cache
3. Phase 3: quantized KV-cache
4. Phase 4: fused projection-plus-loss improvements
5. Phase 5: padding-free packed training
6. Phase 6: activation-memory control
7. Phase 7: native optimizer-state work

## Success Criteria

BarqTrain should not claim a memory win unless at least one of these moves in the right direction on a matched benchmark:

- lower resident VRAM
- lower generation overhead VRAM
- lower training peak VRAM
- same VRAM with materially higher tokens/sec

## Non-Goals For This Plan

These items may still be useful, but they are not the primary path to native memory wins for BarqTrain right now:

- Python-only benchmark reinterpretation without native changes
- claiming inference-memory savings when only latency improved
- model-weight quantization as the main story for BarqTrain

## Deliverable Expectations Per Phase

Each phase should end with all of the following:

- native code landed in `csrc/` and/or `rust/`
- a benchmark that isolates the claimed improvement
- parity tests or tolerance-based numerical tests
- README updates that clearly distinguish shipped features from roadmap work
