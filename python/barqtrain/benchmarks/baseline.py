"""
Benchmark harness for BarqTrain training and inference benchmark suites.

Usage:
    python -m barqtrain.benchmarks.baseline --model tinyllama --mode both --steps 100
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from barqtrain.data import (
    PackedCausalLMDataCollator,
    PaddingFreeCausalLMDataCollator,
)
from barqtrain.memory import (
    BenchmarkMemoryBreakdown,
    build_generation_kwargs,
    build_memory_breakdown,
    capture_cuda_peak_bytes,
    model_resident_cuda_bytes,
    paged_kv_cache_bytes,
    phase1_inference_profiles,
    phase2_kv_cache_profiles,
    phase3_quantized_kv_profiles,
    phase4_vocab_projection_profiles,
    phase5_packed_training_profiles,
    phase6_activation_checkpoint_profiles,
    phase7_optimizer_profiles,
    record_training_peak_bytes,
    set_detailed_profiling_enabled,
)
from barqtrain.checkpointing import (
    apply_activation_checkpointing,
    reset_activation_checkpointing,
)
from barqtrain.ops import (
    chunked_cross_entropy_loss,
    padding_free_attention,
    padding_free_chunked_cross_entropy_loss,
)
from barqtrain.optim import create_optimizer, optimizer_state_bytes
from barqtrain.patch_models import patch_inference, patch_model


@dataclass
class BenchmarkMetrics:
    """Training benchmark metrics."""

    model_name: str
    batch_size: int
    sequence_length: int
    total_steps: int
    total_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    avg_step_time_seconds: float
    memory: BenchmarkMemoryBreakdown = field(default_factory=BenchmarkMemoryBreakdown)
    packing_enabled: bool = False
    optimizer_name: str = "adamw"
    gpu_utilization_percent: Optional[float] = None

    @property
    def peak_vram_mb(self) -> float:
        return self.memory.training_peak_vram_mb


@dataclass
class InferenceBenchmarkMetrics:
    """Inference benchmark metrics for a single decode profile."""

    profile_name: str
    batch_size: int
    prompt_length: int
    decode_length: int
    total_new_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    paged_kv_cache: bool
    last_token_logits_only: bool
    memory: BenchmarkMemoryBreakdown = field(default_factory=BenchmarkMemoryBreakdown)


@dataclass
class KVCacheBenchmarkMetrics:
    """Inference benchmark metrics for contiguous-vs-paged KV scenarios."""

    scenario_name: str
    cache_layout: str
    batch_size: int
    prompt_length: int
    decode_length: int
    requests_attempted: int
    requests_succeeded: int
    total_new_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    oom_rate: float
    fragmentation_ratio: float
    resident_vram_mb: float
    peak_vram_mb: float
    fixed_vram_budget_mb: Optional[float] = None
    memory: BenchmarkMemoryBreakdown = field(default_factory=BenchmarkMemoryBreakdown)


@dataclass
class QuantizedKVBenchmarkMetrics:
    """Inference benchmark metrics for quantized KV-cache quality and memory tradeoffs."""

    scenario_name: str
    cache_layout: str
    batch_size: int
    prompt_length: int
    decode_length: int
    total_new_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    resident_vram_mb: float
    peak_vram_mb: float
    throughput_per_gb: float
    memory_savings_vs_contiguous_percent: float
    latency_vs_contiguous_percent: float
    generation_match_ratio: float
    perplexity: Optional[float] = None
    reference_perplexity: Optional[float] = None
    memory: BenchmarkMemoryBreakdown = field(default_factory=BenchmarkMemoryBreakdown)


@dataclass
class ProjectionBenchmarkMetrics:
    """Benchmark metrics for Phase 4 fused projection/loss comparisons."""

    scenario_name: str
    projection_mode: str
    batch_size: int
    sequence_length: int
    prompt_length: int
    decode_length: int
    vocab_size: int
    training_step_time_seconds: float
    decode_tokens_per_second: float
    training_peak_vram_mb: float
    inference_peak_vram_mb: float
    training_loss: float
    loss_delta_vs_baseline: float
    generation_match_ratio: float
    last_token_logits_only: bool
    memory: BenchmarkMemoryBreakdown = field(default_factory=BenchmarkMemoryBreakdown)


@dataclass
class PackedTrainingBenchmarkMetrics:
    """Benchmark metrics for Phase 5 packed-vs-padded training comparisons."""

    scenario_name: str
    packing_mode: str
    batch_size: int
    sequence_length: int
    document_masked: bool
    effective_tokens: int
    step_time_seconds: float
    effective_tokens_per_second: float
    throughput_at_matched_effective_tokens: float
    peak_vram_mb: float
    loss_value: float
    loss_delta_vs_padded: float
    memory: BenchmarkMemoryBreakdown = field(default_factory=BenchmarkMemoryBreakdown)


@dataclass
class ActivationCheckpointBenchmarkMetrics:
    """Benchmark metrics for Phase 6 activation-checkpoint presets."""

    preset_name: str
    total_steps: int
    total_tokens: int
    tokens_per_second: float
    avg_step_time_seconds: float
    peak_vram_mb: float
    loss_stddev: float
    loss_delta_vs_max_throughput: float
    memory: BenchmarkMemoryBreakdown = field(default_factory=BenchmarkMemoryBreakdown)


@dataclass
class OptimizerBenchmarkMetrics:
    """Benchmark metrics for Phase 7 optimizer-state tradeoffs."""

    optimizer_name: str
    total_steps: int
    total_tokens: int
    tokens_per_second: float
    avg_step_time_seconds: float
    optimizer_state_mb: float
    loss_delta_vs_adamw: float
    memory: BenchmarkMemoryBreakdown = field(default_factory=BenchmarkMemoryBreakdown)


@dataclass
class BenchmarkReport:
    """Combined training + inference benchmark report."""

    model_name: str
    optimizer_name: str
    detailed_profiling: bool
    benchmark_suite: str = "phase1"
    training: Optional[BenchmarkMetrics] = None
    inference_profiles: list[InferenceBenchmarkMetrics] = field(default_factory=list)
    kv_cache_profiles: list[KVCacheBenchmarkMetrics] = field(default_factory=list)
    quantized_kv_profiles: list[QuantizedKVBenchmarkMetrics] = field(default_factory=list)
    projection_profiles: list[ProjectionBenchmarkMetrics] = field(default_factory=list)
    packed_training_profiles: list[PackedTrainingBenchmarkMetrics] = field(default_factory=list)
    checkpoint_profiles: list[ActivationCheckpointBenchmarkMetrics] = field(default_factory=list)
    optimizer_profiles: list[OptimizerBenchmarkMetrics] = field(default_factory=list)


@contextmanager
def _temporary_env(name: str, value: str):
    original = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


class BenchmarkHarness:
    """Main benchmark harness for training and decode-focused inference profiles."""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        batch_size: int = 4,
        sequence_length: int = 512,
        num_steps: int = 100,
        use_packing: bool = False,
        optimizer_name: str = "adamw",
        output_dir: Optional[str] = None,
        detailed_profiling: bool = False,
        inference_batch_sizes: Sequence[int] = (1, 4, 8),
        short_prompt_length: int = 64,
        long_prompt_length: int = 1024,
        short_decode_length: int = 32,
        long_decode_length: int = 256,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_steps = num_steps
        self.use_packing = use_packing
        self.optimizer_name = optimizer_name
        self.output_dir = Path(output_dir) if output_dir else Path("benchmarks/results")
        self.detailed_profiling = detailed_profiling
        self.inference_batch_sizes = tuple(int(size) for size in inference_batch_sizes)
        self.short_prompt_length = short_prompt_length
        self.long_prompt_length = long_prompt_length
        self.short_decode_length = short_decode_length
        self.long_decode_length = long_decode_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self._inference_patched = False

    def setup_model_and_tokenizer(self) -> None:
        """Initialize model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            return

        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = torch.float32
        if self.device.type == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self.device)

    def prepare_dataset(self) -> DataLoader:
        """Prepare a small dataset for benchmarking."""
        print("Preparing dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.sequence_length,
                padding=False if self.use_packing else "max_length",
                return_overflowing_tokens=False,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        collate_fn = None
        if self.use_packing:
            collate_fn = PackedCausalLMDataCollator(
                max_length=self.sequence_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
        )

    def _sync_device(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _ensure_inference_patch(self) -> None:
        if not self._inference_patched:
            self.model = patch_inference(self.model)
            self._inference_patched = True

    def _sequence_output(self, outputs):
        return getattr(outputs, "sequences", outputs)

    def _build_prompt_inputs(self, prompt_length: int, batch_size: int) -> dict[str, torch.Tensor]:
        prompt = " ".join(["barqtrain"] * max(prompt_length, 1))
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        if input_ids.size(1) == 0:
            input_ids = torch.ones((1, 1), dtype=torch.long)
        if input_ids.size(1) < prompt_length:
            repeat_factor = math.ceil(prompt_length / input_ids.size(1))
            input_ids = input_ids.repeat(1, repeat_factor)
        input_ids = input_ids[:, :prompt_length].repeat(batch_size, 1)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    def _last_generate_measurements(self) -> dict[str, object]:
        resident_model_bytes = int(
            getattr(
                self.model,
                "_barqtrain_last_generate_resident_model_bytes",
                model_resident_cuda_bytes(self.model),
            )
        )
        cache = getattr(self.model, "_barqtrain_last_generate_cache", None)
        kv_cache_bytes = int(
            getattr(
                self.model,
                "_barqtrain_last_generate_kv_cache_bytes",
                paged_kv_cache_bytes(cache) if cache is not None else 0,
            )
        )
        inference_peak_bytes = int(
            getattr(
                self.model,
                "_barqtrain_last_generate_inference_peak_bytes",
                capture_cuda_peak_bytes(),
            )
        )
        decode_temp_bytes = int(
            getattr(
                self.model,
                "_barqtrain_last_generate_decode_temp_bytes",
                max(inference_peak_bytes - resident_model_bytes - kv_cache_bytes, 0),
            )
        )
        memory = build_memory_breakdown(
            resident_model_bytes=resident_model_bytes,
            kv_cache_bytes=kv_cache_bytes,
            temporary_decode_buffer_bytes=decode_temp_bytes,
            training_peak_bytes=0,
            inference_peak_bytes=inference_peak_bytes,
            detailed_profiling=self.detailed_profiling,
        )
        return {
            "resident_model_bytes": resident_model_bytes,
            "kv_cache_bytes": kv_cache_bytes,
            "decode_temp_bytes": decode_temp_bytes,
            "inference_peak_bytes": inference_peak_bytes,
            "cache": cache,
            "memory": memory,
        }

    @staticmethod
    def _cache_fragmentation_ratio(cache) -> float:
        if cache is None or not hasattr(cache, "fragmentation_ratio"):
            return 0.0
        fragmentation = cache.fragmentation_ratio
        return float(fragmentation() if callable(fragmentation) else fragmentation)

    @staticmethod
    def _is_oom_like_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "out of memory",
                "allocator exhausted",
                "capacity exceeded",
            )
        )

    def _run_generate_call(
        self,
        *,
        prompt_length: int,
        batch_size: int,
        decode_length: int,
        prefer_last_token_logits: bool = True,
    ) -> dict[str, object]:
        inputs = self._build_prompt_inputs(prompt_length, batch_size)
        generation_kwargs = build_generation_kwargs(
            self.model,
            decode_length,
            prefer_last_token_logits=prefer_last_token_logits,
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._sync_device()
        start_time = time.time()
        outputs = self.model.generate(**inputs, **generation_kwargs)
        self._sync_device()
        total_time = time.time() - start_time

        output_sequences = self._sequence_output(outputs)
        output_length = int(output_sequences.shape[-1])
        input_length = int(inputs["input_ids"].shape[-1])
        total_new_tokens = max(output_length - input_length, 0) * int(output_sequences.shape[0])
        measurements = self._last_generate_measurements()
        measurements.update(
            {
                "outputs": output_sequences,
                "total_time": total_time,
                "total_new_tokens": total_new_tokens,
                "cache_layout": getattr(self.model, "_barqtrain_last_generate_kv_cache_layout", None),
                "last_token_logits_only": bool(
                    getattr(self.model, "_barqtrain_last_generate_last_token_logits_only", False)
                ),
            }
        )
        return measurements

    def _sequence_perplexity(self, sequences: torch.Tensor) -> Optional[float]:
        try:
            outputs = self.model(
                input_ids=sequences,
                attention_mask=torch.ones_like(sequences),
                labels=sequences,
            )
        except Exception:
            return None

        loss = getattr(outputs, "loss", None)
        if loss is None:
            return None
        loss_value = float(loss.detach().float().item())
        return float(math.exp(min(loss_value, 20.0)))

    @staticmethod
    def _generation_match_ratio(reference: torch.Tensor, candidate: torch.Tensor, prompt_length: int) -> float:
        if reference.shape != candidate.shape:
            return 0.0
        reference_tokens = reference[:, prompt_length:]
        candidate_tokens = candidate[:, prompt_length:]
        if reference_tokens.numel() == 0:
            return 1.0
        return float(reference_tokens.eq(candidate_tokens).float().mean().item())

    def run_benchmark(self) -> BenchmarkMetrics:
        """Run the training benchmark."""
        print(f"\n{'='*60}")
        print("Starting Training Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Steps: {self.num_steps}")
        print(f"Packing Enabled: {self.use_packing}")
        print(f"Optimizer: {self.optimizer_name}")
        print(f"Detailed Profiling: {self.detailed_profiling}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        set_detailed_profiling_enabled(self.detailed_profiling)
        self.setup_model_and_tokenizer()
        dataloader = self.prepare_dataset()
        self.model.train()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        resident_model_bytes = model_resident_cuda_bytes(self.model)
        start_time = time.time()
        total_tokens = 0
        data_iter = iter(dataloader)
        optimizer = create_optimizer(
            self.model.parameters(),
            lr=1e-5,
            optimizer_name=self.optimizer_name,
        )

        for step in range(self.num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch.get("labels")
            labels = input_ids if labels is None else labels.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_tokens += labels.ne(-100).sum().item()

            if (step + 1) % 10 == 0:
                elapsed = time.time() - start_time
                tokens_sec = total_tokens / max(elapsed, 1e-9)
                print(
                    f"Step {step + 1}/{self.num_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Tokens/s: {tokens_sec:.1f} | "
                    f"Training Peak: {capture_cuda_peak_bytes() / (1024**2):.1f} MB"
                )

        total_time = time.time() - start_time
        training_peak_bytes = capture_cuda_peak_bytes()
        record_training_peak_bytes(training_peak_bytes)
        memory = build_memory_breakdown(
            resident_model_bytes=resident_model_bytes,
            kv_cache_bytes=0,
            temporary_decode_buffer_bytes=0,
            training_peak_bytes=training_peak_bytes,
            inference_peak_bytes=0,
            detailed_profiling=self.detailed_profiling,
        )

        return BenchmarkMetrics(
            model_name=self.model_name,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            total_steps=self.num_steps,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            tokens_per_second=total_tokens / max(total_time, 1e-9),
            avg_step_time_seconds=total_time / max(self.num_steps, 1),
            memory=memory,
            packing_enabled=self.use_packing,
            optimizer_name=self.optimizer_name,
        )

    def run_inference_benchmarks(self) -> list[InferenceBenchmarkMetrics]:
        """Run the Phase 1 decode benchmark matrix."""
        print(f"\n{'='*60}")
        print("Starting Inference Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Batch Sizes: {', '.join(str(size) for size in self.inference_batch_sizes)}")
        print(f"Detailed Profiling: {self.detailed_profiling}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        set_detailed_profiling_enabled(self.detailed_profiling)
        self.setup_model_and_tokenizer()
        self.model.eval()
        self._ensure_inference_patch()

        profiles = phase1_inference_profiles(
            self.inference_batch_sizes,
            short_prompt_length=self.short_prompt_length,
            long_prompt_length=self.long_prompt_length,
            short_decode_length=self.short_decode_length,
            long_decode_length=self.long_decode_length,
        )

        metrics: list[InferenceBenchmarkMetrics] = []
        with torch.inference_mode():
            for profile in profiles:
                run = self._run_generate_call(
                    prompt_length=profile.prompt_length,
                    batch_size=profile.batch_size,
                    decode_length=profile.decode_length,
                )
                metric = InferenceBenchmarkMetrics(
                    profile_name=profile.name,
                    batch_size=profile.batch_size,
                    prompt_length=profile.prompt_length,
                    decode_length=profile.decode_length,
                    total_new_tokens=int(run["total_new_tokens"]),
                    total_time_seconds=float(run["total_time"]),
                    tokens_per_second=float(run["total_new_tokens"]) / max(float(run["total_time"]), 1e-9),
                    paged_kv_cache=bool(getattr(self.model, "_barqtrain_last_generate_used_paged_kv", False)),
                    last_token_logits_only=bool(run["last_token_logits_only"]),
                    memory=run["memory"],
                )
                metrics.append(metric)

                print(
                    f"{metric.profile_name} | bs={metric.batch_size} | "
                    f"prompt={metric.prompt_length} | decode={metric.decode_length} | "
                    f"tokens/s={metric.tokens_per_second:.1f} | "
                    f"resident={metric.memory.resident_model_mb:.1f} MB | "
                    f"kv={metric.memory.kv_cache_mb:.1f} MB | "
                    f"decode_temp={metric.memory.temporary_decode_buffers_mb:.1f} MB | "
                    f"inference_peak={metric.memory.inference_peak_vram_mb:.1f} MB"
                )

        return metrics

    def run_phase2_kv_benchmarks(
        self,
        *,
        cache_layouts: Sequence[str] = ("contiguous", "paged"),
        serving_request_count: int = 8,
        fixed_vram_budget_mb: float = 2048.0,
    ) -> list[KVCacheBenchmarkMetrics]:
        """Run the Phase 2 contiguous-vs-paged KV benchmark matrix."""
        print(f"\n{'='*60}")
        print("Starting Phase 2 KV Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Cache Layouts: {', '.join(cache_layouts)}")
        print(f"Batch Sizes: {', '.join(str(size) for size in self.inference_batch_sizes)}")
        print(f"Detailed Profiling: {self.detailed_profiling}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        set_detailed_profiling_enabled(self.detailed_profiling)
        self.setup_model_and_tokenizer()
        self.model.eval()
        self._ensure_inference_patch()

        profiles = phase2_kv_cache_profiles(
            self.inference_batch_sizes,
            short_prompt_length=self.short_prompt_length,
            long_prompt_length=self.long_prompt_length,
            short_decode_length=self.short_decode_length,
            long_decode_length=self.long_decode_length,
            serving_request_count=serving_request_count,
            fixed_vram_budget_mb=int(fixed_vram_budget_mb),
        )

        metrics: list[KVCacheBenchmarkMetrics] = []
        with torch.inference_mode():
            for cache_layout in cache_layouts:
                for profile in profiles:
                    max_resident_model_bytes = 0
                    max_kv_cache_bytes = 0
                    max_decode_temp_bytes = 0
                    max_inference_peak_bytes = 0
                    resolved_cache_layout = cache_layout
                    fragmentations: list[float] = []
                    total_time = 0.0
                    total_new_tokens = 0
                    requests_attempted = int(profile.request_count)
                    requests_succeeded = 0

                    with _temporary_env("BARQTRAIN_KV_CACHE_MODE", cache_layout):
                        for _ in range(requests_attempted):
                            try:
                                run = self._run_generate_call(
                                    prompt_length=profile.prompt_length,
                                    batch_size=profile.batch_size,
                                    decode_length=profile.decode_length,
                                )
                            except Exception as exc:
                                if self._is_oom_like_error(exc):
                                    continue
                                raise

                            resident_model_bytes = int(run["resident_model_bytes"])
                            kv_cache_bytes = int(run["kv_cache_bytes"])
                            decode_temp_bytes = int(run["decode_temp_bytes"])
                            inference_peak_bytes = int(run["inference_peak_bytes"])
                            resolved_cache_layout = str(run["cache_layout"] or resolved_cache_layout)
                            resident_total_mb = (resident_model_bytes + kv_cache_bytes) / (1024**2)
                            peak_vram_mb = inference_peak_bytes / (1024**2)

                            max_resident_model_bytes = max(max_resident_model_bytes, resident_model_bytes)
                            max_kv_cache_bytes = max(max_kv_cache_bytes, kv_cache_bytes)
                            max_decode_temp_bytes = max(max_decode_temp_bytes, decode_temp_bytes)
                            max_inference_peak_bytes = max(max_inference_peak_bytes, inference_peak_bytes)
                            fragmentations.append(self._cache_fragmentation_ratio(run["cache"]))

                            budget_mb = float(profile.fixed_vram_budget_mb)
                            if budget_mb > 0 and max(resident_total_mb, peak_vram_mb) > budget_mb:
                                continue

                            requests_succeeded += 1
                            total_time += float(run["total_time"])
                            total_new_tokens += int(run["total_new_tokens"])

                    oom_rate = 1.0 - (requests_succeeded / max(requests_attempted, 1))
                    memory = build_memory_breakdown(
                        resident_model_bytes=max_resident_model_bytes,
                        kv_cache_bytes=max_kv_cache_bytes,
                        temporary_decode_buffer_bytes=max_decode_temp_bytes,
                        training_peak_bytes=0,
                        inference_peak_bytes=max_inference_peak_bytes,
                        detailed_profiling=self.detailed_profiling,
                    )
                    metric = KVCacheBenchmarkMetrics(
                        scenario_name=profile.name,
                        cache_layout=resolved_cache_layout,
                        batch_size=profile.batch_size,
                        prompt_length=profile.prompt_length,
                        decode_length=profile.decode_length,
                        requests_attempted=requests_attempted,
                        requests_succeeded=requests_succeeded,
                        total_new_tokens=total_new_tokens,
                        total_time_seconds=total_time,
                        tokens_per_second=total_new_tokens / max(total_time, 1e-9),
                        oom_rate=oom_rate,
                        fragmentation_ratio=(
                            sum(fragmentations) / len(fragmentations) if fragmentations else 0.0
                        ),
                        resident_vram_mb=memory.resident_model_mb + memory.kv_cache_mb,
                        peak_vram_mb=memory.inference_peak_vram_mb,
                        fixed_vram_budget_mb=(
                            float(profile.fixed_vram_budget_mb) if profile.fixed_vram_budget_mb > 0 else None
                        ),
                        memory=memory,
                    )
                    metrics.append(metric)

                    budget_label = (
                        f" | budget={metric.fixed_vram_budget_mb:.0f} MB"
                        if metric.fixed_vram_budget_mb is not None
                        else ""
                    )
                    print(
                        f"{metric.scenario_name} | layout={metric.cache_layout} | "
                        f"bs={metric.batch_size} | requests={metric.requests_succeeded}/{metric.requests_attempted} | "
                        f"tokens/s={metric.tokens_per_second:.1f} | oom_rate={metric.oom_rate:.2f} | "
                        f"frag={metric.fragmentation_ratio:.2f} | "
                        f"resident={metric.resident_vram_mb:.1f} MB | peak={metric.peak_vram_mb:.1f} MB"
                        f"{budget_label}"
                    )

        return metrics

    def _run_projection_training_step(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        fused_projection: bool,
    ) -> dict[str, float | int]:
        self.model = patch_model(self.model)
        self.model.train()
        setattr(self.model, "_barqtrain_chunked_loss_enabled", fused_projection)

        inputs = self._build_prompt_inputs(sequence_length, batch_size)
        labels = inputs["input_ids"].clone()
        resident_model_bytes = model_resident_cuda_bytes(self.model)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.model.zero_grad(set_to_none=True)
        self._sync_device()
        start_time = time.time()
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("projection benchmark expected the model to return a loss")
        loss.backward()
        self._sync_device()
        total_time = time.time() - start_time
        training_peak_bytes = capture_cuda_peak_bytes()
        record_training_peak_bytes(training_peak_bytes)
        self.model.zero_grad(set_to_none=True)

        return {
            "resident_model_bytes": resident_model_bytes,
            "training_peak_bytes": training_peak_bytes,
            "training_step_time_seconds": total_time,
            "training_loss": float(loss.detach().float().item()),
        }

    def run_phase4_vocab_projection_benchmarks(self) -> list[ProjectionBenchmarkMetrics]:
        """Run the Phase 4 fused projection-plus-loss benchmark matrix."""
        print(f"\n{'='*60}")
        print("Starting Phase 4 Projection Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Batch Sizes: {', '.join(str(size) for size in self.inference_batch_sizes)}")
        print(f"Detailed Profiling: {self.detailed_profiling}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        set_detailed_profiling_enabled(self.detailed_profiling)
        self.setup_model_and_tokenizer()
        self.model = patch_model(self.model)
        self._inference_patched = True

        profiles = phase4_vocab_projection_profiles(
            self.inference_batch_sizes,
            sequence_length=self.sequence_length,
            short_prompt_length=self.short_prompt_length,
            long_prompt_length=self.long_prompt_length,
            short_decode_length=self.short_decode_length,
            long_decode_length=self.long_decode_length,
        )

        metrics: list[ProjectionBenchmarkMetrics] = []
        baselines: dict[tuple[str, int], dict[str, object]] = {}

        for projection_mode, fused_projection in (("baseline", False), ("fused", True)):
            for profile in profiles:
                training = self._run_projection_training_step(
                    batch_size=profile.batch_size,
                    sequence_length=profile.sequence_length,
                    fused_projection=fused_projection,
                )
                self.model.eval()
                setattr(self.model, "_barqtrain_last_token_projection_enabled", fused_projection)

                with torch.inference_mode():
                    with _temporary_env(
                        "BARQTRAIN_LAST_TOKEN_LOGITS_ONLY",
                        "1" if fused_projection else "0",
                    ):
                        run = self._run_generate_call(
                            prompt_length=profile.prompt_length,
                            batch_size=profile.batch_size,
                            decode_length=profile.decode_length,
                            prefer_last_token_logits=fused_projection,
                        )

                key = (profile.name, profile.batch_size)
                baseline = baselines.get(key)
                if baseline is None:
                    baseline = {
                        "outputs": run["outputs"],
                        "training_loss": float(training["training_loss"]),
                    }
                    baselines[key] = baseline

                memory = build_memory_breakdown(
                    resident_model_bytes=int(training["resident_model_bytes"]),
                    kv_cache_bytes=int(run["kv_cache_bytes"]),
                    temporary_decode_buffer_bytes=int(run["decode_temp_bytes"]),
                    training_peak_bytes=int(training["training_peak_bytes"]),
                    inference_peak_bytes=int(run["inference_peak_bytes"]),
                    detailed_profiling=self.detailed_profiling,
                )
                generation_match_ratio = self._generation_match_ratio(
                    baseline["outputs"],
                    run["outputs"],
                    profile.prompt_length,
                )
                metric = ProjectionBenchmarkMetrics(
                    scenario_name=profile.name,
                    projection_mode=projection_mode,
                    batch_size=profile.batch_size,
                    sequence_length=profile.sequence_length,
                    prompt_length=profile.prompt_length,
                    decode_length=profile.decode_length,
                    vocab_size=int(self.model.lm_head.weight.shape[0]),
                    training_step_time_seconds=float(training["training_step_time_seconds"]),
                    decode_tokens_per_second=float(run["total_new_tokens"]) / max(float(run["total_time"]), 1e-9),
                    training_peak_vram_mb=memory.training_peak_vram_mb,
                    inference_peak_vram_mb=memory.inference_peak_vram_mb,
                    training_loss=float(training["training_loss"]),
                    loss_delta_vs_baseline=float(training["training_loss"]) - float(baseline["training_loss"]),
                    generation_match_ratio=generation_match_ratio,
                    last_token_logits_only=bool(run["last_token_logits_only"]),
                    memory=memory,
                )
                metrics.append(metric)

                print(
                    f"{metric.scenario_name} | mode={metric.projection_mode} | "
                    f"bs={metric.batch_size} | train_step={metric.training_step_time_seconds:.4f}s | "
                    f"decode={metric.decode_tokens_per_second:.1f} tok/s | "
                    f"train_peak={metric.training_peak_vram_mb:.1f} MB | "
                    f"inference_peak={metric.inference_peak_vram_mb:.1f} MB | "
                    f"loss_delta={metric.loss_delta_vs_baseline:.6f} | "
                    f"match={metric.generation_match_ratio:.3f}"
                )

        return metrics

    def run_phase4_benchmarks(self) -> BenchmarkReport:
        """Run the requested Phase 4 fused projection benchmark suite."""
        return BenchmarkReport(
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            detailed_profiling=self.detailed_profiling,
            benchmark_suite="phase4",
            projection_profiles=self.run_phase4_vocab_projection_benchmarks(),
        )

    def _phase5_examples(self, batch_size: int, sequence_length: int) -> list[dict[str, object]]:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", 0)
        examples: list[dict[str, object]] = []
        for index in range(batch_size):
            length_delta = (index % 3) * max(sequence_length // 4, 1)
            length = max(3, sequence_length - length_delta)
            tokens = [((position + index) % 31) + 1 for position in range(length - 1)]
            tokens.append(int(eos_token_id))
            examples.append(
                {
                    "input_ids": tokens,
                    "attention_mask": [1] * len(tokens),
                    "document_id": index,
                }
            )
        return examples

    @staticmethod
    def _phase5_attention_layout(hidden_size: int) -> tuple[int, int]:
        num_heads = min(8, max(hidden_size, 1))
        while num_heads > 1 and hidden_size % num_heads != 0:
            num_heads -= 1
        return num_heads, max(hidden_size // max(num_heads, 1), 1)

    def _build_padded_training_batch(
        self,
        examples: Sequence[dict[str, object]],
        sequence_length: int,
    ) -> dict[str, torch.Tensor]:
        pad_token_id = int(getattr(self.tokenizer, "pad_token_id", 0))
        batch_size = len(examples)
        input_ids = torch.full((batch_size, sequence_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long)
        labels = torch.full((batch_size, sequence_length), -100, dtype=torch.long)

        for index, example in enumerate(examples):
            tokens = list(example["input_ids"])[:sequence_length]
            length = len(tokens)
            if length == 0:
                continue
            input_ids[index, :length] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[index, :length] = 1
            labels[index, :length] = torch.tensor(tokens, dtype=torch.long)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
        }

    def _run_phase5_training_step(
        self,
        *,
        packing_mode: str,
        batch_size: int,
        sequence_length: int,
        document_masked: bool,
    ) -> dict[str, float | int]:
        if self.model is None:
            self.setup_model_and_tokenizer()
        if self.model is None or not hasattr(self.model, "get_input_embeddings") or not hasattr(self.model, "lm_head"):
            raise RuntimeError("phase5 benchmark requires a model with embeddings and an lm_head")

        examples = self._phase5_examples(batch_size, sequence_length)
        embed_layer = self.model.get_input_embeddings()
        resident_model_bytes = model_resident_cuda_bytes(self.model)
        self.model.zero_grad(set_to_none=True)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._sync_device()
        start_time = time.time()

        if packing_mode == "padded":
            batch = self._build_padded_training_batch(examples, sequence_length)
            hidden_states = embed_layer(batch["input_ids"])
            hidden_states = hidden_states * batch["attention_mask"].unsqueeze(-1)
            num_heads, head_dim = self._phase5_attention_layout(hidden_states.size(-1))
            qkv = hidden_states.reshape(hidden_states.size(0), hidden_states.size(1), num_heads, head_dim)
            qkv = qkv.permute(0, 2, 1, 3)
            attended = torch.nn.functional.scaled_dot_product_attention(
                qkv,
                qkv,
                qkv,
                attn_mask=None,
                is_causal=True,
            )
            attended = attended.permute(0, 2, 1, 3).reshape(hidden_states.size(0), hidden_states.size(1), -1)
            loss = chunked_cross_entropy_loss(
                attended[:, :-1, :],
                self.model.lm_head.weight,
                batch["labels"][:, 1:],
            )
            effective_tokens = int(batch["labels"][:, 1:].ne(-100).sum().item())
        else:
            collator = PaddingFreeCausalLMDataCollator(
                max_length=sequence_length,
                pad_token_id=int(getattr(self.tokenizer, "pad_token_id", 0)),
                eos_token_id=int(getattr(self.tokenizer, "eos_token_id", 0)),
                document_masked=document_masked,
            )
            batch = collator(examples)
            input_ids = batch["input_ids"].to(self.device)
            hidden_states = embed_layer(input_ids)
            num_heads, head_dim = self._phase5_attention_layout(hidden_states.size(-1))

            flat_hidden_segments = []
            flat_labels = []
            flat_loss_masks = []
            global_cu_seqlens = [0]
            running_tokens = 0

            for index in range(hidden_states.size(0)):
                active_tokens = int(batch["active_tokens"][index].item())
                if active_tokens <= 0:
                    continue
                flat_hidden_segments.append(
                    hidden_states[index, :active_tokens, :].reshape(active_tokens, num_heads, head_dim)
                )
                flat_labels.append(batch["labels"][index, :active_tokens])
                loss_mask_block = batch["loss_mask"][index, :active_tokens].clone()
                if index > 0 and active_tokens > 0:
                    loss_mask_block[0] = 0
                flat_loss_masks.append(loss_mask_block)

                cu_values = batch["cu_seqlens"][index].tolist()
                deduped = [0]
                for value in cu_values[1:]:
                    value = int(value)
                    if value == deduped[-1]:
                        continue
                    deduped.append(value)
                    if value >= active_tokens:
                        break
                if deduped[-1] != active_tokens:
                    deduped.append(active_tokens)
                for value in deduped[1:]:
                    global_cu_seqlens.append(running_tokens + value)
                running_tokens += active_tokens

            hidden_flat = torch.cat(flat_hidden_segments, dim=0).to(self.device)
            labels_flat = torch.cat(flat_labels, dim=0).to(self.device)
            loss_mask_flat = torch.cat(flat_loss_masks, dim=0).to(self.device)
            cu_seqlens = torch.tensor(global_cu_seqlens, device=self.device, dtype=torch.long)
            attended = padding_free_attention(
                hidden_flat,
                hidden_flat,
                hidden_flat,
                cu_seqlens=cu_seqlens,
            ).reshape(1, hidden_flat.size(0), -1)
            loss = padding_free_chunked_cross_entropy_loss(
                attended[:, :-1, :],
                self.model.lm_head.weight,
                labels_flat[1:].unsqueeze(0),
                loss_mask=loss_mask_flat[1:].unsqueeze(0),
            )
            effective_tokens = int(loss_mask_flat[1:].sum().item())

        loss.backward()
        self._sync_device()
        total_time = time.time() - start_time
        training_peak_bytes = capture_cuda_peak_bytes()
        record_training_peak_bytes(training_peak_bytes)
        self.model.zero_grad(set_to_none=True)

        return {
            "resident_model_bytes": resident_model_bytes,
            "training_peak_bytes": training_peak_bytes,
            "step_time_seconds": total_time,
            "effective_tokens": effective_tokens,
            "loss_value": float(loss.detach().float().item()),
        }

    def run_phase5_packed_training_benchmarks(self) -> list[PackedTrainingBenchmarkMetrics]:
        """Run the Phase 5 padded-vs-packed training benchmark matrix."""
        print(f"\n{'='*60}")
        print("Starting Phase 5 Packed Training Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Batch Sizes: {', '.join(str(size) for size in self.inference_batch_sizes)}")
        print(f"Detailed Profiling: {self.detailed_profiling}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        set_detailed_profiling_enabled(self.detailed_profiling)
        self.setup_model_and_tokenizer()
        self.model.train()

        profiles = phase5_packed_training_profiles(
            self.inference_batch_sizes,
            sequence_length=self.sequence_length,
        )

        metrics: list[PackedTrainingBenchmarkMetrics] = []
        padded_baselines: dict[tuple[str, int, bool], dict[str, float | int]] = {}

        for packing_mode in ("padded", "packed"):
            for profile in profiles:
                result = self._run_phase5_training_step(
                    packing_mode=packing_mode,
                    batch_size=profile.batch_size,
                    sequence_length=profile.sequence_length,
                    document_masked=profile.document_masked,
                )
                key = (profile.name, profile.batch_size, profile.document_masked)
                baseline = padded_baselines.get(key)
                if baseline is None:
                    baseline = result
                    padded_baselines[key] = result

                memory = build_memory_breakdown(
                    resident_model_bytes=int(result["resident_model_bytes"]),
                    kv_cache_bytes=0,
                    temporary_decode_buffer_bytes=0,
                    training_peak_bytes=int(result["training_peak_bytes"]),
                    inference_peak_bytes=0,
                    detailed_profiling=self.detailed_profiling,
                )
                metric = PackedTrainingBenchmarkMetrics(
                    scenario_name=profile.name,
                    packing_mode=packing_mode,
                    batch_size=profile.batch_size,
                    sequence_length=profile.sequence_length,
                    document_masked=profile.document_masked,
                    effective_tokens=int(result["effective_tokens"]),
                    step_time_seconds=float(result["step_time_seconds"]),
                    effective_tokens_per_second=(
                        float(result["effective_tokens"]) / max(float(result["step_time_seconds"]), 1e-9)
                    ),
                    throughput_at_matched_effective_tokens=(
                        float(result["effective_tokens"]) / max(float(result["step_time_seconds"]), 1e-9)
                    ),
                    peak_vram_mb=memory.training_peak_vram_mb,
                    loss_value=float(result["loss_value"]),
                    loss_delta_vs_padded=float(result["loss_value"]) - float(baseline["loss_value"]),
                    memory=memory,
                )
                metrics.append(metric)

                print(
                    f"{metric.scenario_name} | mode={metric.packing_mode} | "
                    f"bs={metric.batch_size} | masked={metric.document_masked} | "
                    f"eff_tok/s={metric.effective_tokens_per_second:.1f} | "
                    f"peak={metric.peak_vram_mb:.1f} MB | "
                    f"loss_delta={metric.loss_delta_vs_padded:.6f}"
                )

        return metrics

    def run_phase5_benchmarks(self) -> BenchmarkReport:
        """Run the requested Phase 5 packed training benchmark suite."""
        return BenchmarkReport(
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            detailed_profiling=self.detailed_profiling,
            benchmark_suite="phase5",
            packed_training_profiles=self.run_phase5_packed_training_benchmarks(),
        )

    def run_phase6_activation_checkpoint_benchmarks(self) -> list[ActivationCheckpointBenchmarkMetrics]:
        """Run the Phase 6 activation-checkpoint preset benchmark suite."""
        print(f"\n{'='*60}")
        print("Starting Phase 6 Activation Checkpoint Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Detailed Profiling: {self.detailed_profiling}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        set_detailed_profiling_enabled(self.detailed_profiling)
        self.setup_model_and_tokenizer()
        if hasattr(self.model, "model") and hasattr(self.model, "lm_head"):
            self.model = patch_model(self.model)
        dataloader = self.prepare_dataset()
        profiles = phase6_activation_checkpoint_profiles(num_steps=min(self.num_steps, 3) or 1)
        base_state = copy.deepcopy(self.model.state_dict())

        metrics: list[ActivationCheckpointBenchmarkMetrics] = []
        baselines: dict[str, dict[str, float | int]] = {}

        for profile in profiles:
            self.model.load_state_dict(base_state)
            reset_activation_checkpointing(self.model)
            apply_activation_checkpointing(self.model, preset=profile.name)
            self.model.train()
            optimizer = create_optimizer(
                self.model.parameters(),
                lr=1e-5,
                optimizer_name=self.optimizer_name,
            )
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            resident_model_bytes = model_resident_cuda_bytes(self.model)
            total_tokens = 0
            losses = []
            total_time = 0.0

            data_iter = iter(dataloader)
            for _ in range(profile.num_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels")
                labels = input_ids if labels is None else labels.to(self.device)

                self.model.zero_grad(set_to_none=True)
                self._sync_device()
                start_time = time.time()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("phase6 benchmark expected a training loss")
                loss.backward()
                optimizer.step()
                self._sync_device()
                total_time += time.time() - start_time

                loss_value = float(loss.detach().float().item())
                losses.append(loss_value)
                total_tokens += int(labels.ne(-100).sum().item())

            training_peak_bytes = capture_cuda_peak_bytes()
            record_training_peak_bytes(training_peak_bytes)
            memory = build_memory_breakdown(
                resident_model_bytes=resident_model_bytes,
                kv_cache_bytes=0,
                temporary_decode_buffer_bytes=0,
                training_peak_bytes=training_peak_bytes,
                inference_peak_bytes=0,
                detailed_profiling=self.detailed_profiling,
            )
            baseline = baselines.get("max_throughput")
            if baseline is None:
                baseline = {
                    "loss_mean": float(sum(losses) / max(len(losses), 1)),
                }
                baselines["max_throughput"] = baseline

            metric = ActivationCheckpointBenchmarkMetrics(
                preset_name=profile.name,
                total_steps=profile.num_steps,
                total_tokens=total_tokens,
                tokens_per_second=total_tokens / max(total_time, 1e-9),
                avg_step_time_seconds=total_time / max(profile.num_steps, 1),
                peak_vram_mb=memory.training_peak_vram_mb,
                loss_stddev=statistics.pstdev(losses) if len(losses) > 1 else 0.0,
                loss_delta_vs_max_throughput=(
                    float(sum(losses) / max(len(losses), 1)) - float(baseline["loss_mean"])
                ),
                memory=memory,
            )
            metrics.append(metric)

            print(
                f"{metric.preset_name} | steps={metric.total_steps} | "
                f"tok/s={metric.tokens_per_second:.1f} | "
                f"step={metric.avg_step_time_seconds:.4f}s | "
                f"peak={metric.peak_vram_mb:.1f} MB | "
                f"loss_std={metric.loss_stddev:.6f}"
            )

        reset_activation_checkpointing(self.model)
        self.model.load_state_dict(base_state)
        return metrics

    def run_phase6_benchmarks(self) -> BenchmarkReport:
        """Run the requested Phase 6 activation-checkpoint benchmark suite."""
        return BenchmarkReport(
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            detailed_profiling=self.detailed_profiling,
            benchmark_suite="phase6",
            checkpoint_profiles=self.run_phase6_activation_checkpoint_benchmarks(),
        )

    def run_phase7_optimizer_benchmarks(self) -> list[OptimizerBenchmarkMetrics]:
        """Run the Phase 7 optimizer-state benchmark suite."""
        print(f"\n{'='*60}")
        print("Starting Phase 7 Optimizer Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Detailed Profiling: {self.detailed_profiling}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        set_detailed_profiling_enabled(self.detailed_profiling)
        self.setup_model_and_tokenizer()
        if hasattr(self.model, "model") and hasattr(self.model, "lm_head"):
            self.model = patch_model(self.model)
        dataloader = self.prepare_dataset()
        profiles = phase7_optimizer_profiles(num_steps=min(self.num_steps, 5) or 1)
        base_state = copy.deepcopy(self.model.state_dict())

        metrics: list[OptimizerBenchmarkMetrics] = []
        baselines: dict[str, float] = {}

        for profile in profiles:
            self.model.load_state_dict(base_state)
            self.model.train()
            optimizer = create_optimizer(
                self.model.parameters(),
                lr=1e-5,
                optimizer_name=profile.name,
            )
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            resident_model_bytes = model_resident_cuda_bytes(self.model)
            total_tokens = 0
            losses = []
            total_time = 0.0

            data_iter = iter(dataloader)
            for _ in range(profile.num_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch.get("labels")
                labels = input_ids if labels is None else labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                self._sync_device()
                start_time = time.time()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                if loss is None:
                    raise RuntimeError("phase7 benchmark expected a training loss")
                loss.backward()
                optimizer.step()
                self._sync_device()
                total_time += time.time() - start_time

                losses.append(float(loss.detach().float().item()))
                total_tokens += int(labels.ne(-100).sum().item())

            training_peak_bytes = capture_cuda_peak_bytes()
            record_training_peak_bytes(training_peak_bytes)
            optimizer_state_mb = optimizer_state_bytes(optimizer) / (1024**2)
            memory = build_memory_breakdown(
                resident_model_bytes=resident_model_bytes,
                kv_cache_bytes=0,
                temporary_decode_buffer_bytes=0,
                training_peak_bytes=training_peak_bytes,
                inference_peak_bytes=0,
                detailed_profiling=self.detailed_profiling,
            )
            adamw_loss = baselines.get("adamw")
            current_loss_mean = float(sum(losses) / max(len(losses), 1))
            if adamw_loss is None:
                baselines["adamw"] = current_loss_mean
                adamw_loss = current_loss_mean

            metric = OptimizerBenchmarkMetrics(
                optimizer_name=profile.name,
                total_steps=profile.num_steps,
                total_tokens=total_tokens,
                tokens_per_second=total_tokens / max(total_time, 1e-9),
                avg_step_time_seconds=total_time / max(profile.num_steps, 1),
                optimizer_state_mb=optimizer_state_mb,
                loss_delta_vs_adamw=current_loss_mean - adamw_loss,
                memory=memory,
            )
            metrics.append(metric)

            print(
                f"{metric.optimizer_name} | steps={metric.total_steps} | "
                f"tok/s={metric.tokens_per_second:.1f} | "
                f"step={metric.avg_step_time_seconds:.4f}s | "
                f"state={metric.optimizer_state_mb:.2f} MB | "
                f"loss_delta={metric.loss_delta_vs_adamw:.6f}"
            )

        self.model.load_state_dict(base_state)
        return metrics

    def run_phase7_benchmarks(self) -> BenchmarkReport:
        """Run the requested Phase 7 optimizer benchmark suite."""
        return BenchmarkReport(
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            detailed_profiling=self.detailed_profiling,
            benchmark_suite="phase7",
            optimizer_profiles=self.run_phase7_optimizer_benchmarks(),
        )

    def run_phase1_benchmarks(self, mode: str = "both") -> BenchmarkReport:
        """Run the requested Phase 1 benchmark modes and return a structured report."""
        include_training = mode in {"training", "both"}
        include_inference = mode in {"inference", "both"}
        return BenchmarkReport(
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            detailed_profiling=self.detailed_profiling,
            benchmark_suite="phase1",
            training=self.run_benchmark() if include_training else None,
            inference_profiles=self.run_inference_benchmarks() if include_inference else [],
        )

    def run_phase2_benchmarks(
        self,
        *,
        cache_layouts: Sequence[str] = ("contiguous", "paged"),
        serving_request_count: int = 8,
        fixed_vram_budget_mb: float = 2048.0,
    ) -> BenchmarkReport:
        """Run the requested Phase 2 KV benchmark suite and return a structured report."""
        return BenchmarkReport(
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            detailed_profiling=self.detailed_profiling,
            benchmark_suite="phase2",
            kv_cache_profiles=self.run_phase2_kv_benchmarks(
                cache_layouts=cache_layouts,
                serving_request_count=serving_request_count,
                fixed_vram_budget_mb=fixed_vram_budget_mb,
            ),
        )

    def run_phase3_quantized_kv_benchmarks(
        self,
        *,
        cache_layouts: Sequence[str] = ("contiguous", "paged", "paged_quantized"),
        quantized_residual_window_tokens: int = 128,
    ) -> list[QuantizedKVBenchmarkMetrics]:
        """Run the Phase 3 quantized KV benchmark and quality matrix."""
        print(f"\n{'='*60}")
        print("Starting Phase 3 Quantized KV Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Cache Layouts: {', '.join(cache_layouts)}")
        print(f"Batch Sizes: {', '.join(str(size) for size in self.inference_batch_sizes)}")
        print(f"Residual Window Tokens: {quantized_residual_window_tokens}")
        print(f"Detailed Profiling: {self.detailed_profiling}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        set_detailed_profiling_enabled(self.detailed_profiling)
        self.setup_model_and_tokenizer()
        self.model.eval()
        self._ensure_inference_patch()

        profiles = phase3_quantized_kv_profiles(
            self.inference_batch_sizes,
            short_prompt_length=self.short_prompt_length,
            long_prompt_length=self.long_prompt_length,
            quality_decode_length=self.short_decode_length,
            long_decode_length=self.long_decode_length,
        )

        ordered_layouts = list(dict.fromkeys(("contiguous", *cache_layouts)))
        baselines: dict[tuple[str, int, int, int], dict[str, object]] = {}
        metrics: list[QuantizedKVBenchmarkMetrics] = []

        with torch.inference_mode():
            for cache_layout in ordered_layouts:
                for profile in profiles:
                    with _temporary_env("BARQTRAIN_KV_CACHE_MODE", cache_layout):
                        with _temporary_env(
                            "BARQTRAIN_QUANTIZED_KV_RESIDUAL_TOKENS",
                            str(int(quantized_residual_window_tokens)),
                        ):
                            run = self._run_generate_call(
                                prompt_length=profile.prompt_length,
                                batch_size=profile.batch_size,
                                decode_length=profile.decode_length,
                            )

                    memory = run["memory"]
                    resident_vram_mb = memory.resident_model_mb + memory.kv_cache_mb
                    key = (
                        profile.name,
                        profile.batch_size,
                        profile.prompt_length,
                        profile.decode_length,
                    )
                    perplexity = self._sequence_perplexity(run["outputs"])

                    if cache_layout == "contiguous":
                        baselines[key] = {
                            **run,
                            "resident_vram_mb": resident_vram_mb,
                            "perplexity": perplexity,
                        }

                    baseline = baselines.get(key)
                    if baseline is None:
                        baseline = {
                            **run,
                            "resident_vram_mb": resident_vram_mb,
                            "perplexity": perplexity,
                        }

                    baseline_time = max(float(baseline["total_time"]), 1e-9)
                    baseline_resident_vram_mb = max(float(baseline["resident_vram_mb"]), 1e-9)
                    generation_match_ratio = self._generation_match_ratio(
                        baseline["outputs"],
                        run["outputs"],
                        profile.prompt_length,
                    )
                    metric = QuantizedKVBenchmarkMetrics(
                        scenario_name=profile.name,
                        cache_layout=str(run["cache_layout"] or cache_layout),
                        batch_size=profile.batch_size,
                        prompt_length=profile.prompt_length,
                        decode_length=profile.decode_length,
                        total_new_tokens=int(run["total_new_tokens"]),
                        total_time_seconds=float(run["total_time"]),
                        tokens_per_second=float(run["total_new_tokens"]) / max(float(run["total_time"]), 1e-9),
                        resident_vram_mb=resident_vram_mb,
                        peak_vram_mb=memory.inference_peak_vram_mb,
                        throughput_per_gb=(
                            (float(run["total_new_tokens"]) / max(float(run["total_time"]), 1e-9))
                            / max(resident_vram_mb / 1024.0, 1e-9)
                        ),
                        memory_savings_vs_contiguous_percent=(
                            (baseline_resident_vram_mb - resident_vram_mb) / baseline_resident_vram_mb * 100.0
                        ),
                        latency_vs_contiguous_percent=(
                            (float(run["total_time"]) - baseline_time) / baseline_time * 100.0
                        ),
                        generation_match_ratio=generation_match_ratio,
                        perplexity=perplexity,
                        reference_perplexity=baseline["perplexity"],
                        memory=memory,
                    )
                    metrics.append(metric)

                    perplexity_label = f"{metric.perplexity:.3f}" if metric.perplexity is not None else "n/a"
                    print(
                        f"{metric.scenario_name} | layout={metric.cache_layout} | "
                        f"bs={metric.batch_size} | tokens/s={metric.tokens_per_second:.1f} | "
                        f"resident={metric.resident_vram_mb:.1f} MB | peak={metric.peak_vram_mb:.1f} MB | "
                        f"mem_delta={metric.memory_savings_vs_contiguous_percent:.1f}% | "
                        f"latency_delta={metric.latency_vs_contiguous_percent:.1f}% | "
                        f"match={metric.generation_match_ratio:.3f} | ppl={perplexity_label}"
                    )

        return metrics

    def run_phase3_benchmarks(
        self,
        *,
        cache_layouts: Sequence[str] = ("contiguous", "paged", "paged_quantized"),
        quantized_residual_window_tokens: int = 128,
    ) -> BenchmarkReport:
        """Run the requested Phase 3 quantized KV benchmark suite and return a structured report."""
        return BenchmarkReport(
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            detailed_profiling=self.detailed_profiling,
            benchmark_suite="phase3",
            quantized_kv_profiles=self.run_phase3_quantized_kv_benchmarks(
                cache_layouts=cache_layouts,
                quantized_residual_window_tokens=quantized_residual_window_tokens,
            ),
        )

    def save_results(self, results: BenchmarkMetrics | BenchmarkReport) -> Path:
        """Save benchmark results to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(results, BenchmarkMetrics):
            filename = "baseline_results.json"
        else:
            if results.benchmark_suite == "phase2":
                filename = "phase2_results.json"
            elif results.benchmark_suite == "phase3":
                filename = "phase3_results.json"
            elif results.benchmark_suite == "phase4":
                filename = "phase4_results.json"
            elif results.benchmark_suite == "phase5":
                filename = "phase5_results.json"
            elif results.benchmark_suite == "phase6":
                filename = "phase6_results.json"
            elif results.benchmark_suite == "phase7":
                filename = "phase7_results.json"
            else:
                filename = "phase1_results.json"
        results_file = self.output_dir / filename
        with open(results_file, "w", encoding="utf-8") as handle:
            json.dump(asdict(results), handle, indent=2)

        print(f"\n{'='*60}")
        print("Benchmark Results Summary")
        print(f"{'='*60}")
        if isinstance(results, BenchmarkMetrics):
            print(f"Training Tokens/Second: {results.tokens_per_second:.1f}")
            print(f"Training Peak VRAM: {results.memory.training_peak_vram_mb:.1f} MB")
            print(f"Resident Model VRAM: {results.memory.resident_model_mb:.1f} MB")
        else:
            if results.training is not None:
                print(f"Training Tokens/Second: {results.training.tokens_per_second:.1f}")
                print(f"Training Peak VRAM: {results.training.memory.training_peak_vram_mb:.1f} MB")
            if results.inference_profiles:
                best_profile = max(results.inference_profiles, key=lambda metric: metric.tokens_per_second)
                print(
                    f"Fastest Inference Profile: {best_profile.profile_name} "
                    f"(bs={best_profile.batch_size}, {best_profile.tokens_per_second:.1f} tokens/s)"
                )
                print(
                    f"Inference Resident/KV/Temp: "
                    f"{best_profile.memory.resident_model_mb:.1f} / "
                    f"{best_profile.memory.kv_cache_mb:.1f} / "
                    f"{best_profile.memory.temporary_decode_buffers_mb:.1f} MB"
                )
            if results.kv_cache_profiles:
                best_profile = max(results.kv_cache_profiles, key=lambda metric: metric.tokens_per_second)
                print(
                    f"Fastest KV Scenario: {best_profile.scenario_name} "
                    f"({best_profile.cache_layout}, bs={best_profile.batch_size}, "
                    f"{best_profile.tokens_per_second:.1f} tokens/s)"
                )
                print(
                    f"KV OOM/Fragmentation/Resident/Peak: "
                    f"{best_profile.oom_rate:.2f} / "
                    f"{best_profile.fragmentation_ratio:.2f} / "
                    f"{best_profile.resident_vram_mb:.1f} / "
                    f"{best_profile.peak_vram_mb:.1f} MB"
                )
            if results.quantized_kv_profiles:
                best_profile = max(results.quantized_kv_profiles, key=lambda metric: metric.tokens_per_second)
                print(
                    f"Fastest Quantized KV Scenario: {best_profile.scenario_name} "
                    f"({best_profile.cache_layout}, bs={best_profile.batch_size}, "
                    f"{best_profile.tokens_per_second:.1f} tokens/s)"
                )
                print(
                    f"Quantized KV Delta/Match/Perplexity: "
                    f"{best_profile.memory_savings_vs_contiguous_percent:.1f}% / "
                    f"{best_profile.latency_vs_contiguous_percent:.1f}% / "
                    f"{best_profile.generation_match_ratio:.3f} / "
                    f"{best_profile.perplexity if best_profile.perplexity is not None else 'n/a'}"
                )
            if results.projection_profiles:
                fused_profiles = [
                    metric for metric in results.projection_profiles if metric.projection_mode == "fused"
                ] or results.projection_profiles
                best_profile = max(fused_profiles, key=lambda metric: metric.decode_tokens_per_second)
                print(
                    f"Best Fused Projection Scenario: {best_profile.scenario_name} "
                    f"(bs={best_profile.batch_size}, {best_profile.decode_tokens_per_second:.1f} tok/s)"
                )
                print(
                    f"Projection Train Step/Peak/Match: "
                    f"{best_profile.training_step_time_seconds:.4f}s / "
                    f"{best_profile.training_peak_vram_mb:.1f} MB / "
                    f"{best_profile.generation_match_ratio:.3f}"
                )
            if results.packed_training_profiles:
                packed_profiles = [
                    metric for metric in results.packed_training_profiles if metric.packing_mode == "packed"
                ] or results.packed_training_profiles
                best_profile = max(
                    packed_profiles,
                    key=lambda metric: metric.effective_tokens_per_second,
                )
                print(
                    f"Best Packed Training Scenario: {best_profile.scenario_name} "
                    f"(bs={best_profile.batch_size}, {best_profile.effective_tokens_per_second:.1f} tok/s)"
                )
                print(
                    f"Packed Peak/Loss Delta: "
                    f"{best_profile.peak_vram_mb:.1f} MB / "
                    f"{best_profile.loss_delta_vs_padded:.6f}"
                )
            if results.checkpoint_profiles:
                best_profile = min(results.checkpoint_profiles, key=lambda metric: metric.avg_step_time_seconds)
                print(
                    f"Fastest Checkpoint Preset: {best_profile.preset_name} "
                    f"({best_profile.tokens_per_second:.1f} tok/s)"
                )
                print(
                    f"Checkpoint Peak/Loss Std: "
                    f"{best_profile.peak_vram_mb:.1f} MB / "
                    f"{best_profile.loss_stddev:.6f}"
                )
            if results.optimizer_profiles:
                best_profile = max(results.optimizer_profiles, key=lambda metric: metric.tokens_per_second)
                print(
                    f"Fastest Optimizer Mode: {best_profile.optimizer_name} "
                    f"({best_profile.tokens_per_second:.1f} tok/s)"
                )
                print(
                    f"Optimizer State/Loss Delta: "
                    f"{best_profile.optimizer_state_mb:.2f} MB / "
                    f"{best_profile.loss_delta_vs_adamw:.6f}"
                )
        print(f"\nResults saved to: {results_file}")
        print(f"{'='*60}\n")
        return results_file


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return tuple(sizes)


def _parse_kv_cache_layouts(value: str) -> tuple[str, ...]:
    layouts = []
    for part in value.split(","):
        layout = part.strip().lower()
        if layout == "quantized":
            layout = "paged_quantized"
        if not layout:
            continue
        if layout not in {"contiguous", "paged", "paged_quantized"}:
            raise argparse.ArgumentTypeError(
                "unsupported KV cache layout "
                f"{layout!r}; expected 'contiguous', 'paged', or 'paged_quantized'"
            )
        if layout not in layouts:
            layouts.append(layout)
    if not layouts:
        raise argparse.ArgumentTypeError("at least one KV cache layout is required")
    return tuple(layouts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BarqTrain training and inference benchmarks")
    parser.add_argument(
        "--suite",
        type=str,
        default="phase1",
        choices=["phase1", "phase2", "phase3", "phase4", "phase5", "phase6", "phase7"],
        help="Which benchmark suite to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name or path from Hugging Face",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--sequence_length", type=int, default=512, help="Training sequence length")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--use-packing", action="store_true", help="Enable sequence packing collator")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "paged_adamw_32bit", "paged_adamw_8bit"],
        help="Optimizer to benchmark",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["training", "inference", "both"],
        help="Which benchmark modes to run",
    )
    parser.add_argument(
        "--inference-batch-sizes",
        type=_parse_batch_sizes,
        default=(1, 4, 8),
        help="Comma-separated inference batch sizes, for example 1,4,8",
    )
    parser.add_argument("--short-prompt-length", type=int, default=64)
    parser.add_argument("--long-prompt-length", type=int, default=1024)
    parser.add_argument("--short-decode-length", type=int, default=32)
    parser.add_argument("--long-decode-length", type=int, default=256)
    parser.add_argument(
        "--kv-cache-layouts",
        type=_parse_kv_cache_layouts,
        default=("contiguous", "paged"),
        help="Comma-separated KV cache layouts, for example contiguous,paged,paged_quantized",
    )
    parser.add_argument(
        "--kv-serving-requests",
        type=int,
        default=8,
        help="Number of simulated requests for the multi-request serving scenario",
    )
    parser.add_argument(
        "--kv-fixed-vram-budget-mb",
        type=float,
        default=2048.0,
        help="Resident/peak VRAM budget for the fixed-VRAM batch growth scenario",
    )
    parser.add_argument(
        "--quantized-kv-residual-window-tokens",
        type=int,
        default=128,
        help="Residual full-precision token window for the quantized KV-cache benchmark suite",
    )
    parser.add_argument("--detailed-profiling", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )

    args = parser.parse_args()
    harness = BenchmarkHarness(
        model_name=args.model,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_steps=args.steps,
        use_packing=args.use_packing,
        optimizer_name=args.optimizer,
        output_dir=args.output_dir,
        detailed_profiling=args.detailed_profiling,
        inference_batch_sizes=args.inference_batch_sizes,
        short_prompt_length=args.short_prompt_length,
        long_prompt_length=args.long_prompt_length,
        short_decode_length=args.short_decode_length,
        long_decode_length=args.long_decode_length,
    )

    if args.suite == "phase2":
        phase2_layouts = tuple(layout for layout in args.kv_cache_layouts if layout in {"contiguous", "paged"})
        results = harness.run_phase2_benchmarks(
            cache_layouts=phase2_layouts or ("contiguous", "paged"),
            serving_request_count=args.kv_serving_requests,
            fixed_vram_budget_mb=args.kv_fixed_vram_budget_mb,
        )
    elif args.suite == "phase3":
        phase3_layouts = args.kv_cache_layouts
        if "paged_quantized" not in phase3_layouts:
            phase3_layouts = (*phase3_layouts, "paged_quantized")
        results = harness.run_phase3_benchmarks(
            cache_layouts=phase3_layouts,
            quantized_residual_window_tokens=args.quantized_kv_residual_window_tokens,
        )
    elif args.suite == "phase4":
        results = harness.run_phase4_benchmarks()
    elif args.suite == "phase5":
        results = harness.run_phase5_benchmarks()
    elif args.suite == "phase6":
        results = harness.run_phase6_benchmarks()
    elif args.suite == "phase7":
        results = harness.run_phase7_benchmarks()
    elif args.mode == "training":
        results = harness.run_benchmark()
    else:
        results = harness.run_phase1_benchmarks(mode=args.mode)

    harness.save_results(results)


if __name__ == "__main__":
    main()
