"""
Benchmark harness for BarqTrain training and Phase 1 decode profiles.

Usage:
    python -m barqtrain.benchmarks.baseline --model tinyllama --mode both --steps 100
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from barqtrain.data import PackedCausalLMDataCollator
from barqtrain.memory import (
    BenchmarkMemoryBreakdown,
    build_generation_kwargs,
    build_memory_breakdown,
    capture_cuda_peak_bytes,
    model_resident_cuda_bytes,
    paged_kv_cache_bytes,
    phase1_inference_profiles,
    record_training_peak_bytes,
    set_detailed_profiling_enabled,
)
from barqtrain.optim import create_optimizer
from barqtrain.patch_models import patch_inference


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
class BenchmarkReport:
    """Combined training + inference benchmark report."""

    model_name: str
    optimizer_name: str
    detailed_profiling: bool
    training: Optional[BenchmarkMetrics] = None
    inference_profiles: list[InferenceBenchmarkMetrics] = field(default_factory=list)


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
                inputs = self._build_prompt_inputs(profile.prompt_length, profile.batch_size)
                generation_kwargs = build_generation_kwargs(self.model, profile.decode_length)

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
                metric = InferenceBenchmarkMetrics(
                    profile_name=profile.name,
                    batch_size=profile.batch_size,
                    prompt_length=profile.prompt_length,
                    decode_length=profile.decode_length,
                    total_new_tokens=total_new_tokens,
                    total_time_seconds=total_time,
                    tokens_per_second=total_new_tokens / max(total_time, 1e-9),
                    paged_kv_cache=bool(getattr(self.model, "_barqtrain_last_generate_used_paged_kv", False)),
                    last_token_logits_only=bool(
                        getattr(self.model, "_barqtrain_last_generate_last_token_logits_only", False)
                    ),
                    memory=memory,
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

    def run_phase1_benchmarks(self, mode: str = "both") -> BenchmarkReport:
        """Run the requested Phase 1 benchmark modes and return a structured report."""
        include_training = mode in {"training", "both"}
        include_inference = mode in {"inference", "both"}
        return BenchmarkReport(
            model_name=self.model_name,
            optimizer_name=self.optimizer_name,
            detailed_profiling=self.detailed_profiling,
            training=self.run_benchmark() if include_training else None,
            inference_profiles=self.run_inference_benchmarks() if include_inference else [],
        )

    def save_results(self, results: BenchmarkMetrics | BenchmarkReport) -> Path:
        """Save benchmark results to JSON."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = "baseline_results.json" if isinstance(results, BenchmarkMetrics) else "phase1_results.json"
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
        print(f"\nResults saved to: {results_file}")
        print(f"{'='*60}\n")
        return results_file


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one batch size is required")
    return tuple(sizes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BarqTrain training and inference benchmarks")
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

    if args.mode == "training":
        results = harness.run_benchmark()
    else:
        results = harness.run_phase1_benchmarks(mode=args.mode)

    harness.save_results(results)


if __name__ == "__main__":
    main()
