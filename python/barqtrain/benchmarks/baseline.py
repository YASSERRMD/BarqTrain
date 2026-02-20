"""
Baseline Benchmark Script for BarqTrain

This script establishes performance baselines using standard Hugging Face
and PyTorch implementations. It tracks:
- Tokens per second
- Step time
- Peak VRAM usage
- Training throughput

Usage:
    python -m barqtrain.benchmarks.baseline --model tinyllama --batch_size 4 --steps 100
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics"""

    model_name: str
    batch_size: int
    sequence_length: int
    total_steps: int
    total_tokens: int
    total_time_seconds: float
    tokens_per_second: float
    avg_step_time_seconds: float
    peak_vram_mb: float
    gpu_utilization_percent: Optional[float] = None


class BenchmarkHarness:
    """Main benchmark harness for measuring training performance"""

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        batch_size: int = 4,
        sequence_length: int = 512,
        num_steps: int = 100,
        output_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_steps = num_steps
        self.output_dir = Path(output_dir) if output_dir else Path("benchmarks/results")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)

    def prepare_dataset(self) -> DataLoader:
        """Prepare a small dataset for benchmarking"""
        print("Preparing dataset...")

        # Use a tiny subset of a dataset for quick benchmarking
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.sequence_length,
                padding="max_length",
                return_overflowing_tokens=False,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        # Create DataLoader
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # For baseline, we use Python workers
        )

        return dataloader

    def measure_vram(self) -> float:
        """Measure current VRAM usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        return 0.0

    def run_benchmark(self) -> BenchmarkMetrics:
        """Run the main benchmark loop"""
        print(f"\n{'='*60}")
        print(f"Starting Baseline Benchmark")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Sequence Length: {self.sequence_length}")
        print(f"Steps: {self.num_steps}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")

        # Setup
        self.setup_model_and_tokenizer()
        dataloader = self.prepare_dataset()

        # Training mode
        self.model.train()

        # Reset memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Benchmark loop
        start_time = time.time()
        total_tokens = 0

        data_iter = iter(dataloader)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        for step in range(self.num_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Track metrics
            total_tokens += input_ids.numel()

            if (step + 1) % 10 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                tokens_sec = total_tokens / elapsed
                vram_mb = self.measure_vram()
                print(
                    f"Step {step+1}/{self.num_steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Tokens/s: {tokens_sec:.1f} | "
                    f"VRAM: {vram_mb:.1f} MB"
                )

        # Final metrics
        total_time = time.time() - start_time
        peak_vram = self.measure_vram()

        metrics = BenchmarkMetrics(
            model_name=self.model_name,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
            total_steps=self.num_steps,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            tokens_per_second=total_tokens / total_time,
            avg_step_time_seconds=total_time / self.num_steps,
            peak_vram_mb=peak_vram,
        )

        return metrics

    def save_results(self, metrics: BenchmarkMetrics):
        """Save benchmark results to JSON"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results_file = self.output_dir / "baseline_results.json"
        with open(results_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        print(f"\n{'='*60}")
        print("Benchmark Results Summary")
        print(f"{'='*60}")
        print(f"Total Time: {metrics.total_time_seconds:.2f} seconds")
        print(f"Tokens/Second: {metrics.tokens_per_second:.1f}")
        print(f"Avg Step Time: {metrics.avg_step_time_seconds*1000:.2f} ms")
        print(f"Peak VRAM: {metrics.peak_vram_mb:.1f} MB")
        print(f"Total Tokens: {metrics.total_tokens:,}")
        print(f"\nResults saved to: {results_file}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Run baseline benchmarks for BarqTrain")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name or path from Hugging Face",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of training steps"
    )
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
        output_dir=args.output_dir,
    )

    metrics = harness.run_benchmark()
    harness.save_results(metrics)


if __name__ == "__main__":
    main()
