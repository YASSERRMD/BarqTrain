"""Tests for the benchmark harness."""

import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from barqtrain.benchmarks.baseline import BenchmarkHarness, BenchmarkReport


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 0

    def __call__(
        self,
        text,
        truncation=False,
        max_length=None,
        padding=False,
        return_overflowing_tokens=False,
        return_tensors=None,
        add_special_tokens=False,
    ):
        del truncation, return_overflowing_tokens, return_tensors, add_special_tokens
        if isinstance(text, list):
            sequences = [self._encode(sample, max_length=max_length) for sample in text]
        else:
            sequences = [self._encode(text, max_length=max_length)]

        if padding == "max_length" and max_length is not None:
            sequences = [sequence + [0] * (max_length - len(sequence)) for sequence in sequences]

        max_seq_len = max(len(sequence) for sequence in sequences)
        padded = [sequence + [0] * (max_seq_len - len(sequence)) for sequence in sequences]
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = input_ids.ne(0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @staticmethod
    def _encode(text, max_length=None):
        tokens = [((idx % 31) + 1) for idx, _ in enumerate(str(text).split())] or [1]
        if max_length is not None:
            tokens = tokens[:max_length]
        return tokens


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
        )
        self.embed = torch.nn.Embedding(32, 8)
        self.lm_head = torch.nn.Linear(8, 32, bias=False)
        self.generation_config = SimpleNamespace(
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=None,
            max_length=128,
            output_logits=False,
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, logits_to_keep=None, **kwargs):
        del attention_mask, kwargs
        hidden = self.embed(input_ids)
        logits = self.lm_head(hidden)
        if logits_to_keep == 1:
            logits = logits[:, -1:, :]
        loss = None
        if labels is not None:
            full_logits = self.lm_head(hidden)
            loss = torch.nn.functional.cross_entropy(
                full_logits.view(-1, full_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def generate(self, input_ids=None, max_new_tokens=1, logits_to_keep=None, **kwargs):
        del kwargs
        cache_layout = os.environ.get("BARQTRAIN_KV_CACHE_MODE", "contiguous")
        self._barqtrain_last_generate_used_paged_kv = cache_layout == "paged"
        self._barqtrain_last_generate_used_contiguous_kv = cache_layout == "contiguous"
        self._barqtrain_last_generate_used_quantized_kv = cache_layout == "paged_quantized"
        self._barqtrain_last_generate_kv_cache_layout = cache_layout
        self._barqtrain_last_generate_last_token_logits_only = logits_to_keep == 1
        resident_model_bytes = 64 * 1024 * 1024
        kv_cache_bytes = {
            "paged": 16,
            "paged_quantized": 10,
            "contiguous": 24,
        }.get(cache_layout, 24) * 1024 * 1024
        decode_temp_bytes = {
            "paged": 8,
            "paged_quantized": 10,
            "contiguous": 12,
        }.get(cache_layout, 12) * 1024 * 1024
        self._barqtrain_last_generate_resident_model_bytes = resident_model_bytes
        self._barqtrain_last_generate_kv_cache_bytes = kv_cache_bytes
        self._barqtrain_last_generate_decode_temp_bytes = decode_temp_bytes
        self._barqtrain_last_generate_inference_peak_bytes = (
            resident_model_bytes + kv_cache_bytes + decode_temp_bytes
        )
        self._barqtrain_last_generate_cache = SimpleNamespace(
            barqtrain_cache_layout=cache_layout,
            fragmentation_ratio=lambda: 0.15 if cache_layout == "paged_quantized" else (0.25 if cache_layout == "paged" else 0.0),
        )
        append = torch.full(
            (input_ids.size(0), max_new_tokens),
            7,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat([input_ids, append], dim=1)


def _install_fake_runtime(monkeypatch):
    def fake_setup(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel().to(self.device)

    def fake_prepare_dataset(self):
        sample = {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4], dtype=torch.long),
        }
        return DataLoader([sample, sample], batch_size=self.batch_size)

    monkeypatch.setattr(BenchmarkHarness, "setup_model_and_tokenizer", fake_setup)
    monkeypatch.setattr(BenchmarkHarness, "prepare_dataset", fake_prepare_dataset)
    monkeypatch.setattr("barqtrain.benchmarks.baseline.patch_inference", lambda model: model)


def test_training_benchmark_reports_bucketed_memory(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=2,
        output_dir=str(tmp_path),
    )

    metrics = harness.run_benchmark()

    assert metrics.memory.training_peak_vram_mb >= 0.0
    assert metrics.memory.resident_model_mb == 0.0
    assert metrics.memory.kv_cache_mb == 0.0
    assert metrics.peak_vram_mb == metrics.memory.training_peak_vram_mb


def test_phase1_benchmark_report_serializes_separate_memory_buckets(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=1,
        output_dir=str(tmp_path),
        inference_batch_sizes=(1, 4, 8),
    )

    report = harness.run_phase1_benchmarks(mode="both")
    assert isinstance(report, BenchmarkReport)
    assert report.training is not None
    assert len(report.inference_profiles) == 6
    assert {profile.profile_name for profile in report.inference_profiles} == {
        "short_prompt_long_decode",
        "long_prompt_short_decode",
    }
    assert {profile.batch_size for profile in report.inference_profiles} == {1, 4, 8}
    assert all(profile.last_token_logits_only is True for profile in report.inference_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert "resident_model_mb" in payload
    assert "kv_cache_mb" in payload
    assert "temporary_decode_buffers_mb" in payload
    assert "training_peak_vram_mb" in payload
    assert "inference_peak_vram_mb" in payload


def test_phase2_kv_benchmark_report_serializes_layout_comparison(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=1,
        output_dir=str(tmp_path),
        inference_batch_sizes=(1, 4),
    )

    report = harness.run_phase2_benchmarks(
        cache_layouts=("contiguous", "paged"),
        serving_request_count=3,
        fixed_vram_budget_mb=256.0,
    )

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase2"
    assert len(report.kv_cache_profiles) == 12
    assert {profile.scenario_name for profile in report.kv_cache_profiles} == {
        "long_prompt_generation",
        "multi_request_serving",
        "fixed_vram_batch_growth",
    }
    assert {profile.cache_layout for profile in report.kv_cache_profiles} == {"contiguous", "paged"}
    assert all(profile.oom_rate == 0.0 for profile in report.kv_cache_profiles)
    assert all(profile.peak_vram_mb >= profile.resident_vram_mb for profile in report.kv_cache_profiles)
    assert any(profile.fragmentation_ratio > 0.0 for profile in report.kv_cache_profiles if profile.cache_layout == "paged")

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase2_results.json"
    assert "scenario_name" in payload
    assert "cache_layout" in payload
    assert "oom_rate" in payload
    assert "fragmentation_ratio" in payload
    assert "resident_vram_mb" in payload
    assert "peak_vram_mb" in payload


def test_phase3_quantized_kv_report_serializes_quality_metrics(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=1,
        output_dir=str(tmp_path),
        inference_batch_sizes=(1, 4),
    )

    report = harness.run_phase3_benchmarks(
        cache_layouts=("contiguous", "paged", "paged_quantized"),
        quantized_residual_window_tokens=32,
    )

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase3"
    assert len(report.quantized_kv_profiles) == 18
    assert {profile.scenario_name for profile in report.quantized_kv_profiles} == {
        "memory_savings_vs_latency",
        "long_context_generation_quality",
        "throughput_per_gb",
    }
    assert {profile.cache_layout for profile in report.quantized_kv_profiles} == {
        "contiguous",
        "paged",
        "paged_quantized",
    }
    assert all(profile.generation_match_ratio == 1.0 for profile in report.quantized_kv_profiles)
    quantized_profiles = [
        profile for profile in report.quantized_kv_profiles if profile.cache_layout == "paged_quantized"
    ]
    assert all(profile.memory_savings_vs_contiguous_percent > 0.0 for profile in quantized_profiles)
    assert all(profile.perplexity is not None for profile in report.quantized_kv_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase3_results.json"
    assert "throughput_per_gb" in payload
    assert "memory_savings_vs_contiguous_percent" in payload
    assert "latency_vs_contiguous_percent" in payload
    assert "generation_match_ratio" in payload
    assert "reference_perplexity" in payload
