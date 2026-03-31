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
            full_logits = self.lm_head(hidden[:, :-1, :])
            loss = torch.nn.functional.cross_entropy(
                full_logits.reshape(-1, full_logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return self.embed

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
    class FakePaddingFreeCollator:
        def __init__(
            self,
            max_length,
            pad_token_id,
            eos_token_id=None,
            label_pad_token_id=-100,
            drop_remainder=False,
            document_masked=False,
            document_id_key="document_id",
        ):
            self.max_length = max_length
            self.pad_token_id = pad_token_id
            self.eos_token_id = pad_token_id if eos_token_id is None else eos_token_id
            self.label_pad_token_id = label_pad_token_id
            self.drop_remainder = drop_remainder
            self.document_masked = document_masked
            self.document_id_key = document_id_key

        def __call__(self, examples):
            batches = []
            current_tokens = []
            current_position_ids = []
            current_sequence_ids = []
            current_document_ids = []
            current_starts = []

            def flush():
                if not current_tokens:
                    return
                active_tokens = len(current_tokens)
                input_ids = current_tokens + [self.pad_token_id] * (self.max_length - active_tokens)
                attention_mask = [1] * active_tokens + [0] * (self.max_length - active_tokens)
                labels = list(current_tokens) + [self.label_pad_token_id] * (self.max_length - active_tokens)
                if self.document_masked:
                    for boundary_start in current_starts[1:]:
                        labels[boundary_start] = self.label_pad_token_id
                loss_mask = [0 if label == self.label_pad_token_id else 1 for label in labels]
                position_ids = current_position_ids + [0] * (self.max_length - active_tokens)
                sequence_ids = current_sequence_ids + [-1] * (self.max_length - active_tokens)
                document_ids = current_document_ids + [-1] * (self.max_length - active_tokens)
                cu_seqlens = [0]
                block_offsets = []
                for offset in current_starts:
                    block_offsets.append(offset)
                for start, end in zip(current_starts, current_starts[1:] + [active_tokens]):
                    cu_seqlens.append(end)
                batches.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                        "position_ids": position_ids,
                        "sequence_ids": sequence_ids,
                        "document_ids": document_ids,
                        "loss_mask": loss_mask,
                        "cu_seqlens": cu_seqlens,
                        "block_offsets": block_offsets,
                        "max_sequence_length": max(
                            (end - start) for start, end in zip(current_starts, current_starts[1:] + [active_tokens])
                        ),
                        "active_tokens": active_tokens,
                    }
                )

            for example_index, example in enumerate(examples):
                tokens = [
                    token
                    for token, mask in zip(example["input_ids"], example.get("attention_mask", [1] * len(example["input_ids"])))
                    if mask
                ]
                if not tokens:
                    continue
                if tokens[-1] != self.eos_token_id:
                    tokens = list(tokens) + [self.eos_token_id]
                if current_tokens and len(current_tokens) + len(tokens) > self.max_length:
                    flush()
                    current_tokens = []
                    current_position_ids = []
                    current_sequence_ids = []
                    current_document_ids = []
                    current_starts = []
                if self.drop_remainder and len(tokens) > self.max_length:
                    tokens = tokens[: self.max_length]
                current_starts.append(len(current_tokens))
                document_id = int(example.get(self.document_id_key, example_index))
                for position, token in enumerate(tokens[: self.max_length - len(current_tokens)]):
                    current_tokens.append(int(token))
                    current_position_ids.append(position)
                    current_sequence_ids.append(example_index)
                    current_document_ids.append(document_id)
                if len(current_tokens) == self.max_length:
                    flush()
                    current_tokens = []
                    current_position_ids = []
                    current_sequence_ids = []
                    current_document_ids = []
                    current_starts = []

            flush()
            if not batches:
                batches = [
                    {
                        "input_ids": [self.pad_token_id] * self.max_length,
                        "attention_mask": [0] * self.max_length,
                        "labels": [self.label_pad_token_id] * self.max_length,
                        "position_ids": [0] * self.max_length,
                        "sequence_ids": [-1] * self.max_length,
                        "document_ids": [-1] * self.max_length,
                        "loss_mask": [0] * self.max_length,
                        "cu_seqlens": [0],
                        "block_offsets": [],
                        "max_sequence_length": 0,
                        "active_tokens": 0,
                    }
                ]

            max_cu = max(len(batch["cu_seqlens"]) for batch in batches)
            max_offsets = max(len(batch["block_offsets"]) for batch in batches)
            return {
                "input_ids": torch.tensor([batch["input_ids"] for batch in batches], dtype=torch.long),
                "attention_mask": torch.tensor([batch["attention_mask"] for batch in batches], dtype=torch.long),
                "labels": torch.tensor([batch["labels"] for batch in batches], dtype=torch.long),
                "position_ids": torch.tensor([batch["position_ids"] for batch in batches], dtype=torch.long),
                "sequence_ids": torch.tensor([batch["sequence_ids"] for batch in batches], dtype=torch.long),
                "document_ids": torch.tensor([batch["document_ids"] for batch in batches], dtype=torch.long),
                "loss_mask": torch.tensor([batch["loss_mask"] for batch in batches], dtype=torch.long),
                "cu_seqlens": torch.tensor(
                    [
                        batch["cu_seqlens"] + [batch["cu_seqlens"][-1]] * (max_cu - len(batch["cu_seqlens"]))
                        for batch in batches
                    ],
                    dtype=torch.long,
                ),
                "block_offsets": torch.tensor(
                    [
                        batch["block_offsets"] + [-1] * (max_offsets - len(batch["block_offsets"]))
                        for batch in batches
                    ],
                    dtype=torch.long,
                ),
                "max_sequence_length": torch.tensor(
                    [batch["max_sequence_length"] for batch in batches], dtype=torch.long
                ),
                "active_tokens": torch.tensor([batch["active_tokens"] for batch in batches], dtype=torch.long),
            }

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
    monkeypatch.setattr(
        "barqtrain.benchmarks.baseline.PaddingFreeCausalLMDataCollator",
        FakePaddingFreeCollator,
    )


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


def test_phase4_projection_report_serializes_training_and_decode_metrics(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=1,
        output_dir=str(tmp_path),
        inference_batch_sizes=(1, 4),
        short_prompt_length=4,
        long_prompt_length=6,
        short_decode_length=2,
        long_decode_length=4,
    )

    report = harness.run_phase4_benchmarks()

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase4"
    assert len(report.projection_profiles) == 8
    assert {profile.scenario_name for profile in report.projection_profiles} == {
        "vocab_heavy_long_decode",
        "vocab_heavy_long_context",
    }
    assert {profile.projection_mode for profile in report.projection_profiles} == {"baseline", "fused"}
    fused_profiles = [profile for profile in report.projection_profiles if profile.projection_mode == "fused"]
    assert all(profile.last_token_logits_only is True for profile in fused_profiles)
    assert all(profile.generation_match_ratio == 1.0 for profile in report.projection_profiles)
    assert all(abs(profile.loss_delta_vs_baseline) < 1e-6 for profile in fused_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase4_results.json"
    assert "projection_mode" in payload
    assert "training_step_time_seconds" in payload
    assert "decode_tokens_per_second" in payload
    assert "loss_delta_vs_baseline" in payload
    assert "last_token_logits_only" in payload


def test_phase5_packed_training_report_serializes_padding_free_metrics(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=8,
        num_steps=1,
        output_dir=str(tmp_path),
        inference_batch_sizes=(1, 4),
    )

    report = harness.run_phase5_benchmarks()

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase5"
    assert len(report.packed_training_profiles) == 8
    assert {profile.scenario_name for profile in report.packed_training_profiles} == {
        "matched_effective_tokens",
        "document_masked_training",
    }
    assert {profile.packing_mode for profile in report.packed_training_profiles} == {"padded", "packed"}
    assert all(profile.effective_tokens > 0 for profile in report.packed_training_profiles)
    assert all(profile.peak_vram_mb == profile.memory.training_peak_vram_mb for profile in report.packed_training_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase5_results.json"
    assert "packing_mode" in payload
    assert "effective_tokens_per_second" in payload
    assert "throughput_at_matched_effective_tokens" in payload
    assert "loss_delta_vs_padded" in payload


def test_phase6_checkpoint_report_serializes_preset_metrics(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=2,
        output_dir=str(tmp_path),
    )

    report = harness.run_phase6_benchmarks()

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase6"
    assert len(report.checkpoint_profiles) == 3
    assert {profile.preset_name for profile in report.checkpoint_profiles} == {
        "max_throughput",
        "balanced",
        "max_memory_saving",
    }
    assert all(profile.total_steps >= 1 for profile in report.checkpoint_profiles)
    assert all(profile.peak_vram_mb == profile.memory.training_peak_vram_mb for profile in report.checkpoint_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase6_results.json"
    assert "preset_name" in payload
    assert "tokens_per_second" in payload
    assert "avg_step_time_seconds" in payload
    assert "loss_stddev" in payload
    assert "loss_delta_vs_max_throughput" in payload


def test_phase7_optimizer_report_serializes_state_metrics(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=2,
        output_dir=str(tmp_path),
    )

    report = harness.run_phase7_benchmarks()

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase7"
    assert len(report.optimizer_profiles) == 4
    assert {profile.optimizer_name for profile in report.optimizer_profiles} == {
        "adamw",
        "barqtrain_adamw",
        "barqtrain_adamw_compact",
        "barqtrain_adamw_paged",
    }
    assert all(profile.optimizer_state_mb >= 0.0 for profile in report.optimizer_profiles)
    assert all(profile.memory.training_peak_vram_mb >= 0.0 for profile in report.optimizer_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase7_results.json"
    assert "optimizer_name" in payload
    assert "optimizer_state_mb" in payload
    assert "tokens_per_second" in payload
    assert "loss_delta_vs_adamw" in payload


def test_phase8_rmsnorm_fusion_report_serializes_latency_and_traffic(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=1,
        output_dir=str(tmp_path),
        inference_batch_sizes=(1, 4),
    )

    report = harness.run_phase8_benchmarks()

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase8"
    assert len(report.rmsnorm_fusion_profiles) == 12
    assert {profile.scenario_name for profile in report.rmsnorm_fusion_profiles} == {
        "residual_add_rmsnorm",
        "attention_input_projection",
        "mlp_input_projection",
    }
    assert {profile.fusion_mode for profile in report.rmsnorm_fusion_profiles} == {"separated", "fused"}
    fused_profiles = [
        profile for profile in report.rmsnorm_fusion_profiles if profile.fusion_mode == "fused"
    ]
    assert all(profile.memory_traffic_reduction_percent > 0.0 for profile in fused_profiles)
    assert all(profile.max_abs_error < 1e-5 for profile in report.rmsnorm_fusion_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase8_results.json"
    assert "fusion_mode" in payload
    assert "latency_seconds" in payload
    assert "approximate_memory_traffic_mb" in payload
    assert "memory_traffic_reduction_percent" in payload
    assert "max_abs_error" in payload


def test_phase9_attention_report_serializes_dispatch_metrics(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=4,
        num_steps=1,
        output_dir=str(tmp_path),
        inference_batch_sizes=(1, 4),
        short_prompt_length=4,
        long_prompt_length=8,
        short_decode_length=2,
        long_decode_length=4,
    )

    report = harness.run_phase9_benchmarks()

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase9"
    assert len(report.attention_profiles) == 12
    assert {profile.scenario_name for profile in report.attention_profiles} == {
        "prefill_throughput",
        "decode_throughput",
        "long_context_serving",
    }
    assert {profile.attention_backend for profile in report.attention_profiles} == {
        "sdpa",
        "flash_attention_2",
        "barqtrain_native_decode",
    }
    decode_profiles = [
        profile for profile in report.attention_profiles if profile.scenario_name != "prefill_throughput"
    ]
    assert all(profile.last_token_only is True for profile in decode_profiles)
    assert all(profile.max_abs_error < 1e-5 for profile in report.attention_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase9_results.json"
    assert "attention_backend" in payload
    assert "prefill_tokens_per_second" in payload
    assert "decode_tokens_per_second" in payload
    assert "memory_overhead_mb" in payload
    assert "max_abs_error" in payload


def test_phase10_lora_report_serializes_adapter_metrics(monkeypatch, tmp_path):
    _install_fake_runtime(monkeypatch)

    harness = BenchmarkHarness(
        model_name="fake",
        batch_size=1,
        sequence_length=8,
        num_steps=1,
        output_dir=str(tmp_path),
        inference_batch_sizes=(1, 4),
    )

    report = harness.run_phase10_benchmarks()

    assert isinstance(report, BenchmarkReport)
    assert report.benchmark_suite == "phase10"
    assert len(report.lora_profiles) == 8
    assert {profile.scenario_name for profile in report.lora_profiles} == {
        "dense_chunked_loss",
        "packed_chunked_loss",
    }
    assert {profile.adapter_mode for profile in report.lora_profiles} == {
        "reference",
        "barqtrain_fused",
    }
    assert all(profile.effective_tokens > 0 for profile in report.lora_profiles)
    assert all(profile.effective_tokens_per_second > 0.0 for profile in report.lora_profiles)
    assert all(profile.peak_vram_mb >= 0.0 for profile in report.lora_profiles)

    results_file = harness.save_results(report)
    payload = results_file.read_text(encoding="utf-8")

    assert results_file.name == "phase10_results.json"
    assert "adapter_mode" in payload
    assert "effective_tokens_per_second" in payload
    assert "peak_vram_mb" in payload
    assert "loss_delta_vs_reference" in payload
