"""Tests for BarqTrain memory helpers."""

import types
from types import SimpleNamespace

import torch

from barqtrain.memory import (
    BenchmarkMemoryBreakdown,
    CudaMemorySnapshot,
    build_memory_breakdown,
    build_generation_kwargs,
    cuda_memory_snapshot,
    detailed_profiling_enabled,
    generation_overhead_mb,
    maybe_prepare_last_token_logits_generate_kwargs,
    native_memory_snapshot,
    paged_kv_cache_bytes,
    phase1_inference_profiles,
    preferred_last_token_logits_kwarg,
    set_detailed_profiling_enabled,
    track_decode_temp_memory,
)


def test_cuda_memory_snapshot_is_zero_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    snapshot = cuda_memory_snapshot()

    assert snapshot == CudaMemorySnapshot()


def test_generation_overhead_mb_is_non_negative():
    resident = CudaMemorySnapshot(allocated_mb=256.0)
    peak = CudaMemorySnapshot(max_allocated_mb=384.0)

    assert generation_overhead_mb(resident, peak) == 128.0
    assert generation_overhead_mb(peak, resident) == 0.0


def test_preferred_last_token_logits_kwarg_detects_logits_to_keep():
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids=None, logits_to_keep=None):
            return input_ids, logits_to_keep

    assert preferred_last_token_logits_kwarg(DummyModel()) == "logits_to_keep"


def test_preferred_last_token_logits_kwarg_detects_num_logits_to_keep():
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids=None, num_logits_to_keep=None):
            return input_ids, num_logits_to_keep

    assert preferred_last_token_logits_kwarg(DummyModel()) == "num_logits_to_keep"


def test_preferred_last_token_logits_kwarg_returns_none_when_unsupported():
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids=None):
            return input_ids

    assert preferred_last_token_logits_kwarg(DummyModel()) is None


def test_preferred_last_token_logits_kwarg_uses_generation_mixin_capability():
    class DummyModel(torch.nn.Module):
        def _supports_logits_to_keep(self):
            return True

        def forward(self, input_ids=None):
            return input_ids

    assert preferred_last_token_logits_kwarg(DummyModel()) == "logits_to_keep"


def test_build_generation_kwargs_uses_generation_config_copy():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.generation_config = types.SimpleNamespace(
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
            )

        def forward(self, input_ids=None, logits_to_keep=None):
            return input_ids, logits_to_keep

    model = DummyModel()

    kwargs = build_generation_kwargs(model, 32)

    assert kwargs["max_new_tokens"] == 32
    assert kwargs["logits_to_keep"] == 1
    assert kwargs["generation_config"].do_sample is False
    assert kwargs["generation_config"].temperature is None
    assert kwargs["generation_config"].top_p is None
    assert kwargs["generation_config"].top_k is None
    assert model.generation_config.do_sample is True
    assert model.generation_config.temperature == 0.7
    assert model.generation_config.top_p == 0.9
    assert model.generation_config.top_k == 50


def test_build_generation_kwargs_without_generation_config_falls_back_to_generate_kwargs():
    class DummyModel(torch.nn.Module):
        def forward(self, input_ids=None):
            return input_ids

    kwargs = build_generation_kwargs(DummyModel(), 16)

    assert kwargs == {
        "max_new_tokens": 16,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }


def test_build_memory_breakdown_falls_back_without_rust_backend(monkeypatch):
    monkeypatch.setattr("barqtrain.memory._get_rust_backend", lambda: None)

    report = build_memory_breakdown(
        resident_model_bytes=256 * 1024**2,
        kv_cache_bytes=64 * 1024**2,
        temporary_decode_buffer_bytes=32 * 1024**2,
        training_peak_bytes=640 * 1024**2,
        inference_peak_bytes=352 * 1024**2,
        detailed_profiling=True,
    )

    assert report == BenchmarkMemoryBreakdown(
        resident_model_mb=256.0,
        kv_cache_mb=64.0,
        temporary_decode_buffers_mb=32.0,
        training_peak_vram_mb=640.0,
        inference_peak_vram_mb=352.0,
        detailed_profiling=True,
    )


def test_build_memory_breakdown_uses_rust_backend_when_available(monkeypatch):
    fake_backend = SimpleNamespace(
        build_memory_breakdown=lambda *args: SimpleNamespace(
            resident_model_mb=12.5,
            kv_cache_mb=6.0,
            temporary_decode_buffers_mb=1.5,
            training_peak_vram_mb=18.0,
            inference_peak_vram_mb=14.0,
            detailed_profiling=True,
        )
    )
    monkeypatch.setattr("barqtrain.memory._get_rust_backend", lambda: fake_backend)

    report = build_memory_breakdown(
        resident_model_bytes=0,
        kv_cache_bytes=0,
        temporary_decode_buffer_bytes=0,
        training_peak_bytes=0,
        inference_peak_bytes=0,
    )

    assert report.resident_model_mb == 12.5
    assert report.kv_cache_mb == 6.0
    assert report.temporary_decode_buffers_mb == 1.5
    assert report.training_peak_vram_mb == 18.0
    assert report.inference_peak_vram_mb == 14.0
    assert report.detailed_profiling is True


def test_phase1_inference_profiles_cover_required_matrix(monkeypatch):
    monkeypatch.setattr("barqtrain.memory._get_rust_backend", lambda: None)

    profiles = phase1_inference_profiles()

    assert [profile.name for profile in profiles] == [
        "short_prompt_long_decode",
        "long_prompt_short_decode",
        "short_prompt_long_decode",
        "long_prompt_short_decode",
        "short_prompt_long_decode",
        "long_prompt_short_decode",
    ]
    assert [profile.batch_size for profile in profiles] == [1, 1, 4, 4, 8, 8]


def test_paged_kv_cache_bytes_sums_unique_layer_allocations(monkeypatch):
    sentinel_ptrs = {}

    def fake_storage_bytes(tensor):
        return sentinel_ptrs[id(tensor)]

    monkeypatch.setattr("barqtrain.memory._storage_bytes_for_cuda_tensor", fake_storage_bytes)

    shared = object()
    keys = object()
    values = object()
    seq_lens = object()
    sentinel_ptrs[id(shared)] = (1, 128)
    sentinel_ptrs[id(keys)] = (2, 256)
    sentinel_ptrs[id(values)] = (3, 512)
    sentinel_ptrs[id(seq_lens)] = (4, 64)

    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(keys=keys, values=values, seq_lens=seq_lens),
            SimpleNamespace(keys=shared, values=shared, seq_lens=None),
        ]
    )

    assert paged_kv_cache_bytes(cache) == 960


def test_track_decode_temp_memory_records_remainder(monkeypatch):
    recorded = {}

    def fake_record(bucket, current_bytes):
        recorded[bucket] = current_bytes

    monkeypatch.setattr("barqtrain.memory._record_native_bucket_bytes", fake_record)

    decode_temp = track_decode_temp_memory(
        resident_model_bytes=512,
        kv_cache_bytes=128,
        inference_peak_bytes=800,
    )

    assert decode_temp == 160
    assert recorded["decode_temp"] == 160


def test_set_detailed_profiling_enabled_updates_env_and_backend(monkeypatch):
    calls = []
    fake_backend = SimpleNamespace(barqtrain_memory_set_enabled=lambda enabled: calls.append(enabled))
    monkeypatch.setattr("barqtrain.memory._get_cuda_backend", lambda: fake_backend)

    set_detailed_profiling_enabled(True)

    assert detailed_profiling_enabled() is True
    assert calls == [True]


def test_native_memory_snapshot_falls_back_without_backend(monkeypatch):
    monkeypatch.setattr("barqtrain.memory._get_cuda_backend", lambda: None)

    snapshot = native_memory_snapshot()

    assert snapshot["enabled"] is False
    assert snapshot["resident_model_current_bytes"] == 0


def test_maybe_prepare_last_token_logits_generate_kwargs_sets_supported_kwarg(monkeypatch):
    monkeypatch.setenv("BARQTRAIN_LAST_TOKEN_LOGITS_ONLY", "1")

    class DummyModel(torch.nn.Module):
        def forward(self, input_ids=None, logits_to_keep=None):
            return input_ids, logits_to_keep

    updated_kwargs, enabled = maybe_prepare_last_token_logits_generate_kwargs(
        DummyModel(),
        (),
        {"max_new_tokens": 8},
    )

    assert enabled is True
    assert updated_kwargs["logits_to_keep"] == 1


def test_maybe_prepare_last_token_logits_generate_kwargs_respects_output_logits(monkeypatch):
    monkeypatch.setenv("BARQTRAIN_LAST_TOKEN_LOGITS_ONLY", "1")

    class DummyModel(torch.nn.Module):
        def forward(self, input_ids=None, logits_to_keep=None):
            return input_ids, logits_to_keep

    updated_kwargs, enabled = maybe_prepare_last_token_logits_generate_kwargs(
        DummyModel(),
        (),
        {"max_new_tokens": 8, "output_logits": True},
    )

    assert enabled is False
    assert "logits_to_keep" not in updated_kwargs
