"""Tests for BarqTrain memory helpers."""

import types

import torch

from barqtrain.memory import (
    CudaMemorySnapshot,
    build_generation_kwargs,
    cuda_memory_snapshot,
    generation_overhead_mb,
    preferred_last_token_logits_kwarg,
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
