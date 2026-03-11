"""
Data pipeline tests for BarqTrain.
"""

from types import SimpleNamespace

import pytest
import torch

import barqtrain.data as data


class FakeRustBackend:
    def __init__(self):
        self.pack_sequences_calls = []
        self.parallel_tokenize_calls = []
        self.pack_for_causal_lm_calls = []

    def pack_sequences(self, sequences, max_len):
        self.pack_sequences_calls.append((sequences, max_len))
        return [
            SimpleNamespace(
                input_ids=[1, 2, 3],
                attention_mask=[1, 1, 1],
                sequence_ids=[0, 0, 1],
                position_ids=[0, 1, 2],
            )
        ]

    def parallel_tokenize(self, texts, tokenizer_path):
        self.parallel_tokenize_calls.append((texts, tokenizer_path))
        return [[11, 12], [21]]

    def pack_for_causal_lm(
        self,
        sequences,
        max_length,
        pad_token_id,
        eos_token_id,
        label_pad_token_id,
        drop_remainder,
    ):
        self.pack_for_causal_lm_calls.append(
            (
                sequences,
                max_length,
                pad_token_id,
                eos_token_id,
                label_pad_token_id,
                drop_remainder,
            )
        )
        return [
            SimpleNamespace(
                input_ids=[1, 2, 3, 99, 4],
                attention_mask=[1, 1, 1, 1, 1],
                labels=[1, 2, 3, 99, 4],
            ),
            SimpleNamespace(
                input_ids=[5, 99, 0, 0, 0],
                attention_mask=[1, 1, 0, 0, 0],
                labels=[5, 99, -100, -100, -100],
            ),
        ]


def test_pack_sequences_requires_rust_backend(monkeypatch):
    monkeypatch.setattr(data, "_get_rust_backend", lambda: None)

    with pytest.raises(RuntimeError, match="Rust backend is unavailable"):
        data.pack_sequences([[1, 2, 3]], max_len=8)


def test_pack_sequences_uses_rust_backend(monkeypatch):
    backend = FakeRustBackend()
    monkeypatch.setattr(data, "_get_rust_backend", lambda: backend)

    batches = data.pack_sequences([[1, 2, 3]], max_len=8)

    assert backend.pack_sequences_calls == [([[1, 2, 3]], 8)]
    assert len(batches) == 1
    assert batches[0].input_ids == [1, 2, 3]


def test_parallel_tokenize_requires_rust_backend(monkeypatch):
    monkeypatch.setattr(data, "_get_rust_backend", lambda: None)

    with pytest.raises(RuntimeError, match="Rust backend is unavailable"):
        data.parallel_tokenize(["abc"], tokenizer_path="tokenizer.json")


def test_parallel_tokenize_uses_rust_backend(monkeypatch):
    backend = FakeRustBackend()
    monkeypatch.setattr(data, "_get_rust_backend", lambda: backend)

    tokenized = data.parallel_tokenize(["ab", "c"], tokenizer_path="tokenizer.json")

    assert backend.parallel_tokenize_calls == [(["ab", "c"], "tokenizer.json")]
    assert tokenized == [[11, 12], [21]]


def test_pack_for_causal_lm_requires_rust_backend(monkeypatch):
    monkeypatch.setattr(data, "_get_rust_backend", lambda: None)

    with pytest.raises(RuntimeError, match="Rust backend is unavailable"):
        data.pack_for_causal_lm(
            sequences=[[1, 2, 3]],
            max_length=4,
            pad_token_id=0,
            eos_token_id=2,
        )


def test_pack_for_causal_lm_requires_native_method(monkeypatch):
    monkeypatch.setattr(data, "_get_rust_backend", lambda: object())

    with pytest.raises(RuntimeError, match="missing `pack_for_causal_lm`"):
        data.pack_for_causal_lm(
            sequences=[[1, 2, 3]],
            max_length=4,
            pad_token_id=0,
            eos_token_id=2,
        )


def test_pack_for_causal_lm_uses_rust_backend(monkeypatch):
    backend = FakeRustBackend()
    monkeypatch.setattr(data, "_get_rust_backend", lambda: backend)

    packed = data.pack_for_causal_lm(
        sequences=[[1, 2, 3], [4, 5]],
        max_length=5,
        pad_token_id=0,
        eos_token_id=99,
    )

    assert backend.pack_for_causal_lm_calls == [
        (
            [[1, 2, 3], [4, 5]],
            5,
            0,
            99,
            -100,
            False,
        )
    ]
    assert len(packed) == 2
    assert packed[0]["input_ids"] == [1, 2, 3, 99, 4]
    assert packed[1]["labels"] == [5, 99, -100, -100, -100]


def test_packed_causal_lm_collator_uses_trimmed_sequences(monkeypatch):
    backend = FakeRustBackend()
    monkeypatch.setattr(data, "_get_rust_backend", lambda: backend)

    collator = data.PackedCausalLMDataCollator(
        max_length=5,
        pad_token_id=0,
        eos_token_id=2,
    )
    batch = collator(
        [
            {"input_ids": [11, 12, 0, 0], "attention_mask": [1, 1, 0, 0]},
            {"input_ids": [21, 0, 0, 0], "attention_mask": [1, 0, 0, 0]},
        ]
    )

    assert isinstance(batch["input_ids"], torch.Tensor)
    assert backend.pack_for_causal_lm_calls == [
        (
            [[11, 12], [21]],
            5,
            0,
            2,
            -100,
            False,
        )
    ]
    assert batch["input_ids"].tolist() == [[1, 2, 3, 99, 4], [5, 99, 0, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]
    assert batch["labels"].tolist() == [[1, 2, 3, 99, 4], [5, 99, -100, -100, -100]]
