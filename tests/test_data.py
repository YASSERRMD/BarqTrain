"""
Data pipeline tests for BarqTrain.
"""

import pytest
import torch

from barqtrain.data import PackedCausalLMDataCollator, pack_for_causal_lm


def test_pack_for_causal_lm_appends_eos_and_pads():
    packed = pack_for_causal_lm(
        sequences=[[1, 2, 3], [4, 5]],
        max_length=5,
        pad_token_id=0,
        eos_token_id=99,
    )

    assert len(packed) == 2
    assert packed[0]["input_ids"] == [1, 2, 3, 99, 4]
    assert packed[0]["attention_mask"] == [1, 1, 1, 1, 1]
    assert packed[0]["labels"] == [1, 2, 3, 99, 4]

    assert packed[1]["input_ids"] == [5, 99, 0, 0, 0]
    assert packed[1]["attention_mask"] == [1, 1, 0, 0, 0]
    assert packed[1]["labels"] == [5, 99, -100, -100, -100]


def test_pack_for_causal_lm_can_drop_remainder():
    packed = pack_for_causal_lm(
        sequences=[[1, 2, 3], [4, 5]],
        max_length=5,
        pad_token_id=0,
        eos_token_id=99,
        drop_remainder=True,
    )

    assert len(packed) == 1
    assert packed[0]["input_ids"] == [1, 2, 3, 99, 4]


def test_packed_causal_lm_collator_trims_padding_before_packing():
    collator = PackedCausalLMDataCollator(
        max_length=4,
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
    assert batch["input_ids"].tolist() == [[11, 12, 2, 21], [2, 0, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 0, 0, 0]]
    assert batch["labels"].tolist() == [[11, 12, 2, 21], [2, -100, -100, -100]]


def test_pack_for_causal_lm_raises_when_rust_is_required(monkeypatch):
    import barqtrain.data as data

    monkeypatch.setenv("BARQTRAIN_REQUIRE_RUST", "1")
    monkeypatch.setattr(data, "_get_rust_backend", lambda: None)

    with pytest.raises(RuntimeError, match="Rust backend is required"):
        data.pack_for_causal_lm(
            sequences=[[1, 2, 3]],
            max_length=4,
            pad_token_id=0,
            eos_token_id=2,
        )
