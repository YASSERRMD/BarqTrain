"""Padding-free packing and consumption tests."""

import torch

from barqtrain.ops import (
    padding_free_attention,
    padding_free_chunked_cross_entropy_loss,
)


def test_padding_free_attention_matches_per_segment_sdpa():
    q = torch.randn(5, 2, 4)
    k = torch.randn(5, 2, 4)
    v = torch.randn(5, 2, 4)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.long)

    output = padding_free_attention(q, k, v, cu_seqlens=cu_seqlens)

    expected_segments = []
    for start, end in ((0, 3), (3, 5)):
        q_segment = q[start:end].transpose(0, 1).unsqueeze(0)
        k_segment = k[start:end].transpose(0, 1).unsqueeze(0)
        v_segment = v[start:end].transpose(0, 1).unsqueeze(0)
        expected = torch.nn.functional.scaled_dot_product_attention(
            q_segment,
            k_segment,
            v_segment,
            is_causal=True,
        ).squeeze(0).transpose(0, 1)
        expected_segments.append(expected)

    expected_output = torch.cat(expected_segments, dim=0)
    assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-6)


def test_padding_free_chunked_cross_entropy_respects_loss_mask():
    hidden = torch.randn(5, 8)
    lm_head = torch.randn(16, 8)
    labels = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    loss_mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.long)

    loss = padding_free_chunked_cross_entropy_loss(
        hidden,
        lm_head,
        labels,
        loss_mask=loss_mask,
    )
    manual_labels = labels.masked_fill(~loss_mask.bool(), -100)
    manual_loss = torch.nn.functional.cross_entropy(
        torch.nn.functional.linear(hidden, lm_head),
        manual_labels,
        ignore_index=-100,
    )

    assert torch.allclose(loss, manual_loss, rtol=1e-5, atol=1e-6)
