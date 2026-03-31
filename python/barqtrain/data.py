"""
Rust-powered data pipeline for BarqTrain

This module provides Python wrappers around the Rust implementation
for GIL-free, multi-threaded data processing.
"""

from typing import Dict, List, Optional

import torch

from barqtrain._ffi import load_rust_backend

def _get_rust_backend():
    return load_rust_backend()


def _require_rust_backend():
    rust_backend = _get_rust_backend()
    if rust_backend is None:
        raise RuntimeError(
            "BarqTrain Rust backend is unavailable. "
            "Build/install `barqtrain_rs` first with `pip install -e .` after "
            "installing a Rust toolchain."
        )
    return rust_backend


class PackedBatch:
    """
    A packed batch containing concatenated sequences with metadata.

    Attributes:
        input_ids: The packed token IDs
        attention_mask: Attention mask (1 for real tokens, 0 for padding)
        sequence_ids: Sequence IDs to track original sequence boundaries
        position_ids: Position IDs for each token
    """

    def __init__(
        self,
        input_ids: List[int],
        attention_mask: List[int],
        sequence_ids: List[int],
        position_ids: List[int],
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sequence_ids = sequence_ids
        self.position_ids = position_ids


class PrefetchQueue:
    """
    Async prefetch queue for efficient data loading.

    This wraps the Rust implementation for lock-free iteration
    over packed batches.
    """

    def __init__(self, batches: List[PackedBatch]):
        self._batches = batches
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self) -> PackedBatch:
        if self._index < len(self._batches):
            batch = self._batches[self._index]
            self._index += 1
            return batch
        raise StopIteration

    def __len__(self) -> int:
        return len(self._batches)


def pack_sequences(
    sequences: List[List[int]], max_len: int
) -> List[PackedBatch]:
    """
    Pack sequences efficiently using bin-packing algorithm.

    This uses a first-fit decreasing algorithm implemented in Rust
    to minimize wasted padding tokens.

    Args:
        sequences: List of token sequences to pack
        max_len: Maximum sequence length for packed batches

    Returns:
        List of PackedBatch objects

    Example:
        >>> sequences = [[1, 2, 3, 4], [5, 6], [7, 8, 9]]
        >>> batches = pack_sequences(sequences, max_len=8)
        >>> len(batches)
        1
    """
    rust_batches = _require_rust_backend().pack_sequences(sequences, max_len)
    return [
        PackedBatch(
            input_ids=list(batch.input_ids),
            attention_mask=list(batch.attention_mask),
            sequence_ids=list(batch.sequence_ids),
            position_ids=list(batch.position_ids),
        )
        for batch in rust_batches
    ]


def parallel_tokenize(texts: List[str], tokenizer_path: str) -> List[List[int]]:
    """
    Parallel tokenization using Rayon (Rust implementation).

    Note: This is currently a placeholder. For production use,
    integrate with Hugging Face tokenizers library.

    Args:
        texts: List of text strings to tokenize
        tokenizer_path: Path to the tokenizer

    Returns:
        List of tokenized sequences
    """
    tokenized = _require_rust_backend().parallel_tokenize(texts, tokenizer_path)
    return [list(seq) for seq in tokenized]


def create_prefetch_queue(batches: List[PackedBatch]) -> PrefetchQueue:
    """
    Create a prefetch queue from packed batches.

    Args:
        batches: List of PackedBatch objects

    Returns:
        PrefetchQueue for iteration
    """
    return PrefetchQueue(batches)


def pack_for_causal_lm(
    sequences: List[List[int]],
    max_length: int,
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
    label_pad_token_id: int = -100,
    drop_remainder: bool = False,
) -> List[Dict[str, List[int]]]:
    """
    Pack tokenized sequences into fixed-length causal LM blocks.

    This reduces padding waste by concatenating examples with EOS separators
    and slicing the result into `max_length` training blocks.

    Args:
        sequences: Tokenized sequences without batch dimension
        max_length: Target packed block size
        pad_token_id: Padding token for the final partial block
        eos_token_id: Separator token inserted between sequences.
            Defaults to `pad_token_id` when not provided.
        label_pad_token_id: Label value used for ignored padding positions
        drop_remainder: Drop the final partial block instead of padding it

    Returns:
        List of dicts with `input_ids`, `attention_mask`, and `labels`
    """
    if max_length <= 0:
        raise ValueError("max_length must be > 0")

    rust_backend = _require_rust_backend()
    if not hasattr(rust_backend, "pack_for_causal_lm"):
        raise RuntimeError(
            "BarqTrain Rust backend is missing `pack_for_causal_lm`. "
            "Rebuild the native extension with `pip install -e .`."
        )

    rust_batches = rust_backend.pack_for_causal_lm(
        sequences,
        max_length,
        pad_token_id,
        eos_token_id,
        label_pad_token_id,
        drop_remainder,
    )
    return [
        {
            "input_ids": list(batch.input_ids),
            "attention_mask": list(batch.attention_mask),
            "labels": list(batch.labels),
        }
        for batch in rust_batches
    ]


def pack_for_padding_free_causal_lm(
    sequences: List[List[int]],
    max_length: int,
    pad_token_id: int,
    eos_token_id: Optional[int] = None,
    label_pad_token_id: int = -100,
    drop_remainder: bool = False,
    document_ids: Optional[List[int]] = None,
    document_masked: bool = False,
) -> List[Dict[str, object]]:
    """
    Pack tokenized sequences into fixed-length blocks with jagged metadata.

    This is the Phase 5 packing path: Rust emits sequence/document offsets and
    position metadata that native attention/loss consumers can use directly
    without reconstructing padded examples in Python.
    """
    if max_length <= 0:
        raise ValueError("max_length must be > 0")
    if document_ids is not None and len(document_ids) != len(sequences):
        raise ValueError("document_ids must match the number of sequences")

    rust_backend = _require_rust_backend()
    if not hasattr(rust_backend, "pack_for_padding_free_causal_lm"):
        raise RuntimeError(
            "BarqTrain Rust backend is missing `pack_for_padding_free_causal_lm`. "
            "Rebuild the native extension with `pip install -e .`."
        )

    rust_batches = rust_backend.pack_for_padding_free_causal_lm(
        sequences,
        max_length,
        pad_token_id,
        eos_token_id,
        label_pad_token_id,
        drop_remainder,
        document_ids,
        document_masked,
    )
    return [
        {
            "input_ids": list(batch.input_ids),
            "attention_mask": list(batch.attention_mask),
            "labels": list(batch.labels),
            "position_ids": list(batch.position_ids),
            "sequence_ids": list(batch.sequence_ids),
            "document_ids": list(batch.document_ids),
            "loss_mask": list(batch.loss_mask),
            "cu_seqlens": list(batch.cu_seqlens),
            "block_offsets": list(batch.block_offsets),
            "max_sequence_length": int(batch.max_sequence_length),
            "active_tokens": int(batch.active_tokens),
        }
        for batch in rust_batches
    ]


class PackedCausalLMDataCollator:
    """
    Collate tokenized causal LM samples into packed fixed-length blocks.

    The collator accepts tokenized samples with `input_ids` and optional
    `attention_mask`, trims padding, concatenates sequences, and emits packed
    blocks ready for standard Hugging Face causal LM training.
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int,
        eos_token_id: Optional[int] = None,
        label_pad_token_id: int = -100,
        drop_remainder: bool = False,
    ):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id if eos_token_id is None else eos_token_id
        self.label_pad_token_id = label_pad_token_id
        self.drop_remainder = drop_remainder

    def _trim_tokens(self, example: Dict[str, List[int]]) -> List[int]:
        input_ids = list(example["input_ids"])
        attention_mask = example.get("attention_mask")

        if attention_mask is None:
            return input_ids

        return [token for token, mask in zip(input_ids, attention_mask) if mask]

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        sequences = [self._trim_tokens(example) for example in examples]
        packed_examples = pack_for_causal_lm(
            sequences=sequences,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            label_pad_token_id=self.label_pad_token_id,
            drop_remainder=self.drop_remainder,
        )

        if not packed_examples:
            packed_examples = [
                {
                    "input_ids": [self.pad_token_id] * self.max_length,
                    "attention_mask": [0] * self.max_length,
                    "labels": [self.label_pad_token_id] * self.max_length,
                }
            ]

        return {
            "input_ids": torch.tensor(
                [example["input_ids"] for example in packed_examples], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [example["attention_mask"] for example in packed_examples], dtype=torch.long
            ),
            "labels": torch.tensor(
                [example["labels"] for example in packed_examples], dtype=torch.long
            ),
        }


class PaddingFreeCausalLMDataCollator(PackedCausalLMDataCollator):
    """
    Collate tokenized samples into packed blocks plus jagged metadata tensors.

    This collator preserves the fixed-width packed tensors for safe fallback
    paths, while also returning offsets and boundary metadata for padding-free
    native attention and loss consumers.
    """

    def __init__(
        self,
        max_length: int,
        pad_token_id: int,
        eos_token_id: Optional[int] = None,
        label_pad_token_id: int = -100,
        drop_remainder: bool = False,
        document_masked: bool = False,
        document_id_key: str = "document_id",
    ):
        super().__init__(
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            label_pad_token_id=label_pad_token_id,
            drop_remainder=drop_remainder,
        )
        self.document_masked = document_masked
        self.document_id_key = document_id_key

    def _document_id(self, example: Dict[str, List[int]], default_id: int) -> int:
        return int(example.get(self.document_id_key, default_id))

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        sequences = [self._trim_tokens(example) for example in examples]
        document_ids = [self._document_id(example, index) for index, example in enumerate(examples)]
        packed_examples = pack_for_padding_free_causal_lm(
            sequences=sequences,
            max_length=self.max_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            label_pad_token_id=self.label_pad_token_id,
            drop_remainder=self.drop_remainder,
            document_ids=document_ids,
            document_masked=self.document_masked,
        )

        if not packed_examples:
            packed_examples = [
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

        max_cu_seqlens = max(len(example["cu_seqlens"]) for example in packed_examples)
        max_block_offsets = max(len(example["block_offsets"]) for example in packed_examples)
        padded_cu_seqlens = []
        padded_block_offsets = []
        for example in packed_examples:
            cu_seqlens = list(example["cu_seqlens"])
            block_offsets = list(example["block_offsets"])
            padded_cu_seqlens.append(
                cu_seqlens + [cu_seqlens[-1] if cu_seqlens else 0] * (max_cu_seqlens - len(cu_seqlens))
            )
            padded_block_offsets.append(
                block_offsets + [-1] * (max_block_offsets - len(block_offsets))
            )

        return {
            "input_ids": torch.tensor(
                [example["input_ids"] for example in packed_examples], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [example["attention_mask"] for example in packed_examples], dtype=torch.long
            ),
            "labels": torch.tensor(
                [example["labels"] for example in packed_examples], dtype=torch.long
            ),
            "position_ids": torch.tensor(
                [example["position_ids"] for example in packed_examples], dtype=torch.long
            ),
            "sequence_ids": torch.tensor(
                [example["sequence_ids"] for example in packed_examples], dtype=torch.long
            ),
            "document_ids": torch.tensor(
                [example["document_ids"] for example in packed_examples], dtype=torch.long
            ),
            "loss_mask": torch.tensor(
                [example["loss_mask"] for example in packed_examples], dtype=torch.long
            ),
            "cu_seqlens": torch.tensor(padded_cu_seqlens, dtype=torch.long),
            "block_offsets": torch.tensor(padded_block_offsets, dtype=torch.long),
            "max_sequence_length": torch.tensor(
                [example["max_sequence_length"] for example in packed_examples], dtype=torch.long
            ),
            "active_tokens": torch.tensor(
                [example["active_tokens"] for example in packed_examples], dtype=torch.long
            ),
        }
