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
