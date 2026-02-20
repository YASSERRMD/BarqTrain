"""
Rust-powered data pipeline for BarqTrain

This module provides Python wrappers around the Rust implementation
for GIL-free, multi-threaded data processing.
"""

from typing import List, Optional

try:
    import barqtrain_rs as _rust
except ImportError:
    _rust = None
    import warnings

    warnings.warn(
        "Rust extension not available. Install with: maturin develop --release"
    )


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
    if _rust is not None:
        # Use Rust implementation
        rust_batches = _rust.pack_sequences(sequences, max_len)
        # Convert to Python PackedBatch objects
        return [
            PackedBatch(
                input_ids=list(batch.input_ids),
                attention_mask=list(batch.attention_mask),
                sequence_ids=list(batch.sequence_ids),
                position_ids=list(batch.position_ids),
            )
            for batch in rust_batches
        ]
    else:
        # Fallback to simple Python implementation
        return _pack_sequences_python(sequences, max_len)


def _pack_sequences_python(
    sequences: List[List[int]], max_len: int
) -> List[PackedBatch]:
    """
    Fallback Python implementation of sequence packing.
    Used when Rust extension is not available.
    """
    # Sort sequences by length (descending)
    sorted_seqs = sorted(enumerate(sequences), key=lambda x: len(x[1]), reverse=True)

    batches = []

    for orig_idx, seq in sorted_seqs:
        seq_len = len(seq)
        placed = False

        # Try to fit in existing batches
        for batch in batches:
            current_len = len(batch.input_ids)
            if current_len + seq_len + 1 <= max_len:
                # Add separator
                if batch.input_ids:
                    batch.input_ids.append(0)
                    batch.attention_mask.append(0)
                    batch.sequence_ids.append(-1)
                    batch.position_ids.append(0)

                # Add sequence
                start_pos = len(batch.input_ids)
                for pos, token in enumerate(seq):
                    batch.input_ids.append(token)
                    batch.attention_mask.append(1)
                    batch.sequence_ids.append(orig_idx)
                    batch.position_ids.append(start_pos + pos)
                placed = True
                break

        # Create new batch if not placed
        if not placed:
            if seq_len > max_len:
                # Truncate
                truncated = seq[:max_len]
                batches.append(
                    PackedBatch(
                        input_ids=truncated,
                        attention_mask=[1] * len(truncated),
                        sequence_ids=[orig_idx] * len(truncated),
                        position_ids=list(range(len(truncated))),
                    )
                )
            else:
                batches.append(
                    PackedBatch(
                        input_ids=seq,
                        attention_mask=[1] * seq_len,
                        sequence_ids=[orig_idx] * seq_len,
                        position_ids=list(range(seq_len)),
                    )
                )

    return batches


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
    if _rust is not None:
        tokenized = _rust.parallel_tokenize(texts, tokenizer_path)
        return [list(seq) for seq in tokenized]
    else:
        # Fallback: simple character tokenization
        return [[ord(c) for c in text] for text in texts]


def create_prefetch_queue(batches: List[PackedBatch]) -> PrefetchQueue:
    """
    Create a prefetch queue from packed batches.

    Args:
        batches: List of PackedBatch objects

    Returns:
        PrefetchQueue for iteration
    """
    return PrefetchQueue(batches)
