//! BarqTrain Rust Data Pipeline
//!
//! This module provides GIL-free, multi-threaded data processing
//! for efficient LLM fine-tuning.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Packed batch containing concatenated sequences with metadata
#[pyclass]
#[derive(Clone, Debug)]
pub struct PackedBatch {
    /// The packed token IDs
    #[pyo3(get, set)]
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    #[pyo3(get, set)]
    pub attention_mask: Vec<u8>,
    /// Sequence IDs to track which original sequence each token belongs to
    #[pyo3(get, set)]
    pub sequence_ids: Vec<i64>,
    /// Position IDs for each token
    #[pyo3(get, set)]
    pub position_ids: Vec<u64>,
}

#[pymethods]
impl PackedBatch {
    #[new]
    fn new(
        input_ids: Vec<u32>,
        attention_mask: Vec<u8>,
        sequence_ids: Vec<i64>,
        position_ids: Vec<u64>,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            sequence_ids,
            position_ids,
        }
    }
}

/// Packed causal LM block ready for tensor conversion in Python
#[pyclass]
#[derive(Clone, Debug)]
pub struct PackedCausalLMBatch {
    /// Packed token IDs padded to a fixed length
    #[pyo3(get, set)]
    pub input_ids: Vec<u32>,
    /// Attention mask (1 for real tokens, 0 for padding)
    #[pyo3(get, set)]
    pub attention_mask: Vec<u8>,
    /// Labels with ignored padding positions
    #[pyo3(get, set)]
    pub labels: Vec<i64>,
}

#[pymethods]
impl PackedCausalLMBatch {
    #[new]
    fn new(input_ids: Vec<u32>, attention_mask: Vec<u8>, labels: Vec<i64>) -> Self {
        Self {
            input_ids,
            attention_mask,
            labels,
        }
    }
}

/// Pack sequences efficiently using bin-packing algorithm
///
/// This implements a first-fit decreasing algorithm for efficient
/// sequence packing, minimizing wasted padding tokens.
///
/// Args:
///     sequences: List of token sequences to pack
///     max_len: Maximum sequence length for packed batches
///
/// Returns:
///     Packed batches as a list of PackedBatch objects
#[pyfunction]
fn pack_sequences(sequences: Vec<Vec<u32>>, max_len: usize) -> PyResult<Vec<PackedBatch>> {
    // Sort sequences by length (descending) for better packing
    let mut sorted_seqs: Vec<(usize, Vec<u32>)> = sequences
        .into_iter()
        .enumerate()
        .map(|(i, seq)| (i, seq))
        .collect();
    sorted_seqs.sort_by_key(|(_, seq)| std::cmp::Reverse(seq.len()));

    let mut batches: Vec<PackedBatch> = Vec::new();

    for (orig_idx, seq) in sorted_seqs {
        let seq_len = seq.len();

        // Try to fit in existing batches
        let mut placed = false;
        for batch in &mut batches {
            let current_len = batch.input_ids.len();
            // Check if sequence fits (accounting for possible separator)
            if current_len + seq_len + 1 <= max_len {
                // Add separator (using 0 as placeholder, will be replaced by actual token)
                if !batch.input_ids.is_empty() {
                    batch.input_ids.push(0);
                    batch.attention_mask.push(0);
                    batch.sequence_ids.push(-1);
                    batch.position_ids.push(0);
                }

                // Add sequence tokens
                let start_pos = batch.input_ids.len() as u64;
                for (pos, token) in seq.iter().enumerate() {
                    batch.input_ids.push(*token);
                    batch.attention_mask.push(1);
                    batch.sequence_ids.push(orig_idx as i64);
                    batch.position_ids.push(start_pos + pos as u64);
                }
                placed = true;
                break;
            }
        }

        // If not placed, create new batch
        if !placed {
            if seq_len > max_len {
                // Truncate sequence if too long
                let truncated_seq = &seq[..max_len];
                let input_ids = truncated_seq.to_vec();
                let attention_mask = vec![1u8; truncated_seq.len()];
                let sequence_ids = vec![orig_idx as i64; truncated_seq.len()];
                let position_ids: Vec<u64> = (0..truncated_seq.len() as u64).collect();

                batches.push(PackedBatch {
                    input_ids,
                    attention_mask,
                    sequence_ids,
                    position_ids,
                });
            } else {
                let position_ids: Vec<u64> = (0..seq_len as u64).collect();
                batches.push(PackedBatch {
                    input_ids: seq.clone(),
                    attention_mask: vec![1u8; seq_len],
                    sequence_ids: vec![orig_idx as i64; seq_len],
                    position_ids,
                });
            }
        }
    }

    Ok(batches)
}

fn flush_causal_lm_batch(
    packed_batches: &mut Vec<PackedCausalLMBatch>,
    current_tokens: &mut Vec<u32>,
    max_length: usize,
    pad_token_id: u32,
    label_pad_token_id: i64,
    drop_remainder: bool,
) {
    if current_tokens.is_empty() {
        return;
    }

    let seq_len = current_tokens.len();
    if drop_remainder && seq_len < max_length {
        current_tokens.clear();
        return;
    }

    let mut input_ids = current_tokens.clone();
    let mut attention_mask = vec![1u8; seq_len];
    let mut labels: Vec<i64> = current_tokens.iter().map(|token| *token as i64).collect();

    if seq_len < max_length {
        input_ids.resize(max_length, pad_token_id);
        attention_mask.resize(max_length, 0);
        labels.resize(max_length, label_pad_token_id);
    }

    packed_batches.push(PackedCausalLMBatch {
        input_ids,
        attention_mask,
        labels,
    });
    current_tokens.clear();
}

/// Pack tokenized sequences into fixed-length causal LM blocks.
///
/// Each input sequence is concatenated with an EOS separator when needed,
/// then emitted as dense `max_length` blocks. The final partial block may be
/// padded or dropped.
#[pyfunction]
#[pyo3(signature = (sequences, max_length, pad_token_id, eos_token_id=None, label_pad_token_id=-100, drop_remainder=false))]
fn pack_for_causal_lm(
    sequences: Vec<Vec<u32>>,
    max_length: usize,
    pad_token_id: u32,
    eos_token_id: Option<u32>,
    label_pad_token_id: i64,
    drop_remainder: bool,
) -> PyResult<Vec<PackedCausalLMBatch>> {
    if max_length == 0 {
        return Err(PyValueError::new_err("max_length must be > 0"));
    }

    let eos_token_id = eos_token_id.unwrap_or(pad_token_id);
    let mut packed_batches: Vec<PackedCausalLMBatch> = Vec::new();
    let mut current_tokens: Vec<u32> = Vec::with_capacity(max_length);

    for mut sequence in sequences {
        if sequence.is_empty() {
            continue;
        }

        if sequence.last().copied() != Some(eos_token_id) {
            sequence.push(eos_token_id);
        }

        let mut cursor = 0usize;
        while cursor < sequence.len() {
            let remaining = max_length - current_tokens.len();
            let next_cursor = (cursor + remaining).min(sequence.len());
            current_tokens.extend_from_slice(&sequence[cursor..next_cursor]);
            cursor = next_cursor;

            if current_tokens.len() == max_length {
                flush_causal_lm_batch(
                    &mut packed_batches,
                    &mut current_tokens,
                    max_length,
                    pad_token_id,
                    label_pad_token_id,
                    drop_remainder,
                );
            }
        }
    }

    flush_causal_lm_batch(
        &mut packed_batches,
        &mut current_tokens,
        max_length,
        pad_token_id,
        label_pad_token_id,
        drop_remainder,
    );

    Ok(packed_batches)
}

/// Parallel tokenization using Rayon
///
/// Note: This is a simplified placeholder. In production, you would
/// integrate with an actual tokenizer library (e.g., Hugging Face tokenizers)
///
/// Args:
///     texts: List of text strings to tokenize
///     tokenizer_path: Path to the tokenizer (unused in placeholder)
///
/// Returns:
///     Tokenized sequences (placeholder implementation)
#[pyfunction]
fn parallel_tokenize(texts: Vec<String>, _tokenizer_path: String) -> PyResult<Vec<Vec<u32>>> {
    // Parallel processing with Rayon
    let tokenized: Vec<Vec<u32>> = texts
        .par_iter()
        .map(|text| {
            // Placeholder: simple character-based tokenization
            // In production, integrate with actual tokenizer
            text.chars().map(|c| c as u32).collect()
        })
        .collect();

    Ok(tokenized)
}

/// Prefetch queue for async data loading
#[pyclass]
pub struct PrefetchQueue {
    batches: Vec<PackedBatch>,
    index: usize,
}

#[pymethods]
impl PrefetchQueue {
    #[new]
    fn new(batches: Vec<PackedBatch>) -> Self {
        Self { batches, index: 0 }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PackedBatch> {
        if slf.index < slf.batches.len() {
            let batch = slf.batches[slf.index].clone();
            slf.index += 1;
            Some(batch)
        } else {
            None
        }
    }

    fn __len__(&self) -> usize {
        self.batches.len()
    }
}

/// Create a prefetch queue from packed batches
#[pyfunction]
fn create_prefetch_queue(batches: Vec<PackedBatch>) -> PrefetchQueue {
    PrefetchQueue {
        batches,
        index: 0,
    }
}

/// Rust module definition
#[pymodule]
fn barqtrain_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PackedBatch>()?;
    m.add_class::<PackedCausalLMBatch>()?;
    m.add_class::<PrefetchQueue>()?;
    m.add_function(wrap_pyfunction!(pack_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(pack_for_causal_lm, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(create_prefetch_queue, m)?)?;
    Ok(())
}
