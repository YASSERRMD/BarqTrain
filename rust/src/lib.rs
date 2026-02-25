//! BarqTrain Rust Data Pipeline
//!
//! This module provides GIL-free, multi-threaded data processing
//! for efficient LLM fine-tuning.

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
    m.add_class::<PrefetchQueue>()?;
    m.add_function(wrap_pyfunction!(pack_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(create_prefetch_queue, m)?)?;
    Ok(())
}
