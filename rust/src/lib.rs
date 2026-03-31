//! BarqTrain Rust Data Pipeline
//!
//! This module provides GIL-free, multi-threaded data processing
//! for efficient LLM fine-tuning.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;
use std::collections::HashSet;

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

/// Packed causal LM block with jagged metadata for padding-free training.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PackedPaddingFreeBatch {
    #[pyo3(get, set)]
    pub input_ids: Vec<u32>,
    #[pyo3(get, set)]
    pub attention_mask: Vec<u8>,
    #[pyo3(get, set)]
    pub labels: Vec<i64>,
    #[pyo3(get, set)]
    pub position_ids: Vec<u64>,
    #[pyo3(get, set)]
    pub sequence_ids: Vec<i64>,
    #[pyo3(get, set)]
    pub document_ids: Vec<i64>,
    #[pyo3(get, set)]
    pub loss_mask: Vec<u8>,
    #[pyo3(get, set)]
    pub cu_seqlens: Vec<u32>,
    #[pyo3(get, set)]
    pub block_offsets: Vec<u32>,
    #[pyo3(get, set)]
    pub max_sequence_length: usize,
    #[pyo3(get, set)]
    pub active_tokens: usize,
}

#[pymethods]
impl PackedPaddingFreeBatch {
    #[new]
    fn new(
        input_ids: Vec<u32>,
        attention_mask: Vec<u8>,
        labels: Vec<i64>,
        position_ids: Vec<u64>,
        sequence_ids: Vec<i64>,
        document_ids: Vec<i64>,
        loss_mask: Vec<u8>,
        cu_seqlens: Vec<u32>,
        block_offsets: Vec<u32>,
        max_sequence_length: usize,
        active_tokens: usize,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            labels,
            position_ids,
            sequence_ids,
            document_ids,
            loss_mask,
            cu_seqlens,
            block_offsets,
            max_sequence_length,
            active_tokens,
        }
    }
}

/// Native memory breakdown emitted by the Rust benchmark/reporting helpers.
#[pyclass]
#[derive(Clone, Debug)]
pub struct MemoryBreakdown {
    #[pyo3(get)]
    pub resident_model_mb: f64,
    #[pyo3(get)]
    pub kv_cache_mb: f64,
    #[pyo3(get)]
    pub temporary_decode_buffers_mb: f64,
    #[pyo3(get)]
    pub training_peak_vram_mb: f64,
    #[pyo3(get)]
    pub inference_peak_vram_mb: f64,
    #[pyo3(get)]
    pub detailed_profiling: bool,
}

#[pymethods]
impl MemoryBreakdown {
    #[new]
    fn new(
        resident_model_mb: f64,
        kv_cache_mb: f64,
        temporary_decode_buffers_mb: f64,
        training_peak_vram_mb: f64,
        inference_peak_vram_mb: f64,
        detailed_profiling: bool,
    ) -> Self {
        Self {
            resident_model_mb,
            kv_cache_mb,
            temporary_decode_buffers_mb,
            training_peak_vram_mb,
            inference_peak_vram_mb,
            detailed_profiling,
        }
    }
}

/// Canonical Phase 1 decode benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct DecodeBenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub prompt_length: usize,
    #[pyo3(get)]
    pub decode_length: usize,
    #[pyo3(get)]
    pub batch_size: usize,
}

#[pymethods]
impl DecodeBenchmarkProfile {
    #[new]
    fn new(name: String, prompt_length: usize, decode_length: usize, batch_size: usize) -> Self {
        Self {
            name,
            prompt_length,
            decode_length,
            batch_size,
        }
    }
}

/// Canonical Phase 2 KV-cache benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct KVCacheBenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub prompt_length: usize,
    #[pyo3(get)]
    pub decode_length: usize,
    #[pyo3(get)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub request_count: usize,
    #[pyo3(get)]
    pub fixed_vram_budget_mb: u64,
}

#[pymethods]
impl KVCacheBenchmarkProfile {
    #[new]
    fn new(
        name: String,
        prompt_length: usize,
        decode_length: usize,
        batch_size: usize,
        request_count: usize,
        fixed_vram_budget_mb: u64,
    ) -> Self {
        Self {
            name,
            prompt_length,
            decode_length,
            batch_size,
            request_count,
            fixed_vram_budget_mb,
        }
    }
}

/// Canonical Phase 4 fused projection benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ProjectionBenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub sequence_length: usize,
    #[pyo3(get)]
    pub prompt_length: usize,
    #[pyo3(get)]
    pub decode_length: usize,
}

#[pymethods]
impl ProjectionBenchmarkProfile {
    #[new]
    fn new(
        name: String,
        batch_size: usize,
        sequence_length: usize,
        prompt_length: usize,
        decode_length: usize,
    ) -> Self {
        Self {
            name,
            batch_size,
            sequence_length,
            prompt_length,
            decode_length,
        }
    }
}

/// Canonical Phase 5 packed training benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PackedTrainingBenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub sequence_length: usize,
    #[pyo3(get)]
    pub document_masked: bool,
}

#[pymethods]
impl PackedTrainingBenchmarkProfile {
    #[new]
    fn new(
        name: String,
        batch_size: usize,
        sequence_length: usize,
        document_masked: bool,
    ) -> Self {
        Self {
            name,
            batch_size,
            sequence_length,
            document_masked,
        }
    }
}

/// Canonical Phase 6 activation-checkpoint benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct ActivationCheckpointBenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub num_steps: usize,
}

#[pymethods]
impl ActivationCheckpointBenchmarkProfile {
    #[new]
    fn new(name: String, num_steps: usize) -> Self {
        Self { name, num_steps }
    }
}

/// Canonical Phase 7 optimizer benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct OptimizerBenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub num_steps: usize,
}

#[pymethods]
impl OptimizerBenchmarkProfile {
    #[new]
    fn new(name: String, num_steps: usize) -> Self {
        Self { name, num_steps }
    }
}

/// Canonical Phase 8 RMSNorm block-fusion benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct RMSNormFusionBenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub sequence_length: usize,
    #[pyo3(get)]
    pub hidden_size: usize,
    #[pyo3(get)]
    pub projection_size: usize,
}

#[pymethods]
impl RMSNormFusionBenchmarkProfile {
    #[new]
    fn new(
        name: String,
        batch_size: usize,
        sequence_length: usize,
        hidden_size: usize,
        projection_size: usize,
    ) -> Self {
        Self {
            name,
            batch_size,
            sequence_length,
            hidden_size,
            projection_size,
        }
    }
}

/// Canonical Phase 9 attention benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct AttentionBenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub prompt_length: usize,
    #[pyo3(get)]
    pub decode_length: usize,
    #[pyo3(get)]
    pub cache_layout: String,
}

#[pymethods]
impl AttentionBenchmarkProfile {
    #[new]
    fn new(
        name: String,
        batch_size: usize,
        prompt_length: usize,
        decode_length: usize,
        cache_layout: String,
    ) -> Self {
        Self {
            name,
            batch_size,
            prompt_length,
            decode_length,
            cache_layout,
        }
    }
}

/// Canonical Phase 10 fused LoRA benchmark profile.
#[pyclass]
#[derive(Clone, Debug)]
pub struct LoRABenchmarkProfile {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub sequence_length: usize,
    #[pyo3(get)]
    pub packed_training: bool,
}

#[pymethods]
impl LoRABenchmarkProfile {
    #[new]
    fn new(name: String, batch_size: usize, sequence_length: usize, packed_training: bool) -> Self {
        Self {
            name,
            batch_size,
            sequence_length,
            packed_training,
        }
    }
}

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn storage_nbytes(tensor: &PyAny) -> PyResult<Option<(usize, u64)>> {
    let is_cuda = tensor.getattr("is_cuda")?.extract::<bool>()?;
    if !is_cuda {
        return Ok(None);
    }

    let storage = tensor
        .call_method0("untyped_storage")
        .or_else(|_| tensor.call_method0("storage"))?;
    let storage_ptr = storage.call_method0("data_ptr")?.extract::<usize>()?;
    let nbytes = storage
        .call_method0("nbytes")
        .and_then(|value| value.extract::<u64>())
        .or_else(|_| {
            let storage_size = storage.call_method0("size")?.extract::<u64>()?;
            let element_size = tensor.call_method0("element_size")?.extract::<u64>()?;
            Ok::<u64, PyErr>(storage_size.saturating_mul(element_size))
        })?;

    Ok(Some((storage_ptr, nbytes)))
}

fn accumulate_unique_cuda_bytes(iterable: &PyAny, seen: &mut HashSet<usize>) -> PyResult<u64> {
    let mut total_bytes = 0u64;
    for item in iterable.iter()? {
        let tensor = item?;
        if let Some((storage_ptr, nbytes)) = storage_nbytes(tensor)? {
            if seen.insert(storage_ptr) {
                total_bytes = total_bytes.saturating_add(nbytes);
            }
        }
    }
    Ok(total_bytes)
}

/// Measure the resident CUDA bytes owned by a model's parameters and buffers.
#[pyfunction]
fn model_cuda_bytes(model: &PyAny) -> PyResult<u64> {
    let mut seen_storages = HashSet::new();
    let mut total_bytes = 0u64;

    if let Ok(parameters) = model.call_method0("parameters") {
        total_bytes = total_bytes.saturating_add(accumulate_unique_cuda_bytes(
            parameters,
            &mut seen_storages,
        )?);
    }
    if let Ok(buffers) = model.call_method0("buffers") {
        total_bytes = total_bytes.saturating_add(accumulate_unique_cuda_bytes(
            buffers,
            &mut seen_storages,
        )?);
    }

    Ok(total_bytes)
}

/// Build the canonical benchmark memory report from native byte counters.
#[pyfunction]
#[pyo3(signature = (
    resident_model_bytes,
    kv_cache_bytes,
    temporary_decode_buffer_bytes,
    training_peak_bytes,
    inference_peak_bytes,
    detailed_profiling=false
))]
fn build_memory_breakdown(
    resident_model_bytes: u64,
    kv_cache_bytes: u64,
    temporary_decode_buffer_bytes: u64,
    training_peak_bytes: u64,
    inference_peak_bytes: u64,
    detailed_profiling: bool,
) -> MemoryBreakdown {
    MemoryBreakdown {
        resident_model_mb: bytes_to_mb(resident_model_bytes),
        kv_cache_mb: bytes_to_mb(kv_cache_bytes),
        temporary_decode_buffers_mb: bytes_to_mb(temporary_decode_buffer_bytes),
        training_peak_vram_mb: bytes_to_mb(training_peak_bytes),
        inference_peak_vram_mb: bytes_to_mb(inference_peak_bytes),
        detailed_profiling,
    }
}

/// Emit the required Phase 1 decode benchmark matrix.
#[pyfunction]
#[pyo3(signature = (
    batch_sizes,
    short_prompt_length=64,
    long_prompt_length=1024,
    short_decode_length=32,
    long_decode_length=256
))]
fn phase1_decode_profiles(
    batch_sizes: Vec<usize>,
    short_prompt_length: usize,
    long_prompt_length: usize,
    short_decode_length: usize,
    long_decode_length: usize,
) -> Vec<DecodeBenchmarkProfile> {
    let mut profiles = Vec::with_capacity(batch_sizes.len() * 2);
    for batch_size in batch_sizes {
        profiles.push(DecodeBenchmarkProfile {
            name: "short_prompt_long_decode".to_string(),
            prompt_length: short_prompt_length,
            decode_length: long_decode_length,
            batch_size,
        });
        profiles.push(DecodeBenchmarkProfile {
            name: "long_prompt_short_decode".to_string(),
            prompt_length: long_prompt_length,
            decode_length: short_decode_length,
            batch_size,
        });
    }
    profiles
}

/// Emit the required Phase 2 contiguous-vs-paged KV benchmark matrix.
#[pyfunction]
#[pyo3(signature = (
    batch_sizes,
    short_prompt_length=64,
    long_prompt_length=1024,
    short_decode_length=32,
    long_decode_length=256,
    serving_request_count=8,
    fixed_vram_budget_mb=2048
))]
fn phase2_kv_cache_profiles(
    batch_sizes: Vec<usize>,
    short_prompt_length: usize,
    long_prompt_length: usize,
    short_decode_length: usize,
    long_decode_length: usize,
    serving_request_count: usize,
    fixed_vram_budget_mb: u64,
) -> Vec<KVCacheBenchmarkProfile> {
    let mut profiles = Vec::with_capacity(batch_sizes.len() * 3);
    for batch_size in batch_sizes {
        profiles.push(KVCacheBenchmarkProfile {
            name: "long_prompt_generation".to_string(),
            prompt_length: long_prompt_length,
            decode_length: long_decode_length,
            batch_size,
            request_count: 1,
            fixed_vram_budget_mb: 0,
        });
        profiles.push(KVCacheBenchmarkProfile {
            name: "multi_request_serving".to_string(),
            prompt_length: short_prompt_length,
            decode_length: long_decode_length,
            batch_size,
            request_count: serving_request_count.max(1),
            fixed_vram_budget_mb: 0,
        });
        profiles.push(KVCacheBenchmarkProfile {
            name: "fixed_vram_batch_growth".to_string(),
            prompt_length: long_prompt_length,
            decode_length: short_decode_length,
            batch_size,
            request_count: 1,
            fixed_vram_budget_mb,
        });
    }
    profiles
}

/// Emit the required Phase 3 quantized KV benchmark matrix.
#[pyfunction]
#[pyo3(signature = (
    batch_sizes,
    short_prompt_length=64,
    long_prompt_length=1024,
    quality_decode_length=64,
    long_decode_length=256
))]
fn phase3_quantized_kv_profiles(
    batch_sizes: Vec<usize>,
    short_prompt_length: usize,
    long_prompt_length: usize,
    quality_decode_length: usize,
    long_decode_length: usize,
) -> Vec<KVCacheBenchmarkProfile> {
    let mut profiles = Vec::with_capacity(batch_sizes.len() * 3);
    for batch_size in batch_sizes {
        profiles.push(KVCacheBenchmarkProfile {
            name: "memory_savings_vs_latency".to_string(),
            prompt_length: long_prompt_length,
            decode_length: long_decode_length,
            batch_size,
            request_count: 1,
            fixed_vram_budget_mb: 0,
        });
        profiles.push(KVCacheBenchmarkProfile {
            name: "long_context_generation_quality".to_string(),
            prompt_length: long_prompt_length,
            decode_length: quality_decode_length,
            batch_size,
            request_count: 1,
            fixed_vram_budget_mb: 0,
        });
        profiles.push(KVCacheBenchmarkProfile {
            name: "throughput_per_gb".to_string(),
            prompt_length: short_prompt_length,
            decode_length: long_decode_length,
            batch_size,
            request_count: 1,
            fixed_vram_budget_mb: 0,
        });
    }
    profiles
}

/// Emit the required Phase 4 fused projection benchmark matrix.
#[pyfunction]
#[pyo3(signature = (
    batch_sizes,
    sequence_length=512,
    short_prompt_length=64,
    long_prompt_length=1024,
    short_decode_length=32,
    long_decode_length=256
))]
fn phase4_vocab_projection_profiles(
    batch_sizes: Vec<usize>,
    sequence_length: usize,
    short_prompt_length: usize,
    long_prompt_length: usize,
    short_decode_length: usize,
    long_decode_length: usize,
) -> Vec<ProjectionBenchmarkProfile> {
    let mut profiles = Vec::with_capacity(batch_sizes.len() * 2);
    for batch_size in batch_sizes {
        profiles.push(ProjectionBenchmarkProfile {
            name: "vocab_heavy_long_decode".to_string(),
            batch_size,
            sequence_length,
            prompt_length: short_prompt_length,
            decode_length: long_decode_length,
        });
        profiles.push(ProjectionBenchmarkProfile {
            name: "vocab_heavy_long_context".to_string(),
            batch_size,
            sequence_length,
            prompt_length: long_prompt_length,
            decode_length: short_decode_length,
        });
    }
    profiles
}

/// Emit the required Phase 5 packed training benchmark matrix.
#[pyfunction]
#[pyo3(signature = (
    batch_sizes,
    sequence_length=512
))]
fn phase5_packed_training_profiles(
    batch_sizes: Vec<usize>,
    sequence_length: usize,
) -> Vec<PackedTrainingBenchmarkProfile> {
    let mut profiles = Vec::with_capacity(batch_sizes.len() * 2);
    for batch_size in batch_sizes {
        profiles.push(PackedTrainingBenchmarkProfile {
            name: "matched_effective_tokens".to_string(),
            batch_size,
            sequence_length,
            document_masked: false,
        });
        profiles.push(PackedTrainingBenchmarkProfile {
            name: "document_masked_training".to_string(),
            batch_size,
            sequence_length,
            document_masked: true,
        });
    }
    profiles
}

/// Emit the required Phase 6 activation-checkpoint benchmark presets.
#[pyfunction]
#[pyo3(signature = (num_steps=3))]
fn phase6_activation_checkpoint_profiles(
    num_steps: usize,
) -> Vec<ActivationCheckpointBenchmarkProfile> {
    vec![
        ActivationCheckpointBenchmarkProfile {
            name: "max_throughput".to_string(),
            num_steps: num_steps.max(1),
        },
        ActivationCheckpointBenchmarkProfile {
            name: "balanced".to_string(),
            num_steps: num_steps.max(1),
        },
        ActivationCheckpointBenchmarkProfile {
            name: "max_memory_saving".to_string(),
            num_steps: num_steps.max(1),
        },
    ]
}

/// Emit the required Phase 7 optimizer benchmark modes.
#[pyfunction]
#[pyo3(signature = (num_steps=5))]
fn phase7_optimizer_profiles(
    num_steps: usize,
) -> Vec<OptimizerBenchmarkProfile> {
    let steps = num_steps.max(1);
    vec![
        OptimizerBenchmarkProfile {
            name: "adamw".to_string(),
            num_steps: steps,
        },
        OptimizerBenchmarkProfile {
            name: "barqtrain_adamw".to_string(),
            num_steps: steps,
        },
        OptimizerBenchmarkProfile {
            name: "barqtrain_adamw_compact".to_string(),
            num_steps: steps,
        },
        OptimizerBenchmarkProfile {
            name: "barqtrain_adamw_paged".to_string(),
            num_steps: steps,
        },
    ]
}

/// Emit the required Phase 8 RMSNorm block-fusion benchmark matrix.
#[pyfunction]
#[pyo3(signature = (
    batch_sizes,
    sequence_length=512,
    hidden_size=4096,
    attention_projection_size=4096,
    mlp_projection_size=16384
))]
fn phase8_rmsnorm_fusion_profiles(
    batch_sizes: Vec<usize>,
    sequence_length: usize,
    hidden_size: usize,
    attention_projection_size: usize,
    mlp_projection_size: usize,
) -> Vec<RMSNormFusionBenchmarkProfile> {
    let mut profiles = Vec::with_capacity(batch_sizes.len() * 3);
    for batch_size in batch_sizes {
        profiles.push(RMSNormFusionBenchmarkProfile {
            name: "residual_add_rmsnorm".to_string(),
            batch_size,
            sequence_length,
            hidden_size,
            projection_size: hidden_size,
        });
        profiles.push(RMSNormFusionBenchmarkProfile {
            name: "attention_input_projection".to_string(),
            batch_size,
            sequence_length,
            hidden_size,
            projection_size: attention_projection_size,
        });
        profiles.push(RMSNormFusionBenchmarkProfile {
            name: "mlp_input_projection".to_string(),
            batch_size,
            sequence_length,
            hidden_size,
            projection_size: mlp_projection_size,
        });
    }
    profiles
}

/// Emit the required Phase 9 attention benchmark matrix.
#[pyfunction]
#[pyo3(signature = (
    batch_sizes,
    short_prompt_length=64,
    long_prompt_length=1024,
    short_decode_length=32,
    long_decode_length=256
))]
fn phase9_attention_profiles(
    batch_sizes: Vec<usize>,
    short_prompt_length: usize,
    long_prompt_length: usize,
    short_decode_length: usize,
    long_decode_length: usize,
) -> Vec<AttentionBenchmarkProfile> {
    let mut profiles = Vec::with_capacity(batch_sizes.len() * 3);
    for batch_size in batch_sizes {
        profiles.push(AttentionBenchmarkProfile {
            name: "prefill_throughput".to_string(),
            batch_size,
            prompt_length: long_prompt_length,
            decode_length: 0,
            cache_layout: "none".to_string(),
        });
        profiles.push(AttentionBenchmarkProfile {
            name: "decode_throughput".to_string(),
            batch_size,
            prompt_length: short_prompt_length,
            decode_length: long_decode_length,
            cache_layout: "paged".to_string(),
        });
        profiles.push(AttentionBenchmarkProfile {
            name: "long_context_serving".to_string(),
            batch_size,
            prompt_length: long_prompt_length.saturating_mul(2),
            decode_length: short_decode_length,
            cache_layout: "paged_quantized".to_string(),
        });
    }
    profiles
}

/// Emit the required Phase 10 fused LoRA benchmark matrix.
#[pyfunction]
#[pyo3(signature = (
    batch_sizes,
    sequence_length=512
))]
fn phase10_lora_profiles(
    batch_sizes: Vec<usize>,
    sequence_length: usize,
) -> Vec<LoRABenchmarkProfile> {
    let mut profiles = Vec::with_capacity(batch_sizes.len() * 2);
    for batch_size in batch_sizes {
        profiles.push(LoRABenchmarkProfile {
            name: "dense_chunked_loss".to_string(),
            batch_size,
            sequence_length,
            packed_training: false,
        });
        profiles.push(LoRABenchmarkProfile {
            name: "packed_chunked_loss".to_string(),
            batch_size,
            sequence_length,
            packed_training: true,
        });
    }
    profiles
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
                    input_ids: seq,
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
        *current_tokens = Vec::with_capacity(max_length);
        return;
    }

    let mut input_ids = std::mem::replace(current_tokens, Vec::with_capacity(max_length));
    let mut attention_mask = vec![1u8; seq_len];
    let mut labels: Vec<i64> = input_ids.iter().map(|token| *token as i64).collect();

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
}

fn flush_padding_free_batch(
    packed_batches: &mut Vec<PackedPaddingFreeBatch>,
    current_tokens: &mut Vec<u32>,
    current_position_ids: &mut Vec<u64>,
    current_sequence_ids: &mut Vec<i64>,
    current_document_ids: &mut Vec<i64>,
    current_segment_lengths: &mut Vec<usize>,
    current_boundary_starts: &mut Vec<usize>,
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
        *current_tokens = Vec::with_capacity(max_length);
        *current_position_ids = Vec::with_capacity(max_length);
        *current_sequence_ids = Vec::with_capacity(max_length);
        *current_document_ids = Vec::with_capacity(max_length);
        current_segment_lengths.clear();
        current_boundary_starts.clear();
        return;
    }

    let mut input_ids = std::mem::replace(current_tokens, Vec::with_capacity(max_length));
    let mut position_ids = std::mem::replace(current_position_ids, Vec::with_capacity(max_length));
    let mut sequence_ids = std::mem::replace(current_sequence_ids, Vec::with_capacity(max_length));
    let mut document_ids = std::mem::replace(current_document_ids, Vec::with_capacity(max_length));
    let segment_lengths = std::mem::take(current_segment_lengths);
    let boundary_starts = std::mem::take(current_boundary_starts);

    let mut attention_mask = vec![1u8; seq_len];
    let mut labels: Vec<i64> = input_ids.iter().map(|token| *token as i64).collect();
    for boundary_start in boundary_starts {
        if boundary_start < labels.len() {
            labels[boundary_start] = label_pad_token_id;
        }
    }
    let mut loss_mask: Vec<u8> = labels
        .iter()
        .map(|label| if *label == label_pad_token_id { 0 } else { 1 })
        .collect();

    let mut cu_seqlens = Vec::with_capacity(segment_lengths.len() + 1);
    cu_seqlens.push(0);
    let mut block_offsets = Vec::with_capacity(segment_lengths.len());
    let mut running = 0usize;
    let mut max_sequence_length = 0usize;
    for segment_length in segment_lengths {
        block_offsets.push(running as u32);
        running = running.saturating_add(segment_length);
        cu_seqlens.push(running as u32);
        max_sequence_length = max_sequence_length.max(segment_length);
    }

    if seq_len < max_length {
        input_ids.resize(max_length, pad_token_id);
        attention_mask.resize(max_length, 0);
        labels.resize(max_length, label_pad_token_id);
        position_ids.resize(max_length, 0);
        sequence_ids.resize(max_length, -1);
        document_ids.resize(max_length, -1);
        loss_mask.resize(max_length, 0);
    }

    packed_batches.push(PackedPaddingFreeBatch {
        input_ids,
        attention_mask,
        labels,
        position_ids,
        sequence_ids,
        document_ids,
        loss_mask,
        cu_seqlens,
        block_offsets,
        max_sequence_length,
        active_tokens: seq_len,
    });
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
    let prepared_sequences: Vec<Vec<u32>> = sequences
        .into_par_iter()
        .filter_map(|mut sequence| {
            if sequence.is_empty() {
                return None;
            }

            if sequence.last().copied() != Some(eos_token_id) {
                sequence.push(eos_token_id);
            }
            Some(sequence)
        })
        .collect();

    let total_tokens: usize = prepared_sequences.par_iter().map(Vec::len).sum();
    let estimated_batches = if drop_remainder {
        total_tokens / max_length
    } else {
        (total_tokens + max_length.saturating_sub(1)) / max_length
    };

    let mut packed_batches: Vec<PackedCausalLMBatch> =
        Vec::with_capacity(estimated_batches.max(1));
    let mut current_tokens: Vec<u32> = Vec::with_capacity(max_length);

    for sequence in prepared_sequences {
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

/// Pack tokenized sequences into fixed-length causal LM blocks with jagged metadata.
#[pyfunction]
#[pyo3(signature = (
    sequences,
    max_length,
    pad_token_id,
    eos_token_id=None,
    label_pad_token_id=-100,
    drop_remainder=false,
    document_ids=None,
    document_masked=false
))]
fn pack_for_padding_free_causal_lm(
    sequences: Vec<Vec<u32>>,
    max_length: usize,
    pad_token_id: u32,
    eos_token_id: Option<u32>,
    label_pad_token_id: i64,
    drop_remainder: bool,
    document_ids: Option<Vec<i64>>,
    document_masked: bool,
) -> PyResult<Vec<PackedPaddingFreeBatch>> {
    if max_length == 0 {
        return Err(PyValueError::new_err("max_length must be > 0"));
    }

    if let Some(ref provided_document_ids) = document_ids {
        if provided_document_ids.len() != sequences.len() {
            return Err(PyValueError::new_err(
                "document_ids must match the number of sequences",
            ));
        }
    }

    let eos_token_id = eos_token_id.unwrap_or(pad_token_id);
    let mut prepared_sequences: Vec<(usize, i64, Vec<u32>)> = Vec::new();
    for (index, mut sequence) in sequences.into_iter().enumerate() {
        if sequence.is_empty() {
            continue;
        }
        if sequence.last().copied() != Some(eos_token_id) {
            sequence.push(eos_token_id);
        }
        let document_id = document_ids
            .as_ref()
            .and_then(|ids| ids.get(index).copied())
            .unwrap_or(index as i64);
        prepared_sequences.push((index, document_id, sequence));
    }

    let total_tokens: usize = prepared_sequences.iter().map(|(_, _, sequence)| sequence.len()).sum();
    let estimated_batches = if drop_remainder {
        total_tokens / max_length
    } else {
        (total_tokens + max_length.saturating_sub(1)) / max_length
    };

    let mut packed_batches: Vec<PackedPaddingFreeBatch> =
        Vec::with_capacity(estimated_batches.max(1));
    let mut current_tokens: Vec<u32> = Vec::with_capacity(max_length);
    let mut current_position_ids: Vec<u64> = Vec::with_capacity(max_length);
    let mut current_sequence_ids: Vec<i64> = Vec::with_capacity(max_length);
    let mut current_document_ids: Vec<i64> = Vec::with_capacity(max_length);
    let mut current_segment_lengths: Vec<usize> = Vec::new();
    let mut current_boundary_starts: Vec<usize> = Vec::new();

    for (sequence_index, document_id, sequence) in prepared_sequences {
        let mut cursor = 0usize;
        while cursor < sequence.len() {
            if current_tokens.len() == max_length {
                flush_padding_free_batch(
                    &mut packed_batches,
                    &mut current_tokens,
                    &mut current_position_ids,
                    &mut current_sequence_ids,
                    &mut current_document_ids,
                    &mut current_segment_lengths,
                    &mut current_boundary_starts,
                    max_length,
                    pad_token_id,
                    label_pad_token_id,
                    drop_remainder,
                );
            }

            let available = max_length - current_tokens.len();
            let next_cursor = (cursor + available).min(sequence.len());
            let chunk = &sequence[cursor..next_cursor];
            let segment_start = current_tokens.len();
            if document_masked && cursor == 0 && segment_start > 0 {
                current_boundary_starts.push(segment_start);
            }
            current_segment_lengths.push(chunk.len());

            for (offset, token) in chunk.iter().enumerate() {
                current_tokens.push(*token);
                current_position_ids.push((cursor + offset) as u64);
                current_sequence_ids.push(sequence_index as i64);
                current_document_ids.push(document_id);
            }
            cursor = next_cursor;

            if current_tokens.len() == max_length {
                flush_padding_free_batch(
                    &mut packed_batches,
                    &mut current_tokens,
                    &mut current_position_ids,
                    &mut current_sequence_ids,
                    &mut current_document_ids,
                    &mut current_segment_lengths,
                    &mut current_boundary_starts,
                    max_length,
                    pad_token_id,
                    label_pad_token_id,
                    drop_remainder,
                );
            }
        }
    }

    flush_padding_free_batch(
        &mut packed_batches,
        &mut current_tokens,
        &mut current_position_ids,
        &mut current_sequence_ids,
        &mut current_document_ids,
        &mut current_segment_lengths,
        &mut current_boundary_starts,
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
    m.add_class::<PackedPaddingFreeBatch>()?;
    m.add_class::<MemoryBreakdown>()?;
    m.add_class::<DecodeBenchmarkProfile>()?;
    m.add_class::<KVCacheBenchmarkProfile>()?;
    m.add_class::<ProjectionBenchmarkProfile>()?;
    m.add_class::<PackedTrainingBenchmarkProfile>()?;
    m.add_class::<ActivationCheckpointBenchmarkProfile>()?;
    m.add_class::<OptimizerBenchmarkProfile>()?;
    m.add_class::<RMSNormFusionBenchmarkProfile>()?;
    m.add_class::<AttentionBenchmarkProfile>()?;
    m.add_class::<LoRABenchmarkProfile>()?;
    m.add_class::<PrefetchQueue>()?;
    m.add_function(wrap_pyfunction!(pack_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(pack_for_causal_lm, m)?)?;
    m.add_function(wrap_pyfunction!(pack_for_padding_free_causal_lm, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(create_prefetch_queue, m)?)?;
    m.add_function(wrap_pyfunction!(model_cuda_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(build_memory_breakdown, m)?)?;
    m.add_function(wrap_pyfunction!(phase1_decode_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase2_kv_cache_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase3_quantized_kv_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase4_vocab_projection_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase5_packed_training_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase6_activation_checkpoint_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase7_optimizer_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase8_rmsnorm_fusion_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase9_attention_profiles, m)?)?;
    m.add_function(wrap_pyfunction!(phase10_lora_profiles, m)?)?;
    Ok(())
}
