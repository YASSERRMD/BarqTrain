/**
 * BarqTrain CUDA Extension - Python Bindings
 *
 * This file provides PyBind11 bindings for BarqTrain's CUDA kernels.
 */

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

struct MemoryBucketCounters {
    int64_t current_bytes = 0;
    int64_t peak_bytes = 0;
};

struct MemoryTrackerState {
    bool enabled = false;
    MemoryBucketCounters resident_model;
    MemoryBucketCounters kv_cache;
    MemoryBucketCounters decode_temp;
    int64_t training_peak_bytes = 0;
    int64_t inference_peak_bytes = 0;
};

MemoryTrackerState& memory_tracker_state() {
    static MemoryTrackerState state;
    return state;
}

std::mutex& memory_tracker_mutex() {
    static std::mutex tracker_mutex;
    return tracker_mutex;
}

MemoryBucketCounters* resolve_bucket(MemoryTrackerState& state, const std::string& bucket) {
    if (bucket == "resident_model") {
        return &state.resident_model;
    }
    if (bucket == "kv_cache") {
        return &state.kv_cache;
    }
    if (bucket == "decode_temp") {
        return &state.decode_temp;
    }
    throw std::invalid_argument("Unknown BarqTrain memory bucket: " + bucket);
}

void clear_bucket(MemoryBucketCounters* bucket, bool reset_peak) {
    bucket->current_bytes = 0;
    if (reset_peak) {
        bucket->peak_bytes = 0;
    }
}

}  // namespace

// Forward declarations for CUDA kernels
torch::Tensor fused_rmsnorm_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    float eps
);

torch::Tensor fused_rmsnorm_backward_cuda(
    torch::Tensor grad_out,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor rms
);

torch::Tensor flash_attention_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v
);

std::vector<torch::Tensor> chunked_cross_entropy_cuda(
    torch::Tensor hidden_states,
    torch::Tensor lm_head_weight,
    torch::Tensor labels
);

torch::Tensor fused_lora_forward_cuda(
    torch::Tensor x,
    torch::Tensor W_base,
    torch::Tensor A,
    torch::Tensor B,
    float scaling
);

void paged_kv_append_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor seq_lens,
    torch::Tensor key_states,
    torch::Tensor value_states
);

void barqtrain_memory_set_enabled(bool enabled) {
    auto& state = memory_tracker_state();
    std::lock_guard<std::mutex> lock(memory_tracker_mutex());
    state.enabled = enabled;
}

void barqtrain_memory_reset(bool reset_peak = true) {
    auto& state = memory_tracker_state();
    std::lock_guard<std::mutex> lock(memory_tracker_mutex());
    clear_bucket(&state.resident_model, reset_peak);
    clear_bucket(&state.kv_cache, reset_peak);
    clear_bucket(&state.decode_temp, reset_peak);
    if (reset_peak) {
        state.training_peak_bytes = 0;
        state.inference_peak_bytes = 0;
    }
}

void barqtrain_memory_set_bucket_bytes(const std::string& bucket, int64_t current_bytes) {
    auto& state = memory_tracker_state();
    std::lock_guard<std::mutex> lock(memory_tracker_mutex());
    if (!state.enabled) {
        return;
    }

    auto* counters = resolve_bucket(state, bucket);
    counters->current_bytes = std::max<int64_t>(current_bytes, 0);
    counters->peak_bytes = std::max(counters->peak_bytes, counters->current_bytes);
}

void barqtrain_memory_record_peak(const std::string& mode, int64_t peak_bytes) {
    auto& state = memory_tracker_state();
    std::lock_guard<std::mutex> lock(memory_tracker_mutex());
    if (!state.enabled) {
        return;
    }

    const int64_t clamped_peak = std::max<int64_t>(peak_bytes, 0);
    if (mode == "training") {
        state.training_peak_bytes = std::max(state.training_peak_bytes, clamped_peak);
        return;
    }
    if (mode == "inference") {
        state.inference_peak_bytes = std::max(state.inference_peak_bytes, clamped_peak);
        return;
    }

    throw std::invalid_argument("Unknown BarqTrain memory peak mode: " + mode);
}

py::dict barqtrain_memory_snapshot() {
    auto& state = memory_tracker_state();
    std::lock_guard<std::mutex> lock(memory_tracker_mutex());

    py::dict snapshot;
    snapshot["enabled"] = state.enabled;
    snapshot["resident_model_current_bytes"] = state.resident_model.current_bytes;
    snapshot["resident_model_peak_bytes"] = state.resident_model.peak_bytes;
    snapshot["kv_cache_current_bytes"] = state.kv_cache.current_bytes;
    snapshot["kv_cache_peak_bytes"] = state.kv_cache.peak_bytes;
    snapshot["decode_temp_current_bytes"] = state.decode_temp.current_bytes;
    snapshot["decode_temp_peak_bytes"] = state.decode_temp.peak_bytes;
    snapshot["training_peak_bytes"] = state.training_peak_bytes;
    snapshot["inference_peak_bytes"] = state.inference_peak_bytes;
    return snapshot;
}

// Python module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "BarqTrain CUDA kernels for high-performance LLM fine-tuning";

    // RMSNorm functions
    m.def("fused_rmsnorm", &fused_rmsnorm_cuda,
          "Fused RMSNorm forward pass (CUDA)");
    m.def("fused_rmsnorm_backward", &fused_rmsnorm_backward_cuda,
          "Fused RMSNorm backward pass (CUDA)");

    // Flash Attention functions
    m.def("flash_attention", &flash_attention_cuda,
          "FlashAttention forward/backward with fused RoPE (CUDA)");

    // Chunked Cross-Entropy functions
    m.def("chunked_cross_entropy", &chunked_cross_entropy_cuda,
          "Chunked cross-entropy loss avoiding logit materialization (CUDA)");

    // LoRA functions
    m.def("fused_lora_forward", &fused_lora_forward_cuda,
          "Fused LoRA forward: x @ W_base + (x @ A) @ B (CUDA)");

    // KV-cache functions
    m.def("paged_kv_append_", &paged_kv_append_cuda,
          "Append key/value states into the paged KV cache (CUDA)");

    // Native memory accounting hooks
    m.def("barqtrain_memory_set_enabled", &barqtrain_memory_set_enabled,
          "Enable or disable BarqTrain detailed CUDA memory profiling");
    m.def("barqtrain_memory_reset", &barqtrain_memory_reset,
          py::arg("reset_peak") = true,
          "Reset BarqTrain native memory accounting state");
    m.def("barqtrain_memory_set_bucket_bytes", &barqtrain_memory_set_bucket_bytes,
          "Record current bytes for a BarqTrain memory bucket");
    m.def("barqtrain_memory_record_peak", &barqtrain_memory_record_peak,
          "Record a training or inference peak VRAM measurement");
    m.def("barqtrain_memory_snapshot", &barqtrain_memory_snapshot,
          "Return the current BarqTrain memory accounting snapshot");
}
