/**
 * BarqTrain CUDA Extension - Python Bindings
 *
 * This file provides PyBind11 bindings for BarqTrain's CUDA kernels.
 */

#include <torch/extension.h>

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
}
