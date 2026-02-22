/**
 * BarqTrain Fused RMSNorm Kernel
 *
 * Implements a fused RMSNorm with single read, shared memory reduction,
 * and single write for optimal memory bandwidth utilization.
 *
 * RMSNorm Formula:
 *   rms = sqrt(mean(x^2) + eps)
 *   output = (x / rms) * weight
 *
 * This implementation fuses the entire operation into a single kernel
 * to minimize HBM reads/writes.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

// Fused RMSNorm Forward Kernel
template <typename T>
__global__ void fused_rmsnorm_forward_kernel(const T *__restrict__ input,
                                             const T *__restrict__ weight,
                                             T *__restrict__ output,
                                             float *__restrict__ rms_cache,
                                             float eps, int batch_size,
                                             int hidden_size) {
  // Each thread block processes one row (one sequence in the batch)
  int batch_idx = blockIdx.x;
  int hidden_idx = threadIdx.x;

  // Shared memory for reduction within the block
  extern __shared__ float sdata[];

  // Load input value and compute square
  float x_val = 0.0f;
  if (hidden_idx < hidden_size) {
    size_t idx = batch_idx * hidden_size + hidden_idx;
    if constexpr (std::is_same<T, half>::value) {
      x_val = __half2float(input[idx]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      x_val = __bfloat162float(input[idx]);
    } else {
      x_val = input[idx];
    }
  }

  // Compute sum of squares in shared memory
  sdata[hidden_idx] = x_val * x_val;
  __syncthreads();

  // Reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (hidden_idx < stride) {
      sdata[hidden_idx] += sdata[hidden_idx + stride];
    }
    __syncthreads();
  }

  // Compute RMS
  float mean_square = sdata[0] / (float)hidden_size;
  float rms = sqrtf(mean_square + eps);

  // Cache RMS for backward pass (optional, can be recomputed)
  if (hidden_idx == 0 && rms_cache != nullptr) {
    rms_cache[batch_idx] = rms;
  }

  // Compute and write output
  if (hidden_idx < hidden_size) {
    size_t idx = batch_idx * hidden_size + hidden_idx;

    // Load weight
    float w_val = 0.0f;
    if constexpr (std::is_same<T, half>::value) {
      w_val = __half2float(weight[hidden_idx]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      w_val = __bfloat162float(weight[hidden_idx]);
    } else {
      w_val = weight[hidden_idx];
    }

    // RMSNorm: output = (x / rms) * weight
    float out_val = (x_val / rms) * w_val;

    if constexpr (std::is_same<T, half>::value) {
      output[idx] = __float2half(out_val);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      output[idx] = __float2bfloat16(out_val);
    } else {
      output[idx] = out_val;
    }
  }
}

// Fused RMSNorm Backward Kernel
template <typename T>
__global__ void fused_rmsnorm_backward_kernel(
    const T *__restrict__ grad_output, const T *__restrict__ input,
    const T *__restrict__ weight, const float *__restrict__ rms_cache,
    T *__restrict__ grad_input, T *__restrict__ grad_weight, float eps,
    int batch_size, int hidden_size) {
  int batch_idx = blockIdx.x;
  int hidden_idx = threadIdx.x;

  extern __shared__ float sdata[];

  // Load values
  float grad_out = 0.0f;
  float x_val = 0.0f;
  float w_val = 0.0f;

  if (hidden_idx < hidden_size) {
    size_t idx = batch_idx * hidden_size + hidden_idx;

    if constexpr (std::is_same<T, half>::value) {
      grad_out = __half2float(grad_output[idx]);
      x_val = __half2float(input[idx]);
      w_val = __half2float(weight[hidden_idx]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      grad_out = __bfloat162float(grad_output[idx]);
      x_val = __bfloat162float(input[idx]);
      w_val = __bfloat162float(weight[hidden_idx]);
    } else {
      grad_out = grad_output[idx];
      x_val = input[idx];
      w_val = weight[hidden_idx];
    }
  }

  // Get RMS (cached or recompute)
  float rms = rms_cache[batch_idx];

  // Compute intermediate values for gradient
  float y = x_val / rms;
  float grad_y = grad_out * w_val; // Gradient before scaling by weight

  // Reduction for sum term in gradient
  sdata[hidden_idx] = grad_y * y;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (hidden_idx < stride) {
      sdata[hidden_idx] += sdata[hidden_idx + stride];
    }
    __syncthreads();
  }

  float sum_dy_y = sdata[0];

  // Compute gradients
  if (hidden_idx < hidden_size) {
    size_t idx = batch_idx * hidden_size + hidden_idx;

    // grad_input = (grad_y - y * mean(dy * y)) / rms
    float grad_in_val = (grad_y - y * sum_dy_y / (float)hidden_size) / rms;

    if constexpr (std::is_same<T, half>::value) {
      grad_input[idx] = __float2half(grad_in_val);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      grad_input[idx] = __float2bfloat16(grad_in_val);
    } else {
      grad_input[idx] = grad_in_val;
    }

    // grad_weight = sum over batches of (grad_output * x / rms)
    // atomicAdd requires pointer and value to have the same type,
    // so we cast the float grad_w to T before calling atomicAdd.
    if (grad_weight != nullptr) {
      float grad_w = grad_out * y;
      if constexpr (std::is_same<T, half>::value) {
        atomicAdd(&grad_weight[hidden_idx], __float2half(grad_w));
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        atomicAdd(&grad_weight[hidden_idx], __float2bfloat16(grad_w));
      } else {
        atomicAdd(&grad_weight[hidden_idx], grad_w);
      }
    }
  }
}

// Forward launcher
torch::Tensor fused_rmsnorm_cuda(torch::Tensor input, torch::Tensor weight,
                                 float eps) {
  auto batch_size = input.size(0);
  auto hidden_size = input.size(1);

  // Allocate output tensor
  auto output = torch::empty_like(input);

  // Allocate RMS cache for backward pass
  auto rms_cache =
      torch::empty({batch_size}, input.options().dtype(torch::kFloat32));

  // Launch kernel
  const int threads = 256;
  const int shared_mem_size = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "fused_rmsnorm_forward", ([&] {
        using T = scalar_t;
        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

        // Adjust threads if hidden_size is smaller
        int actual_threads = std::min(threads, (int)hidden_size);

        fused_rmsnorm_forward_kernel<T>
            <<<batch_size, actual_threads, shared_mem_size>>>(
                input.data_ptr<T>(), weight.data_ptr<T>(), output.data_ptr<T>(),
                rms_cache.data_ptr<float>(), eps, batch_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
      }));

  // Attach RMS cache as output metadata for backward pass
  // In a real implementation, you'd use a custom autograd function

  return output;
}

// Backward launcher
torch::Tensor fused_rmsnorm_backward_cuda(torch::Tensor grad_output,
                                          torch::Tensor input,
                                          torch::Tensor weight,
                                          torch::Tensor rms) {
  auto batch_size = input.size(0);
  auto hidden_size = input.size(1);

  // Allocate gradient tensors
  auto grad_input = torch::empty_like(input);
  auto grad_weight = torch::zeros_like(weight);

  // Launch backward kernel
  const int threads = 256;
  const int shared_mem_size = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "fused_rmsnorm_backward", ([&] {
        using T = scalar_t;
        int actual_threads = std::min(threads, (int)hidden_size);

        fused_rmsnorm_backward_kernel<T>
            <<<batch_size, actual_threads, shared_mem_size>>>(
                grad_output.data_ptr<T>(), input.data_ptr<T>(),
                weight.data_ptr<T>(), rms.data_ptr<float>(),
                grad_input.data_ptr<T>(), grad_weight.data_ptr<T>(),
                1e-6f, // eps
                batch_size, hidden_size);
        CUDA_CHECK(cudaGetLastError());
      }));

  return grad_input; // In practice, return tuple of (grad_input, grad_weight)
}
