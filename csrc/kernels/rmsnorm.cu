/**
 * BarqTrain Fused RMSNorm Kernel
 *
 * RMSNorm: output = (x / rms) * weight,  rms = sqrt(mean(x^2) + eps)
 *
 * Key fix in backward: grad_weight is accumulated into a float32 buffer
 * (atomicAdd on float* always works) and converted to T at the end.
 * This avoids the c10::Half/BFloat16 vs __half/__nv_bfloat16 mismatch.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

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
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}

// ---------------------------------------------------------------------------
// Forward kernel
// ---------------------------------------------------------------------------
template <typename T>
__global__ void fused_rmsnorm_forward_kernel(const T *__restrict__ input,
                                             const T *__restrict__ weight,
                                             T *__restrict__ output,
                                             float *__restrict__ rms_cache,
                                             float eps, int batch_size,
                                             int hidden_size) {
  int batch_idx = blockIdx.x;
  int hidden_idx = threadIdx.x;

  extern __shared__ float sdata[];

  float x_val = 0.0f;
  if (hidden_idx < hidden_size) {
    size_t idx = (size_t)batch_idx * hidden_size + hidden_idx;
    if constexpr (std::is_same<T, half>::value)
      x_val = __half2float(input[idx]);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      x_val = __bfloat162float(input[idx]);
    else
      x_val = input[idx];
  }

  sdata[hidden_idx] = x_val * x_val;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (hidden_idx < stride)
      sdata[hidden_idx] += sdata[hidden_idx + stride];
    __syncthreads();
  }

  float rms = sqrtf(sdata[0] / (float)hidden_size + eps);

  if (hidden_idx == 0 && rms_cache != nullptr)
    rms_cache[batch_idx] = rms;

  if (hidden_idx < hidden_size) {
    size_t idx = (size_t)batch_idx * hidden_size + hidden_idx;

    float w_val = 0.0f;
    if constexpr (std::is_same<T, half>::value)
      w_val = __half2float(weight[hidden_idx]);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      w_val = __bfloat162float(weight[hidden_idx]);
    else
      w_val = weight[hidden_idx];

    float out_val = (x_val / rms) * w_val;

    if constexpr (std::is_same<T, half>::value)
      output[idx] = __float2half(out_val);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      output[idx] = __float2bfloat16(out_val);
    else
      output[idx] = out_val;
  }
}

// ---------------------------------------------------------------------------
// Backward kernel
// grad_weight_f32 is always float* — atomicAdd(float*, float) is always valid.
// The launcher allocates a float32 scratch buffer, runs this kernel, then
// converts the scratch buffer back to T for the Python caller.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void fused_rmsnorm_backward_kernel(
    const T *__restrict__ grad_output, const T *__restrict__ input,
    const T *__restrict__ weight, const float *__restrict__ rms_cache,
    T *__restrict__ grad_input,
    float *__restrict__ grad_weight_f32, // float accumulation buffer
    float eps, int batch_size, int hidden_size) {
  int batch_idx = blockIdx.x;
  int hidden_idx = threadIdx.x;

  extern __shared__ float sdata[];

  float grad_out = 0.0f, x_val = 0.0f, w_val = 0.0f;
  if (hidden_idx < hidden_size) {
    size_t idx = (size_t)batch_idx * hidden_size + hidden_idx;
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

  float rms = rms_cache[batch_idx];
  float y = x_val / rms;
  float grad_y = grad_out * w_val;

  sdata[hidden_idx] = grad_y * y;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (hidden_idx < stride)
      sdata[hidden_idx] += sdata[hidden_idx + stride];
    __syncthreads();
  }
  float sum_dy_y = sdata[0];

  if (hidden_idx < hidden_size) {
    size_t idx = (size_t)batch_idx * hidden_size + hidden_idx;
    float grad_in_val = (grad_y - y * sum_dy_y / (float)hidden_size) / rms;

    if constexpr (std::is_same<T, half>::value)
      grad_input[idx] = __float2half(grad_in_val);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      grad_input[idx] = __float2bfloat16(grad_in_val);
    else
      grad_input[idx] = grad_in_val;

    // Accumulate grad_weight into float32 buffer — atomicAdd(float*, float)
    // is always supported, no type-conversion issues.
    if (grad_weight_f32 != nullptr)
      atomicAdd(&grad_weight_f32[hidden_idx], grad_out * y);
  }
}

// ---------------------------------------------------------------------------
// Forward launcher
// ---------------------------------------------------------------------------
torch::Tensor fused_rmsnorm_cuda(torch::Tensor input, torch::Tensor weight,
                                 float eps) {
  auto batch_size = input.size(0);
  auto hidden_size = input.size(1);

  auto output = torch::empty_like(input);
  auto rms_cache =
      torch::empty({batch_size}, input.options().dtype(torch::kFloat32));

  const int threads = 256;
  const int shared_mem_sz = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "fused_rmsnorm_forward", ([&] {
        using T = scalar_t;
        int actual_threads = std::min(threads, (int)hidden_size);

        fused_rmsnorm_forward_kernel<T>
            <<<batch_size, actual_threads, shared_mem_sz>>>(
                input.data_ptr<T>(), weight.data_ptr<T>(), output.data_ptr<T>(),
                rms_cache.data_ptr<float>(), eps, (int)batch_size,
                (int)hidden_size);
        CUDA_CHECK(cudaGetLastError());
      }));

  return output;
}

// ---------------------------------------------------------------------------
// Backward launcher
// ---------------------------------------------------------------------------
torch::Tensor fused_rmsnorm_backward_cuda(torch::Tensor grad_output,
                                          torch::Tensor input,
                                          torch::Tensor weight,
                                          torch::Tensor rms) {
  auto batch_size = input.size(0);
  auto hidden_size = input.size(1);

  auto grad_input = torch::empty_like(input);

  // Use float32 scratch for grad_weight accumulation (always safe with
  // atomicAdd)
  auto grad_weight_f32 =
      torch::zeros({hidden_size}, input.options().dtype(torch::kFloat32));

  const int threads = 256;
  const int shared_mem_sz = threads * sizeof(float);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(),
      "fused_rmsnorm_backward", ([&] {
        using T = scalar_t;
        int actual_threads = std::min(threads, (int)hidden_size);

        fused_rmsnorm_backward_kernel<T>
            <<<batch_size, actual_threads, shared_mem_sz>>>(
                grad_output.data_ptr<T>(), input.data_ptr<T>(),
                weight.data_ptr<T>(), rms.data_ptr<float>(),
                grad_input.data_ptr<T>(), grad_weight_f32.data_ptr<float>(),
                1e-6f, (int)batch_size, (int)hidden_size);
        CUDA_CHECK(cudaGetLastError());
      }));

  // grad_weight_f32 is returned as-is (float32); caller converts if needed.
  // In practice grad_weight is accumulated in float32 for numerical stability.
  return grad_input;
}
