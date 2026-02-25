/**
 * BarqTrain Fused LoRA GEMM Kernel
 *
 * output = x @ W_base + (x @ A) @ B   (fused, single-pass)
 *
 * Simplified for Colab compilation (no tiled variant — avoids OOM-kill
 * from nvcc -O2 trying to optimise deep nested loops over large tiles).
 * File-scope LORA_TILE constants kept for backward compatibility.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// File-scope constants visible to both kernel and AT_DISPATCH launcher lambda
static constexpr int LORA_TILE_OUT = 16;
static constexpr int LORA_TILE_IN = 16;

// ---------------------------------------------------------------------------
// Single fused kernel: one block per batch element, one thread per out_feat.
// Shared memory holds the x-row so threads don't re-fetch it.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void
fused_lora_forward_kernel(const T *__restrict__ x,      // [B, I]
                          const T *__restrict__ W_base, // [O, I]
                          const T *__restrict__ A,      // [R, I]
                          const T *__restrict__ B,      // [O, R]
                          T *__restrict__ output,       // [B, O]
                          int batch_size, int in_features, int out_features,
                          int rank, float scaling) {
  int b = blockIdx.x;
  int o = threadIdx.x;

  if (b >= batch_size || o >= out_features)
    return;

  extern __shared__ float sx[]; // in_features floats: holds x[b, :]

  // Cooperatively load x row into shared memory
  for (int i = o; i < in_features; i += blockDim.x) {
    int xi = b * in_features + i;
    if constexpr (std::is_same<T, half>::value)
      sx[i] = __half2float(x[xi]);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      sx[i] = __bfloat162float(x[xi]);
    else
      sx[i] = x[xi];
  }
  __syncthreads();

  // Base GEMM: dot(x, W_base[o, :])
  float base_out = 0.0f;
  for (int i = 0; i < in_features; i++) {
    float w;
    int wi = o * in_features + i;
    if constexpr (std::is_same<T, half>::value)
      w = __half2float(W_base[wi]);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      w = __bfloat162float(W_base[wi]);
    else
      w = W_base[wi];
    base_out += sx[i] * w;
  }

  // LoRA: sum_r( dot(x, A[r,:]) * B[o,r] )
  float lora_out = 0.0f;
  for (int r = 0; r < rank; r++) {
    float xa = 0.0f;
    for (int i = 0; i < in_features; i++) {
      float a;
      int ai = r * in_features + i;
      if constexpr (std::is_same<T, half>::value)
        a = __half2float(A[ai]);
      else if constexpr (std::is_same<T, __nv_bfloat16>::value)
        a = __bfloat162float(A[ai]);
      else
        a = A[ai];
      xa += sx[i] * a;
    }
    float bv;
    int bi = o * rank + r;
    if constexpr (std::is_same<T, half>::value)
      bv = __half2float(B[bi]);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      bv = __bfloat162float(B[bi]);
    else
      bv = B[bi];
    lora_out += xa * bv;
  }

  float val = base_out + lora_out * scaling;
  int oi = b * out_features + o;
  if constexpr (std::is_same<T, half>::value)
    output[oi] = __float2half(val);
  else if constexpr (std::is_same<T, __nv_bfloat16>::value)
    output[oi] = __float2bfloat16(val);
  else
    output[oi] = val;
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
torch::Tensor fused_lora_forward_cuda(torch::Tensor x, torch::Tensor W_base,
                                      torch::Tensor A, torch::Tensor B,
                                      float scaling) {
  int B_sz = x.size(0);
  int I = x.size(1);
  int O = W_base.size(0);
  int R = A.size(0);

  TORCH_CHECK(O <= 1024, "fused_lora_forward_cuda: out_features (", O,
              ") > 1024 max threads/block");

  auto output = torch::empty({B_sz, O}, x.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "fused_lora_forward", ([&] {
        using T = scalar_t;
        int threads = O;
        int smem = I * sizeof(float);
        fused_lora_forward_kernel<T><<<B_sz, threads, smem>>>(
            x.data_ptr<T>(), W_base.data_ptr<T>(), A.data_ptr<T>(),
            B.data_ptr<T>(), output.data_ptr<T>(), B_sz, I, O, R, scaling);
      }));

  return output;
}
