/**
 * BarqTrain Fused LoRA GEMM Kernel
 *
 * output = x @ W_base + (x @ A) @ B   (fused, single-pass)
 *
 * Key fix: LORA_TILE_OUT / LORA_TILE_IN are file-scope constexpr constants
 * so they are visible inside the AT_DISPATCH lambda in the launcher.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// File-scope tile constants — visible to both kernels AND the launcher lambda
static constexpr int LORA_TILE_OUT = 16;
static constexpr int LORA_TILE_IN = 16;

// ---------------------------------------------------------------------------
// Simple kernel (small matrices, e.g. during unit tests)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void fused_lora_forward_kernel(
    const T *__restrict__ x,      // [batch_size, in_features]
    const T *__restrict__ W_base, // [out_features, in_features]
    const T *__restrict__ A,      // [rank, in_features]
    const T *__restrict__ B,      // [out_features, rank]
    T *__restrict__ output,       // [batch_size, out_features]
    int batch_size, int in_features, int out_features, int rank,
    float scaling) {
  int batch_idx = blockIdx.x;
  int out_idx = threadIdx.x;

  if (batch_idx >= batch_size || out_idx >= out_features)
    return;

  // Shared memory holds one x-row (size = in_features)
  extern __shared__ float sdata[]; // allocated as out_features floats

  float x_val = 0.0f;
  if (out_idx < in_features) {
    int xi = batch_idx * in_features + out_idx;
    if constexpr (std::is_same<T, half>::value)
      x_val = __half2float(x[xi]);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      x_val = __bfloat162float(x[xi]);
    else
      x_val = x[xi];
  }
  sdata[out_idx] = x_val;
  __syncthreads();

  // Base GEMM: dot(x, W_base[out_idx, :])
  float base_out = 0.0f;
  for (int i = 0; i < in_features; i++) {
    float w = 0.0f;
    int wi = out_idx * in_features + i;
    if constexpr (std::is_same<T, half>::value)
      w = __half2float(W_base[wi]);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      w = __bfloat162float(W_base[wi]);
    else
      w = W_base[wi];
    base_out += sdata[i] * w;
  }

  // LoRA: (x @ A) @ B[out_idx, :]
  float lora_out = 0.0f;
  for (int r = 0; r < rank; r++) {
    float xa = 0.0f;
    for (int i = 0; i < in_features; i++) {
      float a = 0.0f;
      int ai = r * in_features + i;
      if constexpr (std::is_same<T, half>::value)
        a = __half2float(A[ai]);
      else if constexpr (std::is_same<T, __nv_bfloat16>::value)
        a = __bfloat162float(A[ai]);
      else
        a = A[ai];
      xa += sdata[i] * a;
    }
    float b = 0.0f;
    int bi = out_idx * rank + r;
    if constexpr (std::is_same<T, half>::value)
      b = __half2float(B[bi]);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      b = __bfloat162float(B[bi]);
    else
      b = B[bi];
    lora_out += xa * b;
  }

  float val = base_out + lora_out * scaling;
  int oi = batch_idx * out_features + out_idx;
  if constexpr (std::is_same<T, half>::value)
    output[oi] = __float2half(val);
  else if constexpr (std::is_same<T, __nv_bfloat16>::value)
    output[oi] = __float2bfloat16(val);
  else
    output[oi] = val;
}

// ---------------------------------------------------------------------------
// Tiled kernel (larger matrices)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void fused_lora_forward_tiled_kernel(
    const T *__restrict__ x, const T *__restrict__ W_base,
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ output,
    int batch_size, int in_features, int out_features, int rank,
    float scaling) {
  int batch_idx = blockIdx.x;
  int tile_out_idx = threadIdx.x; // 0..LORA_TILE_OUT-1
  int tile_in_idx = threadIdx.y;  // 0..LORA_TILE_IN-1

  // This thread's global output column — fixed for the lifetime of this thread
  const int cur_out = blockIdx.y * LORA_TILE_OUT + tile_out_idx;

  __shared__ float x_sh[LORA_TILE_IN];
  __shared__ float W_sh[LORA_TILE_OUT * LORA_TILE_IN];

  float base_acc = 0.0f;
  float lora_acc = 0.0f;

  // --- Base GEMM tiled ---
  for (int in_tile = 0; in_tile < in_features; in_tile += LORA_TILE_IN) {
    int in_idx = in_tile + tile_in_idx;

    // Load x tile (one thread per in_features element)
    x_sh[tile_in_idx] = 0.0f;
    if (batch_idx < batch_size && in_idx < in_features) {
      int xp = batch_idx * in_features + in_idx;
      if constexpr (std::is_same<T, half>::value)
        x_sh[tile_in_idx] = __half2float(x[xp]);
      else if constexpr (std::is_same<T, __nv_bfloat16>::value)
        x_sh[tile_in_idx] = __bfloat162float(x[xp]);
      else
        x_sh[tile_in_idx] = x[xp];
    }

    // Load W tile
    W_sh[tile_out_idx * LORA_TILE_IN + tile_in_idx] = 0.0f;
    if (cur_out < out_features && in_idx < in_features) {
      int wp = cur_out * in_features + in_idx;
      if constexpr (std::is_same<T, half>::value)
        W_sh[tile_out_idx * LORA_TILE_IN + tile_in_idx] =
            __half2float(W_base[wp]);
      else if constexpr (std::is_same<T, __nv_bfloat16>::value)
        W_sh[tile_out_idx * LORA_TILE_IN + tile_in_idx] =
            __bfloat162float(W_base[wp]);
      else
        W_sh[tile_out_idx * LORA_TILE_IN + tile_in_idx] = W_base[wp];
    }
    __syncthreads();

    if (cur_out < out_features) {
      for (int t = 0; t < LORA_TILE_IN && in_tile + t < in_features; t++)
        base_acc += x_sh[t] * W_sh[tile_out_idx * LORA_TILE_IN + t];
    }
    __syncthreads();
  }

  // --- LoRA contribution ---
  for (int r = 0; r < rank; r++) {
    float xa = 0.0f;

    for (int in_tile = 0; in_tile < in_features; in_tile += LORA_TILE_IN) {
      int in_idx = in_tile + tile_in_idx;

      x_sh[tile_in_idx] = 0.0f;
      if (batch_idx < batch_size && in_idx < in_features) {
        int xp = batch_idx * in_features + in_idx;
        if constexpr (std::is_same<T, half>::value)
          x_sh[tile_in_idx] = __half2float(x[xp]);
        else if constexpr (std::is_same<T, __nv_bfloat16>::value)
          x_sh[tile_in_idx] = __bfloat162float(x[xp]);
        else
          x_sh[tile_in_idx] = x[xp];
      }
      __syncthreads();

      // Only thread 0 of the in-tile dimension accumulates xa
      if (tile_in_idx == 0 && in_idx < in_features) {
        int ap = r * in_features + in_idx;
        float a_val = 0.0f;
        if constexpr (std::is_same<T, half>::value)
          a_val = __half2float(A[ap]);
        else if constexpr (std::is_same<T, __nv_bfloat16>::value)
          a_val = __bfloat162float(A[ap]);
        else
          a_val = A[ap];
        for (int t = 0; t < LORA_TILE_IN && in_tile + t < in_features; t++)
          xa += x_sh[t] * a_val;
      }
      __syncthreads();
    }

    if (tile_in_idx == 0 && cur_out < out_features) {
      int bp = cur_out * rank + r;
      float b_val = 0.0f;
      if constexpr (std::is_same<T, half>::value)
        b_val = __half2float(B[bp]);
      else if constexpr (std::is_same<T, __nv_bfloat16>::value)
        b_val = __bfloat162float(B[bp]);
      else
        b_val = B[bp];
      lora_acc += xa * b_val;
    }
  }

  // --- Write output ---
  if (batch_idx < batch_size && cur_out < out_features) {
    int op = batch_idx * out_features + cur_out;
    float val = base_acc + lora_acc * scaling;
    if constexpr (std::is_same<T, half>::value)
      output[op] = __float2half(val);
    else if constexpr (std::is_same<T, __nv_bfloat16>::value)
      output[op] = __float2bfloat16(val);
    else
      output[op] = val;
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
torch::Tensor fused_lora_forward_cuda(torch::Tensor x, torch::Tensor W_base,
                                      torch::Tensor A, torch::Tensor B,
                                      float scaling) {
  int batch_size = x.size(0);
  int in_features = x.size(1);
  int out_features = W_base.size(0);
  int rank = A.size(0);

  auto output = torch::empty({batch_size, out_features}, x.options());
  bool use_tiled = (out_features > 64 || in_features > 64);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "fused_lora_forward", ([&] {
        using T = scalar_t;

        if (use_tiled) {
          // LORA_TILE_OUT / LORA_TILE_IN are file-scope — visible here
          dim3 blocks(batch_size,
                      (out_features + LORA_TILE_OUT - 1) / LORA_TILE_OUT);
          dim3 threads(LORA_TILE_OUT, LORA_TILE_IN);
          int smem =
              (LORA_TILE_OUT * LORA_TILE_IN + LORA_TILE_IN) * sizeof(float);

          fused_lora_forward_tiled_kernel<T><<<blocks, threads, smem>>>(
              x.data_ptr<T>(), W_base.data_ptr<T>(), A.data_ptr<T>(),
              B.data_ptr<T>(), output.data_ptr<T>(), batch_size, in_features,
              out_features, rank, scaling);
        } else {
          dim3 blocks(batch_size);
          dim3 threads(out_features);
          int smem = out_features * sizeof(float);

          fused_lora_forward_kernel<T><<<blocks, threads, smem>>>(
              x.data_ptr<T>(), W_base.data_ptr<T>(), A.data_ptr<T>(),
              B.data_ptr<T>(), output.data_ptr<T>(), batch_size, in_features,
              out_features, rank, scaling);
        }
      }));

  return output;
}
