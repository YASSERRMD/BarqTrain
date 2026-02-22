/**
 * BarqTrain Fused LoRA GEMM Kernel
 *
 * Implements a fused kernel for computing xW_base + x(AB) in a single
 * batched GEMM operation, eliminating the separate LoRA adapter overhead.
 *
 * Formula:
 *   output = x @ W_base + x @ (A @ B)
 *          = x @ W_base + (x @ A) @ B
 *
 * This kernel fuses the base weight multiplication with the LoRA adapter
 * computation, reducing memory reads/writes and improving throughput.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// LoRA forward kernel: output = x @ W_base + (x @ A) @ B
template <typename T>
__global__ void fused_lora_forward_kernel(
    const T *__restrict__ x,      // [batch_size, in_features]
    const T *__restrict__ W_base, // [out_features, in_features]
    const T *__restrict__ A,      // [rank, in_features]
    const T *__restrict__ B,      // [out_features, rank]
    T *__restrict__ output,       // [batch_size, out_features]
    int batch_size, int in_features, int out_features, int rank,
    float scaling) {
  // Each thread block processes one row of the batch
  int batch_idx = blockIdx.x;
  int out_idx = threadIdx.x;

  if (batch_idx >= batch_size || out_idx >= out_features)
    return;

  // Shared memory for intermediate results
  extern __shared__ float sdata[];

  // Load x row into shared memory
  float *x_shared = &sdata[out_idx];
  float x_val = 0.0f;

  if (out_idx < in_features) {
    int x_idx = batch_idx * in_features + out_idx;
    if constexpr (std::is_same<T, half>::value) {
      x_val = __half2float(x[x_idx]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      x_val = __bfloat162float(x[x_idx]);
    } else {
      x_val = x[x_idx];
    }
  }
  x_shared[out_idx] = x_val;

  __syncthreads();

  // Compute base output: x @ W_base[out_idx, :]
  float base_output = 0.0f;

  // Loop over input features (use loop unrolling and vectorization in
  // production)
  for (int in_idx = 0; in_idx < in_features; in_idx++) {
    float w_val = 0.0f;
    int w_idx = out_idx * in_features + in_idx;

    if constexpr (std::is_same<T, half>::value) {
      w_val = __half2float(W_base[w_idx]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      w_val = __bfloat162float(W_base[w_idx]);
    } else {
      w_val = W_base[w_idx];
    }

    base_output += x_shared[in_idx] * w_val;
  }

  // Compute LoRA adapter output: (x @ A) @ B[out_idx, :]
  float lora_output = 0.0f;

  // First compute x @ A (rank intermediate)
  for (int r = 0; r < rank; r++) {
    float xa_result = 0.0f;

    // x @ A[r, :]
    for (int in_idx = 0; in_idx < in_features; in_idx++) {
      float a_val = 0.0f;
      int a_idx = r * in_features + in_idx;

      if constexpr (std::is_same<T, half>::value) {
        a_val = __half2float(A[a_idx]);
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        a_val = __bfloat162float(A[a_idx]);
      } else {
        a_val = A[a_idx];
      }

      xa_result += x_shared[in_idx] * a_val;
    }

    // (x @ A) @ B[out_idx, r]
    float b_val = 0.0f;
    int b_idx = out_idx * rank + r;

    if constexpr (std::is_same<T, half>::value) {
      b_val = __half2float(B[b_idx]);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      b_val = __bfloat162float(B[b_idx]);
    } else {
      b_val = B[b_idx];
    }

    lora_output += xa_result * b_val;
  }

  // Apply LoRA scaling and combine
  float final_output = base_output + lora_output * scaling;

  // Write result
  int out_base = batch_idx * out_features + out_idx;
  if constexpr (std::is_same<T, half>::value) {
    output[out_base] = __float2half(final_output);
  } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    output[out_base] = __float2bfloat16(final_output);
  } else {
    output[out_base] = final_output;
  }
}

// Optimized version using shared memory tiling
template <typename T>
__global__ void fused_lora_forward_tiled_kernel(
    const T *__restrict__ x, const T *__restrict__ W_base,
    const T *__restrict__ A, const T *__restrict__ B, T *__restrict__ output,
    int batch_size, int in_features, int out_features, int rank,
    float scaling) {
  // Thread block organization
  int batch_idx = blockIdx.x;
  int tile_out_idx = threadIdx.x;
  int tile_in_idx = threadIdx.y;

  // Tile dimensions
  const int TILE_OUT = 16;
  const int TILE_IN = 16;

  // Shared memory tiles
  __shared__ float x_shared[TILE_IN];
  __shared__ float W_shared[TILE_OUT * TILE_IN];
  __shared__ float A_shared[TILE_IN]; // For one rank element

  float base_accum = 0.0f;
  float lora_accum = 0.0f;

  // Tile over input features
  for (int in_tile = 0; in_tile < in_features; in_tile += TILE_IN) {
    // Load x tile
    int in_idx = in_tile + tile_in_idx;
    if (batch_idx < batch_size && in_idx < in_features) {
      int x_pos = batch_idx * in_features + in_idx;
      if constexpr (std::is_same<T, half>::value) {
        x_shared[tile_in_idx] = __half2float(x[x_pos]);
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        x_shared[tile_in_idx] = __bfloat162float(x[x_pos]);
      } else {
        x_shared[tile_in_idx] = x[x_pos];
      }
    }
    __syncthreads();

    // Load W_base tile
    int out_idx = blockIdx.y * TILE_OUT + tile_out_idx;
    if (out_idx < out_features && in_idx < in_features) {
      int w_pos = out_idx * in_features + in_idx;
      if constexpr (std::is_same<T, half>::value) {
        W_shared[tile_out_idx * TILE_IN + tile_in_idx] =
            __half2float(W_base[w_pos]);
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        W_shared[tile_out_idx * TILE_IN + tile_in_idx] =
            __bfloat162float(W_base[w_pos]);
      } else {
        W_shared[tile_out_idx * TILE_IN + tile_in_idx] = W_base[w_pos];
      }
    }
    __syncthreads();

    // Compute base matmul for this tile
    if (out_idx < out_features) {
      for (int t = 0; t < TILE_IN && in_tile + t < in_features; t++) {
        base_accum += x_shared[t] * W_shared[tile_out_idx * TILE_IN + t];
      }
    }
    __syncthreads();
  }

  // Compute LoRA contribution
  // Compute this thread's output index once — used throughout this section
  int current_out_idx = blockIdx.y * TILE_OUT + tile_out_idx;

  // For each rank element
  for (int r = 0; r < rank; r++) {
    float xa_result = 0.0f;

    // x @ A[r, :]
    for (int in_tile = 0; in_tile < in_features; in_tile += TILE_IN) {
      int in_idx = in_tile + tile_in_idx;
      if (batch_idx < batch_size && in_idx < in_features) {
        int x_pos = batch_idx * in_features + in_idx;
        if constexpr (std::is_same<T, half>::value) {
          x_shared[tile_in_idx] = __half2float(x[x_pos]);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
          x_shared[tile_in_idx] = __bfloat162float(x[x_pos]);
        } else {
          x_shared[tile_in_idx] = x[x_pos];
        }
      }
      __syncthreads();

      if (in_idx < in_features) {
        int a_pos = r * in_features + in_idx;
        float a_val = 0.0f;
        if constexpr (std::is_same<T, half>::value) {
          a_val = __half2float(A[a_pos]);
        } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
          a_val = __bfloat162float(A[a_pos]);
        } else {
          a_val = A[a_pos];
        }

        // Reduction across threads (simplified)
        if (tile_in_idx == 0) {
          for (int t = 0; t < TILE_IN && in_tile + t < in_features; t++) {
            xa_result += x_shared[t] * a_val;
          }
        }
      }
      __syncthreads();
    }

    // Multiply by B[current_out_idx, r]
    if (tile_in_idx == 0 && current_out_idx < out_features) {
      int b_pos = current_out_idx * rank + r;
      float b_val = 0.0f;
      if constexpr (std::is_same<T, half>::value) {
        b_val = __half2float(B[b_pos]);
      } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        b_val = __bfloat162float(B[b_pos]);
      } else {
        b_val = B[b_pos];
      }
      lora_accum += xa_result * b_val;
    }
  }

  // Write output
  if (batch_idx < batch_size && current_out_idx < out_features) {
    int out_pos = batch_idx * out_features + current_out_idx;
    float final_val = base_accum + lora_accum * scaling;

    if constexpr (std::is_same<T, half>::value) {
      output[out_pos] = __float2half(final_val);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      output[out_pos] = __float2bfloat16(final_val);
    } else {
      output[out_pos] = final_val;
    }
  }
}

// Main launcher function
torch::Tensor
fused_lora_forward_cuda(torch::Tensor x,      // [batch_size, in_features]
                        torch::Tensor W_base, // [out_features, in_features]
                        torch::Tensor A,      // [rank, in_features]
                        torch::Tensor B,      // [out_features, rank]
                        float scaling) {
  int batch_size = x.size(0);
  int in_features = x.size(1);
  int out_features = W_base.size(0);
  int rank = A.size(0);

  // Allocate output
  auto output = torch::empty({batch_size, out_features}, x.options());

  // Choose kernel based on dimensions
  bool use_tiled = (out_features > 64 || in_features > 64);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "fused_lora_forward", ([&] {
        using T = scalar_t;

        if (use_tiled) {
          // Tiled version for larger matrices
          const int TILE_OUT = 16;
          dim3 blocks(batch_size, (out_features + TILE_OUT - 1) / TILE_OUT);
          dim3 threads(TILE_OUT, TILE_IN);
          int shared_mem = TILE_OUT * TILE_IN * sizeof(float);

          fused_lora_forward_tiled_kernel<T><<<blocks, threads, shared_mem>>>(
              x.data_ptr<T>(), W_base.data_ptr<T>(), A.data_ptr<T>(),
              B.data_ptr<T>(), output.data_ptr<T>(), batch_size, in_features,
              out_features, rank, scaling);
        } else {
          // Simple version for small matrices
          dim3 blocks(batch_size);
          dim3 threads(out_features);
          int shared_mem = out_features * sizeof(float);

          fused_lora_forward_kernel<T><<<blocks, threads, shared_mem>>>(
              x.data_ptr<T>(), W_base.data_ptr<T>(), A.data_ptr<T>(),
              B.data_ptr<T>(), output.data_ptr<T>(), batch_size, in_features,
              out_features, rank, scaling);
        }
      }));

  return output;
}
