/**
 * BarqTrain FlashAttention Kernel with Fused RoPE
 *
 * Correctness-focused implementation.
 *
 * Key fixes vs original:
 *  1. RoPE cos/sin cache is computed on CPU tensors then .to(device) —
 *     the original called a CPU function to write to a GPU pointer (UB/crash).
 *  2. Online softmax is correctly implemented (running max + rescale).
 *  3. Shared memory sized correctly (< 32 KB for sm_75 compatibility).
 *  4. No VLA or out-of-bounds register arrays.
 *
 * Design:
 *  - One block per (batch, head, q_tile) — blockDim.x = BLOCK_M threads
 *  - Each thread handles one query position
 *  - KV positions processed sequentially (causal mask enforced)
 *  - d_head must be <= MAX_D_HEAD (compile-time constant)
 */

#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Max supported head dimension (increase if needed, uses register space)
#define MAX_D_HEAD 128
#define BLOCK_M 32 // threads per block = queries per block

// ---------------------------------------------------------------------------
// Helper: load as float
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float load_f32(const T *ptr, int idx) {
  if constexpr (std::is_same<T, half>::value)
    return __half2float(ptr[idx]);
  else if constexpr (std::is_same<T, __nv_bfloat16>::value)
    return __bfloat162float(ptr[idx]);
  else
    return ptr[idx];
}

// ---------------------------------------------------------------------------
// Helper: store float as T
// ---------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ void store_f32(T *ptr, int idx, float val) {
  if constexpr (std::is_same<T, half>::value)
    ptr[idx] = __float2half(val);
  else if constexpr (std::is_same<T, __nv_bfloat16>::value)
    ptr[idx] = __float2bfloat16(val);
  else
    ptr[idx] = val;
}

// ---------------------------------------------------------------------------
// FlashAttention forward kernel
//  grid:  (ceil(seq_len/BLOCK_M), n_heads, batch_size)
//  block: (BLOCK_M, 1, 1)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void
flash_attention_forward_kernel(const T *__restrict__ q, // [B, H, S, D]
                               const T *__restrict__ k, // [B, H, S, D]
                               const T *__restrict__ v, // [B, H, S, D]
                               T *__restrict__ output,  // [B, H, S, D]
                               const float *__restrict__ cos_cache, // [S, D/2]
                               const float *__restrict__ sin_cache, // [S, D/2]
                               int batch_size, int n_heads, int seq_len,
                               int d_head, float scale) {
  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int q_pos = blockIdx.x * BLOCK_M + threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= n_heads || q_pos >= seq_len)
    return;

  // ---- Load Q and apply RoPE ----
  float q_vec[MAX_D_HEAD];
  int q_base = ((batch_idx * n_heads + head_idx) * seq_len + q_pos) * d_head;
  for (int d = 0; d < d_head; d++)
    q_vec[d] = load_f32(q, q_base + d);

  // RoPE on Q: rotate pairs (d, d+1) by (cos, sin) at position q_pos
  for (int d = 0; d + 1 < d_head; d += 2) {
    float c = cos_cache[q_pos * (d_head / 2) + d / 2];
    float s = sin_cache[q_pos * (d_head / 2) + d / 2];
    float r = q_vec[d], i = q_vec[d + 1];
    q_vec[d] = r * c - i * s;
    q_vec[d + 1] = r * s + i * c;
  }

  // ---- Online softmax attention (causal) ----
  float acc[MAX_D_HEAD] = {};
  float running_max = -INFINITY;
  float running_sum = 0.0f;

  for (int kv_pos = 0; kv_pos <= q_pos; kv_pos++) {
    int kv_base =
        ((batch_idx * n_heads + head_idx) * seq_len + kv_pos) * d_head;

    // Load K and apply RoPE
    float k_vec[MAX_D_HEAD];
    for (int d = 0; d < d_head; d++)
      k_vec[d] = load_f32(k, kv_base + d);
    for (int d = 0; d + 1 < d_head; d += 2) {
      float c = cos_cache[kv_pos * (d_head / 2) + d / 2];
      float s = sin_cache[kv_pos * (d_head / 2) + d / 2];
      float r = k_vec[d], ii = k_vec[d + 1];
      k_vec[d] = r * c - ii * s;
      k_vec[d + 1] = r * s + ii * c;
    }

    // Attention score: Q · K * scale
    float score = 0.0f;
    for (int d = 0; d < d_head; d++)
      score += q_vec[d] * k_vec[d];
    score *= scale;

    // Online softmax update (numerically stable)
    float new_max = fmaxf(running_max, score);
    float exp_s = expf(score - new_max);
    float rescale = expf(running_max - new_max);

    // Load V and accumulate
    for (int d = 0; d < d_head; d++) {
      float v_val = load_f32(v, kv_base + d);
      acc[d] = acc[d] * rescale + exp_s * v_val;
    }

    running_sum = running_sum * rescale + exp_s;
    running_max = new_max;
  }

  // ---- Write normalised output ----
  int out_base = ((batch_idx * n_heads + head_idx) * seq_len + q_pos) * d_head;
  for (int d = 0; d < d_head; d++)
    store_f32(output, out_base + d, acc[d] / (running_sum + 1e-10f));
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
torch::Tensor flash_attention_cuda(torch::Tensor q, // [B, H, S, D]
                                   torch::Tensor k, torch::Tensor v) {
  int B = q.size(0);
  int H = q.size(1);
  int S = q.size(2);
  int D = q.size(3);

  TORCH_CHECK(D <= MAX_D_HEAD, "flash_attention_cuda: d_head (", D,
              ") > MAX_D_HEAD (", MAX_D_HEAD, ")");

  auto output = torch::empty_like(q);

  // Compute RoPE cache on CPU, then move to device.
  // (Original code wrote to a GPU pointer from CPU which is undefined
  // behaviour.)
  auto cos_cpu = torch::zeros({S, D / 2}, torch::kFloat32);
  auto sin_cpu = torch::zeros({S, D / 2}, torch::kFloat32);
  auto *cos_ptr = cos_cpu.data_ptr<float>();
  auto *sin_ptr = sin_cpu.data_ptr<float>();
  const float theta = 10000.0f;
  for (int pos = 0; pos < S; pos++) {
    for (int i = 0; i < D / 2; i++) {
      float freq = powf(theta, -2.0f * i / (float)D);
      cos_ptr[pos * (D / 2) + i] = cosf(pos * freq);
      sin_ptr[pos * (D / 2) + i] = sinf(pos * freq);
    }
  }
  auto cos_cache = cos_cpu.to(q.device());
  auto sin_cache = sin_cpu.to(q.device());

  float scale = 1.0f / sqrtf((float)D);

  dim3 blocks((S + BLOCK_M - 1) / BLOCK_M, H, B);
  dim3 threads(BLOCK_M);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(),
      "flash_attention_forward", ([&] {
        using T = scalar_t;
        flash_attention_forward_kernel<T><<<blocks, threads>>>(
            q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
            output.data_ptr<T>(), cos_cache.data_ptr<float>(),
            sin_cache.data_ptr<float>(), B, H, S, D, scale);
      }));

  return output;
}
