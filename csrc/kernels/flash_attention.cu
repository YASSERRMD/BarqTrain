/**
 * BarqTrain FlashAttention Kernel with Fused RoPE
 *
 * Fixes vs original:
 *  1. RoPE cache computed on CPU tensor then .to(device) — not written
 *     to a GPU pointer from CPU (which is UB / silent wrong results).
 *  2. Correct online softmax: running_max + rescale each step.
 *  3. No #include <cmath> (conflicts with CUDA math headers).
 *  4. No "= {}" array zero-init (not supported in all nvcc device contexts).
 *  5. k_vec declared outside inner loop to avoid unsupported re-declarations.
 *  6. BLOCK_M=32 → shared memory usage is well within sm_75 32-KB limit.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// Max supported head dimension.
// Uses register storage — increase only if your GPU has enough registers.
#define MAX_D_HEAD 128
// Threads per block = queries processed per block
#define BLOCK_M 32

// ---------------------------------------------------------------------------
// Helpers
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
// Kernel: one block per (batch, head, q_tile)
//         threadIdx.x = which query in the tile (0..BLOCK_M-1)
// ---------------------------------------------------------------------------
template <typename T>
__global__ void
flash_attention_forward_kernel(const T *__restrict__ q, const T *__restrict__ k,
                               const T *__restrict__ v, T *__restrict__ output,
                               const float *__restrict__ cos_cache, // [S, D/2]
                               const float *__restrict__ sin_cache, // [S, D/2]
                               int batch_size, int n_heads, int seq_len,
                               int d_head, float scale) {
  int batch_idx = blockIdx.z;
  int head_idx = blockIdx.y;
  int q_pos = (int)blockIdx.x * BLOCK_M + (int)threadIdx.x;

  if (batch_idx >= batch_size || head_idx >= n_heads || q_pos >= seq_len)
    return;

  // ---- Load Q ----
  float q_vec[MAX_D_HEAD];
  // explicit zero-init (avoid "= {}" which some nvcc versions reject in device
  // code)
#pragma unroll
  for (int d = 0; d < MAX_D_HEAD; d++)
    q_vec[d] = 0.0f;

  int q_base = ((batch_idx * n_heads + head_idx) * seq_len + q_pos) * d_head;
  for (int d = 0; d < d_head; d++)
    q_vec[d] = load_f32(q, q_base + d);

  // Apply RoPE to Q: rotate pairs (d, d+1)
  for (int d = 0; d + 1 < d_head; d += 2) {
    float c = cos_cache[q_pos * (d_head / 2) + d / 2];
    float s = sin_cache[q_pos * (d_head / 2) + d / 2];
    float r0 = q_vec[d], r1 = q_vec[d + 1];
    q_vec[d] = r0 * c - r1 * s;
    q_vec[d + 1] = r0 * s + r1 * c;
  }

  // ---- Online softmax attention (causal: kv_pos <= q_pos) ----
  float acc[MAX_D_HEAD];
#pragma unroll
  for (int d = 0; d < MAX_D_HEAD; d++)
    acc[d] = 0.0f;

  float running_max = -1e38f; // avoid -INFINITY which can produce NaN in expf
  float running_sum = 0.0f;

  // Declare kv buffers OUTSIDE the loop to avoid re-declaration issues
  float k_vec[MAX_D_HEAD];
  float v_vec[MAX_D_HEAD];

  for (int kv_pos = 0; kv_pos <= q_pos; kv_pos++) {
    int kv_base =
        ((batch_idx * n_heads + head_idx) * seq_len + kv_pos) * d_head;

    // Load K
    for (int d = 0; d < d_head; d++)
      k_vec[d] = load_f32(k, kv_base + d);

    // Apply RoPE to K
    for (int d = 0; d + 1 < d_head; d += 2) {
      float c = cos_cache[kv_pos * (d_head / 2) + d / 2];
      float s = sin_cache[kv_pos * (d_head / 2) + d / 2];
      float r0 = k_vec[d], r1 = k_vec[d + 1];
      k_vec[d] = r0 * c - r1 * s;
      k_vec[d + 1] = r0 * s + r1 * c;
    }

    // Dot product Q · K
    float score = 0.0f;
    for (int d = 0; d < d_head; d++)
      score += q_vec[d] * k_vec[d];
    score *= scale;

    // Online softmax update
    float new_max = (score > running_max) ? score : running_max;
    float exp_s = expf(score - new_max);
    float rescale = expf(running_max - new_max);

    // Load V and update accumulator
    for (int d = 0; d < d_head; d++)
      v_vec[d] = load_f32(v, kv_base + d);

    for (int d = 0; d < d_head; d++)
      acc[d] = acc[d] * rescale + exp_s * v_vec[d];

    running_sum = running_sum * rescale + exp_s;
    running_max = new_max;
  }

  // ---- Write output ----
  int out_base = ((batch_idx * n_heads + head_idx) * seq_len + q_pos) * d_head;
  float inv_sum = 1.0f / (running_sum + 1e-10f);
  for (int d = 0; d < d_head; d++)
    store_f32(output, out_base + d, acc[d] * inv_sum);
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
torch::Tensor flash_attention_cuda(torch::Tensor q, torch::Tensor k,
                                   torch::Tensor v) {
  int B = (int)q.size(0);
  int H = (int)q.size(1);
  int S = (int)q.size(2);
  int D = (int)q.size(3);

  TORCH_CHECK(D <= MAX_D_HEAD, "flash_attention_cuda: d_head=", D,
              " > MAX_D_HEAD=", MAX_D_HEAD);

  auto output = torch::empty_like(q);

  // Compute RoPE cos/sin on CPU, then move to device.
  // The original code wrote to a GPU data_ptr from CPU — that is UB.
  auto cos_cpu = torch::zeros({S, D / 2}, torch::kFloat32);
  auto sin_cpu = torch::zeros({S, D / 2}, torch::kFloat32);
  float *cp = cos_cpu.data_ptr<float>();
  float *sp = sin_cpu.data_ptr<float>();
  const float theta = 10000.0f;
  for (int pos = 0; pos < S; pos++) {
    for (int i = 0; i < D / 2; i++) {
      float ang = pos * powf(theta, -2.0f * i / (float)D);
      cp[pos * (D / 2) + i] = cosf(ang);
      sp[pos * (D / 2) + i] = sinf(ang);
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
