/**
 * BarqTrain Chunked Cross-Entropy Loss
 *
 * Correctness-focused implementation:
 *  - One block per (batch, seq) token position
 *  - Threads iterate over vocab_size; each thread owns N vocab entries
 *  - Correct full dot product: each thread loops over all hidden_dim dims
 *  - Two passes: (1) find max logit, (2) sum_exp + target_logit → loss
 *  - No atomicMax(float*) — use CAS-based atomicMaxFloat helper instead
 *  - No type-mismatched atomicAdd
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// ---------------------------------------------------------------------------
// Helper: load one element of type T as float
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
// Warp reduce: max
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
  return v;
}

// ---------------------------------------------------------------------------
// Warp reduce: sum
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffff, v, off);
  return v;
}

// ---------------------------------------------------------------------------
// Block reduce: max  (uses smem scratch[0..blockDim.x-1])
// ---------------------------------------------------------------------------
__device__ float block_reduce_max(float v, float *smem) {
  int tid = threadIdx.x;
  smem[tid] = warp_reduce_max(v);
  __syncthreads();
  // reduce across warps (assuming blockDim.x <= 1024, ≤32 warps)
  if (tid < 32) {
    float wv = (tid < (blockDim.x + 31) / 32) ? smem[tid * 32] : -INFINITY;
    wv = warp_reduce_max(wv);
    if (tid == 0)
      smem[0] = wv;
  }
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// Block reduce: sum
// ---------------------------------------------------------------------------
__device__ float block_reduce_sum(float v, float *smem) {
  int tid = threadIdx.x;
  smem[tid] = warp_reduce_sum(v);
  __syncthreads();
  if (tid < 32) {
    float wv = (tid < (blockDim.x + 31) / 32) ? smem[tid * 32] : 0.0f;
    wv = warp_reduce_sum(wv);
    if (tid == 0)
      smem[0] = wv;
  }
  __syncthreads();
  return smem[0];
}

// ---------------------------------------------------------------------------
// Forward kernel: one block per token position (batch_idx, seq_idx)
// Shared memory layout:
//   [0 .. hidden_dim-1]        : hidden state (float)
//   [hidden_dim .. hidden_dim+blockDim.x-1] : scratch for reductions
// ---------------------------------------------------------------------------
template <typename T>
__global__ void chunked_cross_entropy_fwd_kernel(
    const T *__restrict__ hidden_states, // [B, S, H]
    const T *__restrict__ weight,        // [V, H]
    const int64_t *__restrict__ labels,  // [B, S]
    float *__restrict__ loss_out,        // [B, S]
    int batch_size, int seq_len, int hidden_dim, int vocab_size) {
  int batch_idx = blockIdx.y;
  int seq_idx = blockIdx.x;
  if (batch_idx >= batch_size || seq_idx >= seq_len)
    return;

  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  // Shared: hidden state + scratch buffer
  extern __shared__ float smem[];
  float *shidden = smem;              // hidden_dim floats
  float *scratch = smem + hidden_dim; // nthreads floats

  // Load hidden state into shared memory
  int h_base = (batch_idx * seq_len + seq_idx) * hidden_dim;
  for (int d = tid; d < hidden_dim; d += nthreads)
    shidden[d] = load_f32(hidden_states, h_base + d);
  __syncthreads();

  int64_t target = labels[batch_idx * seq_len + seq_idx];

  // ---- Pass 1: find global max logit ----
  float local_max = -INFINITY;
  for (int vi = tid; vi < vocab_size; vi += nthreads) {
    float dot = 0.0f;
    int w_base = vi * hidden_dim;
    for (int d = 0; d < hidden_dim; d++)
      dot += shidden[d] * load_f32(weight, w_base + d);
    local_max = fmaxf(local_max, dot);
  }
  float global_max = block_reduce_max(local_max, scratch);

  // ---- Pass 2: sum_exp + target_logit ----
  float local_sum = 0.0f;
  float local_tgt = -INFINITY; // only one thread will have the real value
  for (int vi = tid; vi < vocab_size; vi += nthreads) {
    float dot = 0.0f;
    int w_base = vi * hidden_dim;
    for (int d = 0; d < hidden_dim; d++)
      dot += shidden[d] * load_f32(weight, w_base + d);
    local_sum += expf(dot - global_max);
    if (vi == (int)target)
      local_tgt = dot;
  }

  // Reduce sum_exp
  float total_sum = block_reduce_sum(local_sum, scratch);

  // Find target logit: use max-reduction (all other threads have -INFINITY)
  scratch[tid] = local_tgt;
  __syncthreads();
  for (int s = nthreads / 2; s > 0; s >>= 1) {
    if (tid < s)
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
    __syncthreads();
  }
  float target_logit = scratch[0];

  if (tid == 0) {
    float loss = -(target_logit - global_max - logf(total_sum + 1e-10f));
    loss_out[batch_idx * seq_len + seq_idx] = loss;
  }
}

// ---------------------------------------------------------------------------
// Backward kernel: compute grad_hidden for each token position
// grad_hidden[b,s,:] = sum_v [ softmax[v] * weight[v,:] ] - weight[target,:]
// ---------------------------------------------------------------------------
template <typename T>
__global__ void chunked_cross_entropy_bwd_kernel(
    const T *__restrict__ hidden_states, const T *__restrict__ weight,
    const int64_t *__restrict__ labels,
    const float *__restrict__ loss_vals, // needed only for sign
    T *__restrict__ grad_hidden, int batch_size, int seq_len, int hidden_dim,
    int vocab_size) {
  int batch_idx = blockIdx.y;
  int seq_idx = blockIdx.x;
  if (batch_idx >= batch_size || seq_idx >= seq_len)
    return;

  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  extern __shared__ float smem[];
  float *shidden = smem;
  float *scratch = smem + hidden_dim;

  int h_base = (batch_idx * seq_len + seq_idx) * hidden_dim;
  for (int d = tid; d < hidden_dim; d += nthreads)
    shidden[d] = load_f32(hidden_states, h_base + d);
  __syncthreads();

  int64_t target = labels[batch_idx * seq_len + seq_idx];

  // Find max logit for numerical stability
  float local_max = -INFINITY;
  for (int vi = tid; vi < vocab_size; vi += nthreads) {
    float dot = 0.0f;
    int w_base = vi * hidden_dim;
    for (int d = 0; d < hidden_dim; d++)
      dot += shidden[d] * load_f32(weight, w_base + d);
    local_max = fmaxf(local_max, dot);
  }
  float global_max = block_reduce_max(local_max, scratch);

  // Compute sum_exp
  float local_sum = 0.0f;
  for (int vi = tid; vi < vocab_size; vi += nthreads) {
    float dot = 0.0f;
    int w_base = vi * hidden_dim;
    for (int d = 0; d < hidden_dim; d++)
      dot += shidden[d] * load_f32(weight, w_base + d);
    local_sum += expf(dot - global_max);
  }
  float total_sum = block_reduce_sum(local_sum, scratch);

  // Accumulate grad_hidden into shared memory scratch for each d
  // We process d in batches to reuse scratch
  // Strategy: for each d, atomicAdd over vocab entries is too slow.
  // Instead: each thread accumulates its own partial grad, then reduce.
  // We loop over hidden_dim in chunks to avoid over-using registers.
  const int D_CHUNK = 32;
  for (int d_start = 0; d_start < hidden_dim; d_start += D_CHUNK) {
    // Zero out scratch (nthreads * D_CHUNK floats — we need a sub-scratch)
    // Simplification: use one thread to write per-d gradient serially
    // This is correct but slow for large hidden_dim; acceptable for prototype.
    if (tid == 0) {
      for (int d = d_start; d < min(d_start + D_CHUNK, hidden_dim); d++) {
        float grad_d = 0.0f;
        for (int vi = 0; vi < vocab_size; vi++) {
          float dot = 0.0f;
          int w_base = vi * hidden_dim;
          for (int d2 = 0; d2 < hidden_dim; d2++)
            dot += shidden[d2] * load_f32(weight, w_base + d2);
          float sm = expf(dot - global_max) / (total_sum + 1e-10f);
          grad_d += sm * load_f32(weight, vi * hidden_dim + d);
          if (vi == (int)target)
            grad_d -= load_f32(weight, vi * hidden_dim + d);
        }
        float out_val = grad_d;
        if constexpr (std::is_same<T, half>::value)
          grad_hidden[h_base + d] = __float2half(out_val);
        else if constexpr (std::is_same<T, __nv_bfloat16>::value)
          grad_hidden[h_base + d] = __float2bfloat16(out_val);
        else
          grad_hidden[h_base + d] = out_val;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------
std::vector<torch::Tensor>
chunked_cross_entropy_cuda(torch::Tensor hidden_states,  // [B, S, H]
                           torch::Tensor lm_head_weight, // [V, H]
                           torch::Tensor labels)         // [B, S]
{
  int B = hidden_states.size(0);
  int S = hidden_states.size(1);
  int H = hidden_states.size(2);
  int V = lm_head_weight.size(0);

  auto losses = torch::zeros({B, S}, torch::TensorOptions()
                                         .dtype(torch::kFloat32)
                                         .device(hidden_states.device()));
  auto grad_hidden = torch::zeros_like(hidden_states);

  const int THREADS = 256;
  // smem: H floats (hidden) + THREADS floats (scratch)
  int smem = (H + THREADS) * sizeof(float);

  dim3 blocks(S, B);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      hidden_states.scalar_type(), "chunked_cross_entropy", ([&] {
        using T = scalar_t;

        chunked_cross_entropy_fwd_kernel<T><<<blocks, THREADS, smem>>>(
            hidden_states.data_ptr<T>(), lm_head_weight.data_ptr<T>(),
            labels.data_ptr<int64_t>(), losses.data_ptr<float>(), B, S, H, V);

        chunked_cross_entropy_bwd_kernel<T><<<blocks, THREADS, smem>>>(
            hidden_states.data_ptr<T>(), lm_head_weight.data_ptr<T>(),
            labels.data_ptr<int64_t>(), losses.data_ptr<float>(),
            grad_hidden.data_ptr<T>(), B, S, H, V);
      }));

  return {losses, grad_hidden};
}
