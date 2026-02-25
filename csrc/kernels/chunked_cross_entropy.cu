/**
 * BarqTrain Chunked Cross-Entropy — float32 only.
 * No templates. One block per (batch, seq) token position.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void cce_fwd_f32(const float *__restrict__ hidden,   // [B,S,H]
                            const float *__restrict__ weight,   // [V,H]
                            const int64_t *__restrict__ labels, // [B,S]
                            float *__restrict__ loss_out,       // [B,S]
                            int S, int H, int V) {
  int b = blockIdx.y;
  int s = blockIdx.x;
  int tid = threadIdx.x;
  int nth = blockDim.x;

  extern __shared__ float smem[];
  float *sh = smem;      // H floats: hidden state
  float *scr = smem + H; // nth floats: scratch for reduction

  int hbase = (b * S + s) * H;
  for (int d = tid; d < H; d += nth)
    sh[d] = hidden[hbase + d];
  __syncthreads();

  // Pass 1: max logit
  float lmax = -1e38f;
  for (int vi = tid; vi < V; vi += nth) {
    float dot = 0.f;
    for (int d = 0; d < H; d++)
      dot += sh[d] * weight[vi * H + d];
    lmax = (dot > lmax) ? dot : lmax;
  }
  scr[tid] = lmax;
  __syncthreads();
  for (int s2 = nth / 2; s2 > 0; s2 >>= 1) {
    if (tid < s2)
      scr[tid] = (scr[tid] > scr[tid + s2]) ? scr[tid] : scr[tid + s2];
    __syncthreads();
  }
  float gmax = scr[0];

  // Pass 2: sum_exp + target logit
  int64_t tgt = labels[b * S + s];
  float lsum = 0.f, ltgt = -1e38f;
  for (int vi = tid; vi < V; vi += nth) {
    float dot = 0.f;
    for (int d = 0; d < H; d++)
      dot += sh[d] * weight[vi * H + d];
    lsum += expf(dot - gmax);
    if (vi == (int)tgt)
      ltgt = dot;
  }
  scr[tid] = lsum;
  __syncthreads();
  for (int s2 = nth / 2; s2 > 0; s2 >>= 1) {
    if (tid < s2)
      scr[tid] += scr[tid + s2];
    __syncthreads();
  }
  float gsum = scr[0];

  // Reduce target logit (max-reduction; only one thread has it)
  scr[tid] = ltgt;
  __syncthreads();
  for (int s2 = nth / 2; s2 > 0; s2 >>= 1) {
    if (tid < s2)
      scr[tid] = (scr[tid] > scr[tid + s2]) ? scr[tid] : scr[tid + s2];
    __syncthreads();
  }

  if (tid == 0)
    loss_out[b * S + s] = -(scr[0] - gmax - logf(gsum + 1e-10f));
}

std::vector<torch::Tensor>
chunked_cross_entropy_cuda(torch::Tensor hidden_states,
                           torch::Tensor lm_head_weight, torch::Tensor labels) {
  int B = hidden_states.size(0);
  int S = hidden_states.size(1);
  int H = hidden_states.size(2);
  int V = lm_head_weight.size(0);

  auto hf = hidden_states.to(torch::kFloat32).contiguous();
  auto wf = lm_head_weight.to(torch::kFloat32).contiguous();
  auto loss = torch::zeros({B, S}, hf.options());
  auto grad = torch::zeros_like(hf); // simplified: return zero grad

  const int THREADS = 256;
  int smem = (H + THREADS) * sizeof(float);
  dim3 blocks(S, B);

  cce_fwd_f32<<<blocks, THREADS, smem>>>(
      hf.data_ptr<float>(), wf.data_ptr<float>(), labels.data_ptr<int64_t>(),
      loss.data_ptr<float>(), S, H, V);

  // grad_hidden returned as original dtype zeros (placeholder)
  return {loss, grad.to(hidden_states.scalar_type())};
}
