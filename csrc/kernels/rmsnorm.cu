/**
 * BarqTrain Fused RMSNorm — float32 only kernels.
 * No templates → single nvcc specialization → safe on Colab 12 GB RAM.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

// Warp reduce sum
__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffff, v, off);
  return v;
}

// Forward: output = (x / rms) * weight
__global__ void rmsnorm_fwd_f32(const float *__restrict__ x,
                                const float *__restrict__ w,
                                float *__restrict__ out,
                                float *__restrict__ rms_cache, float eps,
                                int H) {
  int b = blockIdx.x;
  int tid = threadIdx.x;

  extern __shared__ float smem[];

  float sq = 0.f;
  for (int i = tid; i < H; i += blockDim.x)
    sq += x[b * H + i] * x[b * H + i];

  smem[tid] = sq;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }

  float rms = sqrtf(smem[0] / H + eps);
  if (tid == 0 && rms_cache)
    rms_cache[b] = rms;

  for (int i = tid; i < H; i += blockDim.x)
    out[b * H + i] = (x[b * H + i] / rms) * w[i];
}

// Backward
__global__ void
rmsnorm_bwd_f32(const float *__restrict__ dy, const float *__restrict__ x,
                const float *__restrict__ w, const float *__restrict__ rms,
                float *__restrict__ dx,
                float *__restrict__ dw_acc, // float accumulator for grad_weight
                int H) {
  int b = blockIdx.x;
  int tid = threadIdx.x;

  extern __shared__ float smem[];

  float r = rms[b];

  // sum(dy * w * x/rms * x/rms)
  float sum = 0.f;
  for (int i = tid; i < H; i += blockDim.x) {
    float y = x[b * H + i] / r;
    sum += dy[b * H + i] * w[i] * y;
  }
  smem[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }
  float sum_dy_y = smem[0];

  for (int i = tid; i < H; i += blockDim.x) {
    float y = x[b * H + i] / r;
    float gy = dy[b * H + i] * w[i];
    dx[b * H + i] = (gy - y * sum_dy_y / H) / r;
    atomicAdd(&dw_acc[i], dy[b * H + i] * y);
  }
}

// ---- Launchers ----

torch::Tensor fused_rmsnorm_cuda(torch::Tensor input, torch::Tensor weight,
                                 float eps) {
  int B = input.size(0), H = input.size(1);
  auto xf = input.to(torch::kFloat32).contiguous();
  auto wf = weight.to(torch::kFloat32).contiguous();
  auto out = torch::empty_like(xf);
  auto rms = torch::empty({B}, xf.options());

  int threads = std::min(H, 256);
  rmsnorm_fwd_f32<<<B, threads, threads * sizeof(float)>>>(
      xf.data_ptr<float>(), wf.data_ptr<float>(), out.data_ptr<float>(),
      rms.data_ptr<float>(), eps, H);

  return out.to(input.scalar_type());
}

torch::Tensor fused_rmsnorm_backward_cuda(torch::Tensor grad_out,
                                          torch::Tensor input,
                                          torch::Tensor weight,
                                          torch::Tensor rms) {
  int B = input.size(0), H = input.size(1);
  auto dyf = grad_out.to(torch::kFloat32).contiguous();
  auto xf = input.to(torch::kFloat32).contiguous();
  auto wf = weight.to(torch::kFloat32).contiguous();
  auto rmsf = rms.to(torch::kFloat32).contiguous();
  auto dxf = torch::empty_like(xf);
  auto dwf = torch::zeros({H}, xf.options());

  int threads = std::min(H, 256);
  rmsnorm_bwd_f32<<<B, threads, threads * sizeof(float)>>>(
      dyf.data_ptr<float>(), xf.data_ptr<float>(), wf.data_ptr<float>(),
      rmsf.data_ptr<float>(), dxf.data_ptr<float>(), dwf.data_ptr<float>(), H);

  return dxf.to(input.scalar_type());
}
