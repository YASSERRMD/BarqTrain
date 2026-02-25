/**
 * BarqTrain Fused LoRA — float32 only kernel.
 * Launcher casts inputs to float32, runs kernel, casts output back.
 * No templates → nvcc compiles ONE specialization → no OOM-kill.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void
lora_fwd_f32(const float *__restrict__ x, const float *__restrict__ W,
             const float *__restrict__ A, const float *__restrict__ B,
             float *__restrict__ out, int I, int O, int R, float scaling) {
  int b = blockIdx.x;
  int o = threadIdx.x;
  if (o >= O)
    return;

  extern __shared__ float sx[];
  for (int i = o; i < I; i += blockDim.x)
    sx[i] = x[b * I + i];
  __syncthreads();

  float base = 0.f;
  for (int i = 0; i < I; i++)
    base += sx[i] * W[o * I + i];

  float lora = 0.f;
  for (int r = 0; r < R; r++) {
    float xa = 0.f;
    for (int i = 0; i < I; i++)
      xa += sx[i] * A[r * I + i];
    lora += xa * B[o * R + r];
  }

  out[b * O + o] = base + lora * scaling;
}

torch::Tensor fused_lora_forward_cuda(torch::Tensor x, torch::Tensor W_base,
                                      torch::Tensor A, torch::Tensor B,
                                      float scaling) {
  int B_sz = x.size(0), I = x.size(1);
  int O = W_base.size(0), R = A.size(0);

  TORCH_CHECK(O <= 1024, "out_features (", O, ") > 1024 max threads/block");

  // Cast everything to float32 — single non-templated kernel
  auto xf = x.to(torch::kFloat32).contiguous();
  auto Wf = W_base.to(torch::kFloat32).contiguous();
  auto Af = A.to(torch::kFloat32).contiguous();
  auto Bf = B.to(torch::kFloat32).contiguous();
  auto of = torch::empty({B_sz, O}, xf.options());

  lora_fwd_f32<<<B_sz, O, I * sizeof(float)>>>(
      xf.data_ptr<float>(), Wf.data_ptr<float>(), Af.data_ptr<float>(),
      Bf.data_ptr<float>(), of.data_ptr<float>(), I, O, R, scaling);

  // Cast back to original dtype
  return of.to(x.scalar_type());
}
