/**
 * BarqTrain FlashAttention — float32 only kernel.
 * MAX_D_HEAD=64. No templates. RoPE cache computed on CPU then to(device).
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

#define MAX_D_HEAD 64
#define BLOCK_M 32

__global__ void flash_attn_fwd_f32(const float *__restrict__ q,
                                   const float *__restrict__ k,
                                   const float *__restrict__ v,
                                   float *__restrict__ out,
                                   const float *__restrict__ cos_c,
                                   const float *__restrict__ sin_c, int B,
                                   int H, int S, int D, float scale) {
  int b = blockIdx.z;
  int h = blockIdx.y;
  int qp = (int)blockIdx.x * BLOCK_M + (int)threadIdx.x;
  if (b >= B || h >= H || qp >= S)
    return;

  int half_D = D / 2;

  // Load Q
  float qv[MAX_D_HEAD];
  for (int d = 0; d < MAX_D_HEAD; d++)
    qv[d] = 0.f;
  int qbase = ((b * H + h) * S + qp) * D;
  for (int d = 0; d < D; d++)
    qv[d] = q[qbase + d];

  // RoPE on Q
  for (int d = 0; d + 1 < D; d += 2) {
    float c = cos_c[qp * half_D + d / 2];
    float s = sin_c[qp * half_D + d / 2];
    float r0 = qv[d], r1 = qv[d + 1];
    qv[d] = r0 * c - r1 * s;
    qv[d + 1] = r0 * s + r1 * c;
  }

  float acc[MAX_D_HEAD];
  for (int d = 0; d < MAX_D_HEAD; d++)
    acc[d] = 0.f;
  float rm = -1e38f, rs = 0.f;

  float kv[MAX_D_HEAD], vv[MAX_D_HEAD];

  for (int kp = 0; kp <= qp; kp++) {
    int kvbase = ((b * H + h) * S + kp) * D;
    for (int d = 0; d < D; d++)
      kv[d] = k[kvbase + d];
    for (int d = 0; d + 1 < D; d += 2) {
      float c = cos_c[kp * half_D + d / 2];
      float s = sin_c[kp * half_D + d / 2];
      float r0 = kv[d], r1 = kv[d + 1];
      kv[d] = r0 * c - r1 * s;
      kv[d + 1] = r0 * s + r1 * c;
    }
    float score = 0.f;
    for (int d = 0; d < D; d++)
      score += qv[d] * kv[d];
    score *= scale;

    float nm = (score > rm) ? score : rm;
    float es = expf(score - nm);
    float rc = expf(rm - nm);

    for (int d = 0; d < D; d++)
      vv[d] = v[kvbase + d];
    for (int d = 0; d < D; d++)
      acc[d] = acc[d] * rc + es * vv[d];
    rs = rs * rc + es;
    rm = nm;
  }

  int obase = ((b * H + h) * S + qp) * D;
  float inv = 1.f / (rs + 1e-10f);
  for (int d = 0; d < D; d++)
    out[obase + d] = acc[d] * inv;
}

torch::Tensor flash_attention_cuda(torch::Tensor q, torch::Tensor k,
                                   torch::Tensor v) {
  int B = q.size(0), H = q.size(1), S = q.size(2), D = q.size(3);
  TORCH_CHECK(D <= MAX_D_HEAD, "flash_attention_cuda: d_head=", D,
              " > MAX_D_HEAD=", MAX_D_HEAD);

  auto qf = q.to(torch::kFloat32).contiguous();
  auto kf = k.to(torch::kFloat32).contiguous();
  auto vf = v.to(torch::kFloat32).contiguous();
  auto of = torch::empty_like(qf);

  // RoPE cache on CPU then to device
  auto cc = torch::zeros({S, D / 2}, torch::kFloat32);
  auto sc = torch::zeros({S, D / 2}, torch::kFloat32);
  auto *cp = cc.data_ptr<float>();
  auto *sp = sc.data_ptr<float>();
  for (int pos = 0; pos < S; pos++)
    for (int i = 0; i < D / 2; i++) {
      float ang = pos * powf(10000.f, -2.f * i / D);
      cp[pos * (D / 2) + i] = cosf(ang);
      sp[pos * (D / 2) + i] = sinf(ang);
    }
  auto cc_d = cc.to(q.device());
  auto sc_d = sc.to(q.device());

  float scale = 1.f / sqrtf((float)D);
  dim3 blocks((S + BLOCK_M - 1) / BLOCK_M, H, B);
  flash_attn_fwd_f32<<<blocks, BLOCK_M>>>(
      qf.data_ptr<float>(), kf.data_ptr<float>(), vf.data_ptr<float>(),
      of.data_ptr<float>(), cc_d.data_ptr<float>(), sc_d.data_ptr<float>(), B,
      H, S, D, scale);

  return of.to(q.scalar_type());
}
