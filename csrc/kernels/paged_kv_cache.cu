/**
 * BarqTrain paged KV-cache append kernel.
 *
 * Layout:
 *   cache = [batch, kv_heads, max_blocks, page_size, head_dim]
 * This keeps each logical page contiguous while still allowing a view into
 * [batch, kv_heads, seq_len, head_dim] without growing tensors via torch.cat.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void paged_kv_append_kernel(
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int32_t* __restrict__ seq_lens,
    const scalar_t* __restrict__ key_states,
    const scalar_t* __restrict__ value_states,
    int batch_size,
    int kv_heads,
    int token_count,
    int head_dim,
    int max_blocks,
    int page_size) {
  int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int token_idx = blockIdx.y;
  int batch_head_idx = blockIdx.z;

  if (dim_idx >= head_dim || token_idx >= token_count || batch_head_idx >= batch_size * kv_heads) {
    return;
  }

  const int batch_idx = batch_head_idx / kv_heads;
  const int kv_head_idx = batch_head_idx % kv_heads;
  const int absolute_pos = seq_lens[batch_idx] + token_idx;
  const int block_idx = absolute_pos / page_size;
  const int page_offset = absolute_pos % page_size;

  if (block_idx >= max_blocks) {
    return;
  }

  const int64_t cache_offset =
      ((((int64_t)batch_idx * kv_heads + kv_head_idx) * max_blocks + block_idx) * page_size + page_offset) * head_dim
      + dim_idx;
  const int64_t state_offset =
      ((((int64_t)batch_idx * kv_heads + kv_head_idx) * token_count + token_idx) * head_dim) + dim_idx;

  key_cache[cache_offset] = key_states[state_offset];
  value_cache[cache_offset] = value_states[state_offset];
}

}  // namespace

void paged_kv_append_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor seq_lens,
    torch::Tensor key_states,
    torch::Tensor value_states) {
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(key_states.is_cuda(), "key_states must be a CUDA tensor");
  TORCH_CHECK(value_states.is_cuda(), "value_states must be a CUDA tensor");
  TORCH_CHECK(key_cache.dim() == 5, "key_cache must have shape [batch, kv_heads, max_blocks, page_size, head_dim]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache must match key_cache");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [batch]");
  TORCH_CHECK(key_states.dim() == 4, "key_states must have shape [batch, kv_heads, seq, head_dim]");
  TORCH_CHECK(value_states.sizes() == key_states.sizes(), "value_states must match key_states");
  TORCH_CHECK(key_cache.scalar_type() == key_states.scalar_type(), "key_cache and key_states must share dtype");
  TORCH_CHECK(value_cache.scalar_type() == value_states.scalar_type(), "value_cache and value_states must share dtype");
  TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
  TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");

  const auto batch_size = key_states.size(0);
  const auto kv_heads = key_states.size(1);
  const auto token_count = key_states.size(2);
  const auto head_dim = key_states.size(3);
  const auto max_blocks = key_cache.size(2);
  const auto page_size = key_cache.size(3);
  const auto max_cache_len = max_blocks * page_size;

  TORCH_CHECK(batch_size <= key_cache.size(0), "key_cache batch dimension is too small");
  TORCH_CHECK(kv_heads == key_cache.size(1), "key_cache kv_heads dimension does not match key_states");
  TORCH_CHECK(head_dim == key_cache.size(4), "key_cache head_dim dimension does not match key_states");
  TORCH_CHECK(batch_size <= seq_lens.size(0), "seq_lens batch dimension is too small");

  auto seq_lens_int = seq_lens.to(torch::kInt32).contiguous();
  const auto max_start = seq_lens_int.narrow(0, 0, batch_size).max().item<int32_t>();
  TORCH_CHECK(
      max_start + token_count <= max_cache_len,
      "paged KV cache capacity exceeded: requested ",
      max_start + token_count,
      " tokens, max_cache_len=",
      max_cache_len);

  auto key_states_contig = key_states.contiguous();
  auto value_states_contig = value_states.contiguous();

  constexpr int threads = 128;
  const dim3 blocks((head_dim + threads - 1) / threads, token_count, batch_size * kv_heads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      key_states_contig.scalar_type(),
      "paged_kv_append_cuda",
      [&] {
        paged_kv_append_kernel<scalar_t><<<blocks, threads>>>(
            key_cache.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            seq_lens_int.data_ptr<int32_t>(),
            key_states_contig.data_ptr<scalar_t>(),
            value_states_contig.data_ptr<scalar_t>(),
            batch_size,
            kv_heads,
            token_count,
            head_dim,
            max_blocks,
            page_size);
      });

  seq_lens.narrow(0, 0, batch_size).add_(token_count);
}
