/**
 * BarqTrain paged KV-cache append/gather kernels.
 *
 * Layout:
 *   cache = [physical_blocks, kv_heads, page_size, head_dim]
 *   page_table = [batch, max_blocks_per_sequence]
 *
 * Each logical block for a sequence resolves to a physical block via the page table,
 * which allows the allocator to recycle freed blocks without moving live cache pages.
 */

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void paged_kv_append_kernel(
    scalar_t* __restrict__ key_cache,
    scalar_t* __restrict__ value_cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_lens,
    const scalar_t* __restrict__ key_states,
    const scalar_t* __restrict__ value_states,
    int batch_size,
    int kv_heads,
    int token_count,
    int head_dim,
    int max_blocks_per_sequence,
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
  const int physical_block =
      page_table[batch_idx * max_blocks_per_sequence + block_idx];

  if (block_idx >= max_blocks_per_sequence || physical_block < 0) {
    return;
  }

  const int64_t cache_offset =
      ((((int64_t)physical_block * kv_heads) + kv_head_idx) * page_size + page_offset) * head_dim
      + dim_idx;
  const int64_t state_offset =
      ((((int64_t)batch_idx * kv_heads + kv_head_idx) * token_count + token_idx) * head_dim) + dim_idx;

  key_cache[cache_offset] = key_states[state_offset];
  value_cache[cache_offset] = value_states[state_offset];
}

template <typename scalar_t>
__global__ void paged_kv_gather_kernel(
    const scalar_t* __restrict__ cache,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ seq_lens,
    scalar_t* __restrict__ output,
    int batch_size,
    int kv_heads,
    int max_seq_len,
    int head_dim,
    int max_blocks_per_sequence,
    int page_size) {
  int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int token_idx = blockIdx.y;
  int batch_head_idx = blockIdx.z;

  if (dim_idx >= head_dim || token_idx >= max_seq_len || batch_head_idx >= batch_size * kv_heads) {
    return;
  }

  const int batch_idx = batch_head_idx / kv_heads;
  if (token_idx >= seq_lens[batch_idx]) {
    return;
  }

  const int kv_head_idx = batch_head_idx % kv_heads;
  const int logical_block = token_idx / page_size;
  const int page_offset = token_idx % page_size;
  const int physical_block =
      page_table[batch_idx * max_blocks_per_sequence + logical_block];

  if (physical_block < 0) {
    return;
  }

  const int64_t cache_offset =
      ((((int64_t)physical_block * kv_heads) + kv_head_idx) * page_size + page_offset) * head_dim
      + dim_idx;
  const int64_t output_offset =
      ((((int64_t)batch_idx * kv_heads + kv_head_idx) * max_seq_len + token_idx) * head_dim) + dim_idx;

  output[output_offset] = cache[cache_offset];
}

template <typename scalar_t>
__global__ void paged_kv_gather_quantized_kernel(
    const int8_t* __restrict__ quantized_cache,
    const scalar_t* __restrict__ residual_cache,
    const float* __restrict__ scales,
    const int32_t* __restrict__ page_table,
    const int32_t* __restrict__ residual_page_table,
    const int32_t* __restrict__ seq_lens,
    scalar_t* __restrict__ output,
    int batch_size,
    int kv_heads,
    int max_seq_len,
    int head_dim,
    int max_blocks_per_sequence,
    int page_size) {
  int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int token_idx = blockIdx.y;
  int batch_head_idx = blockIdx.z;

  if (dim_idx >= head_dim || token_idx >= max_seq_len || batch_head_idx >= batch_size * kv_heads) {
    return;
  }

  const int batch_idx = batch_head_idx / kv_heads;
  if (token_idx >= seq_lens[batch_idx]) {
    return;
  }

  const int kv_head_idx = batch_head_idx % kv_heads;
  const int logical_block = token_idx / page_size;
  const int page_offset = token_idx % page_size;
  const int residual_slot =
      residual_page_table[batch_idx * max_blocks_per_sequence + logical_block];
  const int64_t output_offset =
      ((((int64_t)batch_idx * kv_heads + kv_head_idx) * max_seq_len + token_idx) * head_dim) + dim_idx;

  if (residual_slot >= 0) {
    const int64_t residual_offset =
        ((((int64_t)residual_slot * kv_heads) + kv_head_idx) * page_size + page_offset) * head_dim + dim_idx;
    output[output_offset] = residual_cache[residual_offset];
    return;
  }

  const int physical_block =
      page_table[batch_idx * max_blocks_per_sequence + logical_block];
  if (physical_block < 0) {
    return;
  }

  const int64_t cache_offset =
      ((((int64_t)physical_block * kv_heads) + kv_head_idx) * page_size + page_offset) * head_dim
      + dim_idx;
  const float scale = scales[physical_block * kv_heads + kv_head_idx];
  output[output_offset] = static_cast<scalar_t>(static_cast<float>(quantized_cache[cache_offset]) * scale);
}

}  // namespace

void paged_kv_append_cuda(
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor page_table,
    torch::Tensor seq_lens,
    torch::Tensor key_states,
    torch::Tensor value_states) {
  TORCH_CHECK(key_cache.is_cuda(), "key_cache must be a CUDA tensor");
  TORCH_CHECK(value_cache.is_cuda(), "value_cache must be a CUDA tensor");
  TORCH_CHECK(page_table.is_cuda(), "page_table must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(key_states.is_cuda(), "key_states must be a CUDA tensor");
  TORCH_CHECK(value_states.is_cuda(), "value_states must be a CUDA tensor");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must have shape [physical_blocks, kv_heads, page_size, head_dim]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache must match key_cache");
  TORCH_CHECK(page_table.dim() == 2, "page_table must have shape [batch, max_blocks_per_sequence]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [batch]");
  TORCH_CHECK(key_states.dim() == 4, "key_states must have shape [batch, kv_heads, seq, head_dim]");
  TORCH_CHECK(value_states.sizes() == key_states.sizes(), "value_states must match key_states");
  TORCH_CHECK(key_cache.scalar_type() == key_states.scalar_type(), "key_cache and key_states must share dtype");
  TORCH_CHECK(value_cache.scalar_type() == value_states.scalar_type(), "value_cache and value_states must share dtype");
  TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
  TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt32, "page_table must use int32");

  const auto batch_size = key_states.size(0);
  const auto kv_heads = key_states.size(1);
  const auto token_count = key_states.size(2);
  const auto head_dim = key_states.size(3);
  const auto page_size = key_cache.size(2);
  const auto max_blocks_per_sequence = page_table.size(1);
  const auto max_cache_len = max_blocks_per_sequence * page_size;

  TORCH_CHECK(kv_heads == key_cache.size(1), "key_cache kv_heads dimension does not match key_states");
  TORCH_CHECK(head_dim == key_cache.size(3), "key_cache head_dim dimension does not match key_states");
  TORCH_CHECK(batch_size <= page_table.size(0), "page_table batch dimension is too small");
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
  auto page_table_contig = page_table.contiguous();

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
            page_table_contig.data_ptr<int32_t>(),
            seq_lens_int.data_ptr<int32_t>(),
            key_states_contig.data_ptr<scalar_t>(),
            value_states_contig.data_ptr<scalar_t>(),
            batch_size,
            kv_heads,
            token_count,
            head_dim,
            max_blocks_per_sequence,
            page_size);
      });

  seq_lens.narrow(0, 0, batch_size).add_(token_count);
}

torch::Tensor paged_kv_gather_cuda(
    torch::Tensor cache,
    torch::Tensor page_table,
    torch::Tensor seq_lens) {
  TORCH_CHECK(cache.is_cuda(), "cache must be a CUDA tensor");
  TORCH_CHECK(page_table.is_cuda(), "page_table must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(cache.dim() == 4, "cache must have shape [physical_blocks, kv_heads, page_size, head_dim]");
  TORCH_CHECK(page_table.dim() == 2, "page_table must have shape [batch, max_blocks_per_sequence]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [batch]");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt32, "page_table must use int32");

  const auto batch_size = page_table.size(0);
  const auto kv_heads = cache.size(1);
  const auto page_size = cache.size(2);
  const auto head_dim = cache.size(3);
  const auto max_blocks_per_sequence = page_table.size(1);

  auto seq_lens_int = seq_lens.to(torch::kInt32).contiguous();
  const auto max_seq_len = batch_size == 0 ? 0 : seq_lens_int.max().item<int32_t>();
  auto output = torch::zeros(
      {batch_size, kv_heads, max_seq_len, head_dim},
      cache.options());
  if (max_seq_len == 0) {
    return output;
  }

  auto page_table_contig = page_table.contiguous();
  constexpr int threads = 128;
  const dim3 blocks((head_dim + threads - 1) / threads, max_seq_len, batch_size * kv_heads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      cache.scalar_type(),
      "paged_kv_gather_cuda",
      [&] {
        paged_kv_gather_kernel<scalar_t><<<blocks, threads>>>(
            cache.data_ptr<scalar_t>(),
            page_table_contig.data_ptr<int32_t>(),
            seq_lens_int.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            kv_heads,
            max_seq_len,
            head_dim,
            max_blocks_per_sequence,
            page_size);
      });

  return output;
}

torch::Tensor paged_kv_gather_quantized_cuda(
    torch::Tensor quantized_cache,
    torch::Tensor residual_cache,
    torch::Tensor scales,
    torch::Tensor page_table,
    torch::Tensor residual_page_table,
    torch::Tensor seq_lens) {
  TORCH_CHECK(quantized_cache.is_cuda(), "quantized_cache must be a CUDA tensor");
  TORCH_CHECK(residual_cache.is_cuda(), "residual_cache must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
  TORCH_CHECK(page_table.is_cuda(), "page_table must be a CUDA tensor");
  TORCH_CHECK(residual_page_table.is_cuda(), "residual_page_table must be a CUDA tensor");
  TORCH_CHECK(seq_lens.is_cuda(), "seq_lens must be a CUDA tensor");
  TORCH_CHECK(
      quantized_cache.dim() == 4,
      "quantized_cache must have shape [physical_blocks, kv_heads, page_size, head_dim]");
  TORCH_CHECK(
      residual_cache.dim() == 4,
      "residual_cache must have shape [residual_slots, kv_heads, page_size, head_dim]");
  TORCH_CHECK(page_table.dim() == 2, "page_table must have shape [batch, max_blocks_per_sequence]");
  TORCH_CHECK(
      residual_page_table.dim() == 2,
      "residual_page_table must have shape [batch, max_blocks_per_sequence]");
  TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must have shape [batch]");
  TORCH_CHECK(quantized_cache.scalar_type() == torch::kChar, "quantized_cache must use int8");
  TORCH_CHECK(
      residual_cache.scalar_type() == at::kHalf
          || residual_cache.scalar_type() == at::kBFloat16
          || residual_cache.scalar_type() == at::kFloat,
      "residual_cache must use fp16, bf16, or fp32");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must use float32");
  TORCH_CHECK(page_table.scalar_type() == torch::kInt32, "page_table must use int32");
  TORCH_CHECK(
      residual_page_table.scalar_type() == torch::kInt32,
      "residual_page_table must use int32");

  const auto batch_size = page_table.size(0);
  const auto kv_heads = quantized_cache.size(1);
  const auto page_size = quantized_cache.size(2);
  const auto head_dim = quantized_cache.size(3);
  const auto max_blocks_per_sequence = page_table.size(1);

  TORCH_CHECK(
      residual_cache.size(1) == kv_heads,
      "residual_cache kv_heads dimension does not match quantized_cache");
  TORCH_CHECK(
      residual_cache.size(2) == page_size,
      "residual_cache page_size dimension does not match quantized_cache");
  TORCH_CHECK(
      residual_cache.size(3) == head_dim,
      "residual_cache head_dim dimension does not match quantized_cache");
  TORCH_CHECK(
      scales.size(0) == quantized_cache.size(0) && scales.size(1) == kv_heads,
      "scales must have shape [physical_blocks, kv_heads]");

  auto seq_lens_int = seq_lens.to(torch::kInt32).contiguous();
  const auto max_seq_len = batch_size == 0 ? 0 : seq_lens_int.max().item<int32_t>();
  auto output = torch::zeros(
      {batch_size, kv_heads, max_seq_len, head_dim},
      residual_cache.options());
  if (max_seq_len == 0) {
    return output;
  }

  auto page_table_contig = page_table.contiguous();
  auto residual_page_table_contig = residual_page_table.contiguous();
  auto scales_contig = scales.contiguous();
  constexpr int threads = 128;
  const dim3 blocks((head_dim + threads - 1) / threads, max_seq_len, batch_size * kv_heads);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      residual_cache.scalar_type(),
      "paged_kv_gather_quantized_cuda",
      [&] {
        paged_kv_gather_quantized_kernel<scalar_t><<<blocks, threads>>>(
            quantized_cache.data_ptr<int8_t>(),
            residual_cache.data_ptr<scalar_t>(),
            scales_contig.data_ptr<float>(),
            page_table_contig.data_ptr<int32_t>(),
            residual_page_table_contig.data_ptr<int32_t>(),
            seq_lens_int.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            kv_heads,
            max_seq_len,
            head_dim,
            max_blocks_per_sequence,
            page_size);
      });

  return output;
}
