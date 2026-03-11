/**
 * BarqTrain chunked linear cross-entropy.
 *
 * This implementation follows the same high-level idea used by recent
 * fused-loss work: avoid materializing the full [tokens, vocab] logits tensor
 * by processing the vocabulary dimension in chunks. We rely on ATen matmul
 * kernels for the heavy GEMMs and keep only a vocab chunk live at a time.
 */

#include <limits>
#include <vector>

#include <torch/extension.h>

namespace {

constexpr int64_t kIgnoreIndex = -100;
constexpr int64_t kDefaultChunkSize = 4096;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
chunked_cross_entropy_impl(torch::Tensor hidden_states,
                           torch::Tensor lm_head_weight,
                           torch::Tensor labels) {
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be a CUDA tensor");
  TORCH_CHECK(lm_head_weight.is_cuda(), "lm_head_weight must be a CUDA tensor");
  TORCH_CHECK(labels.is_cuda(), "labels must be a CUDA tensor");
  TORCH_CHECK(hidden_states.dim() == 3,
              "hidden_states must have shape [batch, seq, hidden]");
  TORCH_CHECK(lm_head_weight.dim() == 2,
              "lm_head_weight must have shape [vocab, hidden]");
  TORCH_CHECK(labels.dim() == 2, "labels must have shape [batch, seq]");
  TORCH_CHECK(hidden_states.size(0) == labels.size(0) &&
                  hidden_states.size(1) == labels.size(1),
              "labels must match hidden_states batch and sequence dims");
  TORCH_CHECK(hidden_states.size(2) == lm_head_weight.size(1),
              "hidden dimension must match lm_head_weight");

  auto hidden_contig = hidden_states.contiguous();
  auto weight_contig = lm_head_weight.contiguous();
  auto labels_contig = labels.contiguous();

  auto hidden_flat =
      hidden_contig.view({-1, hidden_contig.size(-1)}).to(torch::kFloat32);
  auto labels_flat = labels_contig.view({-1}).to(torch::kLong);
  auto weight_float = weight_contig.to(torch::kFloat32);

  const int64_t num_tokens = hidden_flat.size(0);
  const int64_t vocab_size = weight_float.size(0);
  const int64_t chunk_size = std::min<int64_t>(kDefaultChunkSize, vocab_size);

  auto active_mask = labels_flat.ne(kIgnoreIndex);
  const int64_t active_count = active_mask.sum().item<int64_t>();

  auto loss_options = hidden_flat.options();
  auto hidden_options = hidden_contig.options();
  auto weight_options = weight_contig.options();

  if (active_count == 0) {
    return {
        torch::zeros({}, loss_options),
        torch::zeros_like(hidden_contig),
        torch::zeros_like(weight_contig),
    };
  }

  auto neg_inf =
      torch::full({num_tokens}, -std::numeric_limits<float>::infinity(), loss_options);
  auto global_max = neg_inf.clone();
  auto target_logits = torch::zeros({num_tokens}, loss_options);
  auto logsumexp_acc = torch::zeros({num_tokens}, loss_options);

  auto active_mask_f = active_mask.to(loss_options.dtype());

  // Pass 1: compute global max logits per token.
  for (int64_t start = 0; start < vocab_size; start += chunk_size) {
    const int64_t current_chunk = std::min<int64_t>(chunk_size, vocab_size - start);
    auto weight_chunk = weight_float.narrow(0, start, current_chunk);
    auto logits_chunk = torch::matmul(hidden_flat, weight_chunk.transpose(0, 1));
    auto chunk_max = std::get<0>(logits_chunk.max(/*dim=*/1));
    global_max = torch::maximum(global_max, chunk_max);
  }

  // Pass 2: accumulate logsumexp and gather target logits.
  for (int64_t start = 0; start < vocab_size; start += chunk_size) {
    const int64_t current_chunk = std::min<int64_t>(chunk_size, vocab_size - start);
    auto weight_chunk = weight_float.narrow(0, start, current_chunk);
    auto logits_chunk = torch::matmul(hidden_flat, weight_chunk.transpose(0, 1));
    logsumexp_acc +=
        torch::exp(logits_chunk - global_max.unsqueeze(1)).sum(/*dim=*/1);

    auto local_labels = labels_flat - start;
    auto in_chunk =
        active_mask.logical_and(local_labels.ge(0)).logical_and(local_labels.lt(current_chunk));
    if (in_chunk.any().item<bool>()) {
      auto token_indices = torch::nonzero(in_chunk).squeeze(1);
      auto gather_indices =
          local_labels.index_select(0, token_indices).unsqueeze(1);
      auto selected_logits = logits_chunk.index_select(0, token_indices)
                                 .gather(1, gather_indices)
                                 .squeeze(1);
      target_logits.index_copy_(0, token_indices, selected_logits);
    }
  }

  auto per_token_loss =
      global_max + torch::log(logsumexp_acc + 1e-10f) - target_logits;
  per_token_loss.masked_fill_(labels_flat.eq(kIgnoreIndex), 0.0f);
  auto loss = per_token_loss.sum() / static_cast<double>(active_count);

  auto grad_hidden = torch::zeros_like(hidden_flat);
  auto grad_weight = torch::zeros_like(weight_float);
  auto inv_active_count = 1.0 / static_cast<double>(active_count);

  // Pass 3: recompute chunked probabilities and gradients without
  // materializing the full logits tensor.
  for (int64_t start = 0; start < vocab_size; start += chunk_size) {
    const int64_t current_chunk = std::min<int64_t>(chunk_size, vocab_size - start);
    auto weight_chunk = weight_float.narrow(0, start, current_chunk);
    auto logits_chunk = torch::matmul(hidden_flat, weight_chunk.transpose(0, 1));
    auto probs =
        torch::exp(logits_chunk - global_max.unsqueeze(1)) / logsumexp_acc.unsqueeze(1);
    probs *= active_mask_f.unsqueeze(1);

    auto local_labels = labels_flat - start;
    auto in_chunk =
        active_mask.logical_and(local_labels.ge(0)).logical_and(local_labels.lt(current_chunk));
    if (in_chunk.any().item<bool>()) {
      auto token_indices = torch::nonzero(in_chunk).squeeze(1);
      auto gather_indices =
          local_labels.index_select(0, token_indices).unsqueeze(1);
      auto selected_probs = probs.index_select(0, token_indices);
      auto minus_one = torch::zeros_like(selected_probs);
      minus_one.scatter_(1, gather_indices, 1.0);
      probs.index_copy_(0, token_indices, selected_probs - minus_one);
    }

    probs *= inv_active_count;
    grad_hidden += torch::matmul(probs, weight_chunk);
    grad_weight.narrow(0, start, current_chunk)
        .copy_(torch::matmul(probs.transpose(0, 1), hidden_flat));
  }

  return {
      loss,
      grad_hidden.view_as(hidden_contig).to(hidden_options.dtype()),
      grad_weight.to(weight_options.dtype()),
  };
}

} // namespace

std::vector<torch::Tensor>
chunked_cross_entropy_cuda(torch::Tensor hidden_states,
                           torch::Tensor lm_head_weight,
                           torch::Tensor labels) {
  auto [loss, grad_hidden, grad_weight] =
      chunked_cross_entropy_impl(hidden_states, lm_head_weight, labels);
  return {loss, grad_hidden, grad_weight};
}
