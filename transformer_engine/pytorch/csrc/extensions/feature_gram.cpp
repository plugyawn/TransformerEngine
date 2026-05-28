/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/feature_gram.h"

#include <ATen/cuda/CUDAContext.h>

#include "../extensions.h"

namespace transformer_engine::pytorch {

namespace {

TensorWrapper make_input_tensor(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "feature_gram: x must be a CUDA tensor.");
  TORCH_CHECK(x.dim() == 2, "feature_gram: x must be 2D, got ndim=", x.dim());
  TORCH_CHECK(x.is_contiguous(), "feature_gram: x must be contiguous.");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf ||
                  x.scalar_type() == at::kFloat,
              "feature_gram: x must be bfloat16, float16, or float32 (got ", x.scalar_type(),
              ").");

  std::vector<size_t> shape{static_cast<size_t>(x.size(0)), static_cast<size_t>(x.size(1))};
  return TensorWrapper(x.data_ptr(), shape, GetTransformerEngineDType(x.scalar_type()));
}

TensorWrapper make_output_tensor(const at::Tensor& out) {
  TORCH_CHECK(out.is_cuda(), "feature_gram: out must be a CUDA tensor.");
  TORCH_CHECK(out.is_contiguous(), "feature_gram: out must be contiguous.");
  TORCH_CHECK(out.scalar_type() == at::kFloat,
              "feature_gram: out must be float32 (got ", out.scalar_type(), ").");

  std::vector<size_t> shape;
  shape.reserve(out.dim());
  for (int64_t i = 0; i < out.dim(); ++i) {
    shape.push_back(static_cast<size_t>(out.size(i)));
  }
  return TensorWrapper(out.data_ptr(), shape, GetTransformerEngineDType(out.scalar_type()));
}

}  // namespace

void feature_gram_diag(at::Tensor x, at::Tensor out) {
  auto x_tensor = make_input_tensor(x);
  auto out_tensor = make_output_tensor(out);
  TORCH_CHECK(out.dim() == 1, "feature_gram_diag: out must be 1D, got ndim=", out.dim());
  TORCH_CHECK(out.size(0) == x.size(1),
              "feature_gram_diag: out length (", out.size(0),
              ") must match x trailing dim (", x.size(1), ").");

  auto stream = at::cuda::getCurrentCUDAStream();
  nvte_feature_gram_diag(stream.stream(), x_tensor.data(), out_tensor.data());
}

void feature_gram_block_diag(at::Tensor x, at::Tensor out, int64_t block_size) {
  auto x_tensor = make_input_tensor(x);
  auto out_tensor = make_output_tensor(out);
  TORCH_CHECK(out.dim() == 3,
              "feature_gram_block_diag: out must be 3D, got ndim=", out.dim());
  TORCH_CHECK(out.size(1) == block_size && out.size(2) == block_size,
              "feature_gram_block_diag: out shape must be [num_blocks, ", block_size, ", ",
              block_size, "], got [", out.size(0), ", ", out.size(1), ", ", out.size(2), "].");
  TORCH_CHECK(out.size(0) * block_size >= x.size(1),
              "feature_gram_block_diag: num_blocks * block_size (",
              out.size(0) * block_size, ") must be >= trailing dim of x (", x.size(1), ").");

  auto stream = at::cuda::getCurrentCUDAStream();
  nvte_feature_gram_block_diag(stream.stream(), x_tensor.data(), out_tensor.data(), block_size);
}

}  // namespace transformer_engine::pytorch
