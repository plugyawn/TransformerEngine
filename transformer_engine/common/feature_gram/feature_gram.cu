/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/feature_gram.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.cuh"

namespace transformer_engine {

namespace {

// Each block handles a horizontal tile of kColTile columns. Each thread owns
// one column and accumulates X[r, c]^2 across all rows. Threads of a warp hit
// consecutive columns of the same row, so global loads of `x` are coalesced.
constexpr int kDiagColTile = 256;

template <typename T>
__global__ __launch_bounds__(kDiagColTile) void feature_gram_diag_kernel(const T* __restrict__ x,
                                                                        float* __restrict__ out,
                                                                        int64_t num_rows,
                                                                        int64_t num_cols) {
  const int64_t col = static_cast<int64_t>(blockIdx.x) * kDiagColTile + threadIdx.x;
  if (col >= num_cols) {
    return;
  }
  float acc = 0.f;
  for (int64_t row = 0; row < num_rows; ++row) {
    const float v = static_cast<float>(x[row * num_cols + col]);
    acc += v * v;
  }
  // Each (block, threadIdx.x) maps to a unique column, so no atomics needed.
  out[col] += acc;
}

// Each block handles one column-block (BxB output tile). Threads are launched
// with a fixed 16x16 layout to stay within the 1024-threads-per-block CUDA
// limit for any supported block_size; each thread accumulates a per_thread x
// per_thread tile of outputs in registers, striding by 16 along both axes.
constexpr int kBlockRowChunk = 16;
constexpr int kBlockThreadsPerDim = 16;
constexpr int kMaxBlockSize = 128;
constexpr int kMaxPerThread =
    (kMaxBlockSize + kBlockThreadsPerDim - 1) / kBlockThreadsPerDim;  // 8

template <typename T>
__global__ __launch_bounds__(kBlockThreadsPerDim* kBlockThreadsPerDim) void
feature_gram_block_diag_kernel(const T* __restrict__ x, float* __restrict__ out, int64_t num_rows,
                               int64_t num_cols, int64_t num_blocks, int64_t block_size) {
  const int64_t blk = blockIdx.x;
  if (blk >= num_blocks) {
    return;
  }
  const int64_t col_start = blk * block_size;
  const int B = static_cast<int>(block_size);
  const int per_thread = (B + kBlockThreadsPerDim - 1) / kBlockThreadsPerDim;

  __shared__ float tile[kBlockRowChunk][kMaxBlockSize];

  float acc[kMaxPerThread][kMaxPerThread];
#pragma unroll
  for (int a = 0; a < kMaxPerThread; ++a) {
#pragma unroll
    for (int b = 0; b < kMaxPerThread; ++b) {
      acc[a][b] = 0.f;
    }
  }

  for (int64_t row_base = 0; row_base < num_rows; row_base += kBlockRowChunk) {
    const int chunk = static_cast<int>(min(static_cast<int64_t>(kBlockRowChunk),
                                           num_rows - row_base));

    // Stage [chunk x B] into shared memory cooperatively. Linear thread id
    // walks the tile; consecutive threads hit consecutive columns of x for
    // coalesced loads.
    const int lin_tid = threadIdx.y * kBlockThreadsPerDim + threadIdx.x;
    const int num_threads = kBlockThreadsPerDim * kBlockThreadsPerDim;
    const int total_elems = chunk * B;
    for (int e = lin_tid; e < total_elems; e += num_threads) {
      const int r = e / B;
      const int c = e - r * B;
      const int64_t src_col = col_start + static_cast<int64_t>(c);
      float v = 0.f;
      if (src_col < num_cols) {
        v = static_cast<float>(x[(row_base + r) * num_cols + src_col]);
      }
      tile[r][c] = v;
    }
    __syncthreads();

    // Per-thread accumulate over the chunk. Each thread owns
    // {(i, j) : i = ti*16+threadIdx.y, j = tj*16+threadIdx.x, ti, tj in [0, per_thread)}.
    for (int ti = 0; ti < per_thread; ++ti) {
      const int i = ti * kBlockThreadsPerDim + threadIdx.y;
      if (i >= B) break;
      for (int tj = 0; tj < per_thread; ++tj) {
        const int j = tj * kBlockThreadsPerDim + threadIdx.x;
        if (j >= B) continue;
        float s = 0.f;
        for (int r = 0; r < chunk; ++r) {
          s += tile[r][i] * tile[r][j];
        }
        acc[ti][tj] += s;
      }
    }
    __syncthreads();
  }

  // Write back. Strided thread layout gives coalesced output writes along the
  // last (column-j) axis within each warp.
  for (int ti = 0; ti < per_thread; ++ti) {
    const int i = ti * kBlockThreadsPerDim + threadIdx.y;
    if (i >= B) break;
    for (int tj = 0; tj < per_thread; ++tj) {
      const int j = tj * kBlockThreadsPerDim + threadIdx.x;
      if (j >= B) continue;
      const int64_t out_idx = (blk * static_cast<int64_t>(B) + i) *
                                  static_cast<int64_t>(B) +
                              j;
      out[out_idx] += acc[ti][tj];
    }
  }
}

void check_input_dtype(const Tensor& x) {
  const DType dt = x.data.dtype;
  NVTE_CHECK(dt == DType::kBFloat16 || dt == DType::kFloat16 || dt == DType::kFloat32,
             "nvte_feature_gram: input x must be bfloat16, float16, or float32, got ",
             static_cast<int>(dt));
}

void check_output_fp32(const Tensor& out) {
  NVTE_CHECK(out.data.dtype == DType::kFloat32,
             "nvte_feature_gram: output buffer must be float32 (got dtype ",
             static_cast<int>(out.data.dtype), ").");
}

}  // namespace

}  // namespace transformer_engine

void nvte_feature_gram_diag(cudaStream_t stream, const NVTETensor x, NVTETensor out) {
  NVTE_API_CALL(nvte_feature_gram_diag);
  using namespace transformer_engine;

  const Tensor* x_t = convertNVTETensorCheck(x);
  Tensor* out_t = convertNVTETensor(out);

  check_input_dtype(*x_t);
  check_output_fp32(*out_t);

  NVTE_CHECK(x_t->data.shape.size() == 2,
             "nvte_feature_gram_diag: x must be 2D, got ndim=", x_t->data.shape.size());
  NVTE_CHECK(out_t->data.shape.size() == 1,
             "nvte_feature_gram_diag: out must be 1D, got ndim=", out_t->data.shape.size());

  const int64_t num_rows = static_cast<int64_t>(x_t->data.shape[0]);
  const int64_t num_cols = static_cast<int64_t>(x_t->data.shape[1]);
  NVTE_CHECK(static_cast<int64_t>(out_t->data.shape[0]) == num_cols,
             "nvte_feature_gram_diag: out length (", out_t->data.shape[0],
             ") must match x trailing dim (", num_cols, ").");

  if (num_rows == 0 || num_cols == 0) {
    return;
  }

  float* out_ptr = reinterpret_cast<float*>(out_t->data.dptr);
  const int64_t blocks_x = (num_cols + kDiagColTile - 1) / kDiagColTile;
  const dim3 grid(static_cast<unsigned int>(blocks_x), 1, 1);
  const dim3 block(kDiagColTile, 1, 1);

#define DISPATCH_DIAG(T)                                                                     \
  do {                                                                                       \
    const T* x_ptr = reinterpret_cast<const T*>(x_t->data.dptr);                             \
    feature_gram_diag_kernel<T><<<grid, block, 0, stream>>>(x_ptr, out_ptr, num_rows,        \
                                                            num_cols);                       \
  } while (0)

  switch (x_t->data.dtype) {
    case DType::kBFloat16:
      DISPATCH_DIAG(__nv_bfloat16);
      break;
    case DType::kFloat16:
      DISPATCH_DIAG(__half);
      break;
    case DType::kFloat32:
      DISPATCH_DIAG(float);
      break;
    default:
      NVTE_ERROR("nvte_feature_gram_diag: unreachable dtype.");
  }
#undef DISPATCH_DIAG

  NVTE_CHECK_CUDA(cudaGetLastError());
}

void nvte_feature_gram_block_diag(cudaStream_t stream, const NVTETensor x, NVTETensor out,
                                  int64_t block_size) {
  NVTE_API_CALL(nvte_feature_gram_block_diag);
  using namespace transformer_engine;

  const Tensor* x_t = convertNVTETensorCheck(x);
  Tensor* out_t = convertNVTETensor(out);

  check_input_dtype(*x_t);
  check_output_fp32(*out_t);

  NVTE_CHECK(x_t->data.shape.size() == 2,
             "nvte_feature_gram_block_diag: x must be 2D, got ndim=", x_t->data.shape.size());
  NVTE_CHECK(out_t->data.shape.size() == 3,
             "nvte_feature_gram_block_diag: out must be 3D, got ndim=", out_t->data.shape.size());
  NVTE_CHECK(block_size > 0 && block_size <= kMaxBlockSize,
             "nvte_feature_gram_block_diag: block_size must be in [1, ", kMaxBlockSize,
             "], got ", block_size);

  const int64_t num_rows = static_cast<int64_t>(x_t->data.shape[0]);
  const int64_t num_cols = static_cast<int64_t>(x_t->data.shape[1]);
  const int64_t num_blocks = static_cast<int64_t>(out_t->data.shape[0]);
  NVTE_CHECK(static_cast<int64_t>(out_t->data.shape[1]) == block_size &&
                 static_cast<int64_t>(out_t->data.shape[2]) == block_size,
             "nvte_feature_gram_block_diag: out shape must be [num_blocks, ", block_size, ", ",
             block_size, "].");
  NVTE_CHECK(num_blocks * block_size >= num_cols,
             "nvte_feature_gram_block_diag: num_blocks * block_size (", num_blocks * block_size,
             ") must be >= trailing dim of x (", num_cols, ").");

  if (num_rows == 0 || num_blocks == 0) {
    return;
  }

  float* out_ptr = reinterpret_cast<float*>(out_t->data.dptr);
  const dim3 grid(static_cast<unsigned int>(num_blocks), 1, 1);
  const dim3 block(kBlockThreadsPerDim, kBlockThreadsPerDim, 1);

#define DISPATCH_BLOCK_DIAG(T)                                                                 \
  do {                                                                                         \
    const T* x_ptr = reinterpret_cast<const T*>(x_t->data.dptr);                               \
    feature_gram_block_diag_kernel<T><<<grid, block, 0, stream>>>(x_ptr, out_ptr, num_rows,    \
                                                                  num_cols, num_blocks,        \
                                                                  block_size);                 \
  } while (0)

  switch (x_t->data.dtype) {
    case DType::kBFloat16:
      DISPATCH_BLOCK_DIAG(__nv_bfloat16);
      break;
    case DType::kFloat16:
      DISPATCH_BLOCK_DIAG(__half);
      break;
    case DType::kFloat32:
      DISPATCH_BLOCK_DIAG(float);
      break;
    default:
      NVTE_ERROR("nvte_feature_gram_block_diag: unreachable dtype.");
  }
#undef DISPATCH_BLOCK_DIAG

  NVTE_CHECK_CUDA(cudaGetLastError());
}
