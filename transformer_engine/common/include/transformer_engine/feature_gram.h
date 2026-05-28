/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file feature_gram.h
 *  \brief Functions for accumulating feature Gram statistics.
 *
 *  These routines compute the accumulating-into-buffer feature Gram factor
 *  \f$C \mathrel{+}= X^\top X\f$ that preconditioned optimizers consume beside
 *  the ordinary weight gradient \f$G = dY^\top X\f$. The input \f$X\f$ is the
 *  same logical 2D feature matrix that the wgrad GEMM consumes; the output is
 *  always accumulated in fp32. Two output factor shapes are supported:
 *  - **Diagonal**: out has shape \f$[N]\f$ and accumulates
 *    \f$\mathrm{out}[j] \mathrel{+}= \sum_i X_{ij}^2\f$.
 *  - **Block-diagonal**: out has shape \f$[\mathrm{num\_blocks}, B, B]\f$ and
 *    accumulates the per-column-block Gram
 *    \f$\mathrm{out}[b] \mathrel{+}= X_b^\top X_b\f$ where \f$X_b\f$ is the
 *    column slice \f$X[:, bB : (b+1)B]\f$. If the feature dimension is not a
 *    multiple of \f$B\f$, out-of-range columns are treated as zero. The full
 *    \f$X^\top X\f$ case is intentionally not exposed here -- callers should
 *    use the existing cuBLAS GEMM path.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_FEATURE_GRAM_H_
#define TRANSFORMER_ENGINE_COMMON_FEATURE_GRAM_H_

#include <cuda_runtime.h>
#include <stdint.h>

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Accumulate the diagonal of \f$X^\top X\f$ into \p out.
 *
 *  \p x must be a 2D contiguous tensor of shape \f$[M, N]\f$ with dtype
 *  bfloat16, float16, or float32. \p out must be a 1D fp32 tensor of
 *  shape \f$[N]\f$. The operation is performed on \p stream and is
 *  CUDA-graph capture safe.
 *
 *  \param[in]     stream  CUDA stream.
 *  \param[in]     x       Input feature matrix, shape [M, N].
 *  \param[in,out] out     fp32 accumulator, shape [N].
 */
void nvte_feature_gram_diag(cudaStream_t stream, const NVTETensor x, NVTETensor out);

/*! \brief Accumulate per-block Gram matrices \f$X_b^\top X_b\f$ into \p out.
 *
 *  \p x must be a 2D contiguous tensor of shape \f$[M, N]\f$ with dtype
 *  bfloat16, float16, or float32. \p out must be a 3D fp32 tensor of
 *  shape \f$[\mathrm{num\_blocks}, B, B]\f$ where
 *  \f$\mathrm{num\_blocks} \cdot B \geq N\f$. Padding columns
 *  \f$[N, \mathrm{num\_blocks} \cdot B)\f$ are treated as zero.
 *
 *  \param[in]     stream      CUDA stream.
 *  \param[in]     x           Input feature matrix, shape [M, N].
 *  \param[in,out] out         fp32 accumulator, shape [num_blocks, B, B].
 *  \param[in]     block_size  Per-block edge length \f$B\f$.
 */
void nvte_feature_gram_block_diag(cudaStream_t stream, const NVTETensor x, NVTETensor out,
                                  int64_t block_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_COMMON_FEATURE_GRAM_H_
