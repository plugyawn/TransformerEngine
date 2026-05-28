# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Optional extra wgrad factor accumulation for Transformer Engine modules.

This module exposes a typed contract that callers (e.g. Megatron Core) can use
to request additional wgrad-side statistics beside the ordinary main_grad. The
canonical use case is preconditioned optimizers (Newton-Muon, affine
LocoProp-S) that consume an extra Feature Gram factor C = X^T X formed from
the same logical 2D feature matrix used by the ordinary wgrad GEMM
G = dY^T X.

The contract is intentionally narrow:

  * Callers attach an :class:`ExtraWgradRequest` to a weight via the attribute
    ``weight._te_extra_wgrad``. The request carries pre-allocated accumulator
    buffers and a recipe describing the requested factor approximation.
  * TE invokes the kernels at the true wgrad feature site, i.e. immediately
    before the wgrad GEMM consumes ``X``. Accumulation is in-place into the
    caller-owned buffers and is CUDA-graph capture safe.
  * Heavy lifting (Cholesky/inverse, normalization, lifecycle, distributed
    routing) belongs to the caller. TE only sees a feature matrix and writes
    fp32 sums.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

import torch

import transformer_engine_torch as tex

from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage


__all__ = [
    "FACTOR_FEATURE_GRAM",
    "FACTOR_FEATURE_SUM",
    "FeatureGramRecipe",
    "ExtraWgradRequest",
    "get_extra_wgrad_request",
    "assert_no_extra_wgrad_factors_requested",
    "validate_extra_wgrad_factors",
    "maybe_accumulate_feature_factors",
    "wrap_wgrad_closure_with_feature_factors",
]


# Bitfield values defining which extra wgrad factors a caller requests.
FACTOR_FEATURE_GRAM: int = 1
FACTOR_FEATURE_SUM: int = 2
_SUPPORTED_FACTORS: int = FACTOR_FEATURE_GRAM | FACTOR_FEATURE_SUM


@dataclass(frozen=True)
class FeatureGramRecipe:
    """Describes how to form and store the feature Gram factor C = X^T X."""

    # Storage approximation. ``full`` accumulates the dense [N, N] matrix,
    # ``diag`` only the per-feature sums-of-squares, and ``block_diag`` the
    # per-column-block dense Gram matrices.
    approximation: str

    # Block edge length when ``approximation == "block_diag"``.
    block_size: int = 0

    # Working dtype to which ``X`` is cast before forming ``X^T X``. The two
    # supported modes are: ``"bf16_saved"`` (use the saved feature tensor as
    # provided, casting only if needed), and ``"fp32_cast"`` (explicitly cast
    # to fp32 before forming the product).
    source_dtype: str = "bf16_saved"

    # Accumulator dtype. Always fp32 in this release; reserved for future use.
    accumulation_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.approximation not in ("full", "diag", "block_diag"):
            raise ValueError(
                f"FeatureGramRecipe: unsupported approximation {self.approximation!r}; "
                "expected one of 'full', 'diag', 'block_diag'."
            )
        if self.approximation == "block_diag" and self.block_size <= 0:
            raise ValueError(
                "FeatureGramRecipe: block_size must be > 0 for block_diag approximation."
            )
        if self.source_dtype not in ("bf16_saved", "fp32_cast"):
            raise ValueError(
                f"FeatureGramRecipe: unsupported source_dtype {self.source_dtype!r}; "
                "expected 'bf16_saved' or 'fp32_cast'."
            )
        if self.accumulation_dtype is not torch.float32:
            raise ValueError(
                "FeatureGramRecipe: only fp32 accumulation is supported in this release."
            )


@dataclass
class ExtraWgradRequest:
    """Caller-supplied request for extra wgrad-side statistics on a weight.

    The lifecycle of the buffers is owned by the caller (typically Megatron).
    TE only writes into them at the wgrad feature site; finalization, refresh,
    distributed reduction, and consumption by the optimizer are all
    out-of-scope for TE.
    """

    # Bitfield of FACTOR_* values describing which factors are requested.
    factors: int

    # Recipe for the FEATURE_GRAM factor. Required iff FACTOR_FEATURE_GRAM is
    # set in ``factors``.
    feature_gram: Optional[FeatureGramRecipe]

    # Accumulator for X^T X. Shape depends on the recipe approximation:
    # ``[N]`` for diag, ``[num_blocks, B, B]`` for block_diag, ``[N, N]`` for
    # full. Dtype must be fp32. Required iff FACTOR_FEATURE_GRAM is set.
    gram_buffer: Optional[torch.Tensor]

    # Per-call accumulator for the row count of X. Shape ``[]`` (scalar), fp32.
    # Required iff FACTOR_FEATURE_GRAM is set. TE does not gate accumulation
    # on this value (no host sync); it is purely informational for callers.
    count_buffer: Optional[torch.Tensor]

    # Accumulator for sum_i X[i, :]. Shape ``[N]``, fp32. Required iff
    # FACTOR_FEATURE_SUM is set in ``factors``.
    sum_buffer: Optional[torch.Tensor] = None

    # When False, TE skips the accumulation for this microbatch. Useful for
    # callers that want to gate based on a step schedule without unbinding
    # the request.
    active: bool = True

    def __post_init__(self) -> None:
        if self.factors & ~_SUPPORTED_FACTORS:
            raise ValueError(
                f"ExtraWgradRequest: unsupported factor bits "
                f"{self.factors & ~_SUPPORTED_FACTORS:#x}."
            )
        if (self.factors & FACTOR_FEATURE_SUM) and not (self.factors & FACTOR_FEATURE_GRAM):
            raise ValueError(
                "ExtraWgradRequest: FEATURE_SUM requires FEATURE_GRAM in the "
                "same collection window."
            )
        if self.factors & FACTOR_FEATURE_GRAM:
            if self.feature_gram is None:
                raise ValueError("ExtraWgradRequest: FEATURE_GRAM requires a recipe.")
            if self.gram_buffer is None or self.count_buffer is None:
                raise ValueError(
                    "ExtraWgradRequest: FEATURE_GRAM requires gram_buffer and count_buffer."
                )
            if self.gram_buffer.dtype is not torch.float32:
                raise ValueError("ExtraWgradRequest: gram_buffer must be float32.")
            if self.count_buffer.dtype is not torch.float32:
                raise ValueError("ExtraWgradRequest: count_buffer must be float32.")
        if (self.factors & FACTOR_FEATURE_SUM) and self.sum_buffer is None:
            raise ValueError("ExtraWgradRequest: FEATURE_SUM requires sum_buffer.")


_REQUEST_ATTR = "_te_extra_wgrad"


def get_extra_wgrad_request(weight: Any) -> Optional[ExtraWgradRequest]:
    """Return the :class:`ExtraWgradRequest` attached to ``weight`` if any."""

    if weight is None:
        return None
    request = getattr(weight, _REQUEST_ATTR, None)
    if request is None:
        return None
    if not isinstance(request, ExtraWgradRequest):
        raise TypeError(
            f"weight.{_REQUEST_ATTR} must be an ExtraWgradRequest, got {type(request)!r}."
        )
    return request


def assert_no_extra_wgrad_factors_requested(weight: Any, module_name: str) -> None:
    """Fail closed for TE modules that do not implement extra wgrad factors."""

    if get_extra_wgrad_request(weight) is not None:
        raise NotImplementedError(
            f"{module_name} does not support extra wgrad factors. Per-weight "
            "FEATURE_GRAM collection must be implemented before enabling "
            "matrix optimizers for this module."
        )


def validate_extra_wgrad_factors(
    weight: Any,
    module_name: str,
    *,
    fuse_wgrad_accumulation: bool,
    requires_wgrad: bool,
) -> None:
    """Validate that an attached request is compatible with the TE call site."""

    if not requires_wgrad:
        return
    request = get_extra_wgrad_request(weight)
    if request is None:
        return
    if not fuse_wgrad_accumulation:
        raise NotImplementedError(
            f"{module_name} extra wgrad factors require fuse_wgrad_accumulation=True, "
            "matching the existing main_grad accumulation path."
        )


# Map from per-recipe source_dtype to a working torch dtype for the kernel
# input. fp32 always works; bf16/fp16 inputs are accepted by the kernels too.
_T = TypeVar("_T")


def _dequantize_if_needed(inputmat: Any, target_dtype: torch.dtype) -> torch.Tensor:
    """Return a dense torch.Tensor for the kernel, dequantizing on demand."""

    if isinstance(inputmat, (QuantizedTensor, QuantizedTensorStorage)):
        return inputmat.dequantize(dtype=target_dtype)
    return inputmat


def _materialize_feature_matrix(inputmat: Any, recipe: FeatureGramRecipe) -> torch.Tensor:
    """Project ``inputmat`` to the 2D dense feature matrix the kernels expect."""

    if recipe.source_dtype == "fp32_cast":
        x = _dequantize_if_needed(inputmat, torch.float32)
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)
    else:
        # source_dtype == "bf16_saved": prefer the saved working dtype. If the
        # tensor is FP8-quantized, dequantize to bf16 for the kernel.
        if isinstance(inputmat, (QuantizedTensor, QuantizedTensorStorage)):
            x = inputmat.dequantize(dtype=torch.bfloat16)
        else:
            x = inputmat
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def _accumulate_feature_gram(
    request: ExtraWgradRequest, x: torch.Tensor
) -> None:
    recipe = request.feature_gram
    assert recipe is not None  # validated in __post_init__
    gram = request.gram_buffer
    assert gram is not None

    if recipe.approximation == "diag":
        tex.feature_gram_diag(x, gram)
    elif recipe.approximation == "block_diag":
        tex.feature_gram_block_diag(x, gram, recipe.block_size)
    elif recipe.approximation == "full":
        # X^T X via the existing aten/cuBLAS path. ``addmm_(beta, alpha, m1, m2)``
        # forms ``gram = beta*gram + alpha*(m1 @ m2)`` in-place.
        if x.dtype is not torch.float32:
            x32 = x.to(torch.float32)
        else:
            x32 = x
        gram.addmm_(x32.t(), x32, beta=1.0, alpha=1.0)
    else:  # pragma: no cover - guarded by recipe __post_init__
        raise RuntimeError(f"Unreachable approximation {recipe.approximation!r}.")


def maybe_accumulate_feature_factors(weight: Any, inputmat: Any) -> None:
    """Accumulate any requested extra wgrad factors for ``weight`` at the wgrad site.

    Safe to call unconditionally; no-op when no request is attached or when
    the request is inactive. CUDA-graph capture safe: no host syncs, no
    Python control flow on tensor values.
    """

    request = get_extra_wgrad_request(weight)
    if request is None or not request.active:
        return
    if not (request.factors & FACTOR_FEATURE_GRAM):
        # FEATURE_SUM-only is rejected by ExtraWgradRequest.__post_init__, so
        # nothing remains to do here.
        return
    recipe = request.feature_gram
    assert recipe is not None
    x = _materialize_feature_matrix(inputmat, recipe)

    _accumulate_feature_gram(request, x)

    # Update count (number of feature rows seen this microbatch). Scalar
    # add_ is encoded into the kernel call and is CUDA-graph capture safe.
    count_buffer = request.count_buffer
    assert count_buffer is not None
    count_buffer.add_(float(x.shape[0]))

    if request.factors & FACTOR_FEATURE_SUM:
        sum_buffer = request.sum_buffer
        assert sum_buffer is not None
        if x.dtype is sum_buffer.dtype:
            sum_buffer.add_(x.sum(dim=0))
        else:
            sum_buffer.add_(x.sum(dim=0).to(sum_buffer.dtype))


WgradClosure = Callable[..., Any]


def wrap_wgrad_closure_with_feature_factors(
    weight: Any, wgrad_closure: WgradClosure
) -> WgradClosure:
    """Return a wgrad closure that also accumulates extra wgrad factors.

    Use when the wgrad GEMM is queued onto a :class:`WeightGradStore` for
    delayed execution: this lets the FEATURE_GRAM accumulation share the same
    microbatch ordering as the wgrad GEMM. If no extra factors are requested,
    the original closure is returned unchanged.
    """

    if get_extra_wgrad_request(weight) is None:
        return wgrad_closure

    def _wrapped(x: Any, *args: Any, **kwargs: Any) -> Any:
        maybe_accumulate_feature_factors(weight, x)
        return wgrad_closure(x, *args, **kwargs)

    return _wrapped
