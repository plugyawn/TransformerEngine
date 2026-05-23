# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Optional extra wgrad factor accumulation for Transformer Engine modules."""

from __future__ import annotations

from typing import Any

import torch

from ..quantized_tensor import QuantizedTensor, QuantizedTensorStorage


_FEATURE_GRAM_FLAG = 1
_FEATURE_SUM_FLAG = 2
_SUPPORTED_FLAGS = _FEATURE_GRAM_FLAG | _FEATURE_SUM_FLAG


def _value(value: Any) -> Any:
    """Return enum values without importing Megatron enums into TE."""

    return getattr(value, "value", value)


def _flags_value(weight: Any) -> int:
    flags = getattr(weight, "_extra_wgrad_factors", 0)
    flags = _value(flags)
    if flags is None:
        return 0
    return int(flags)


def has_extra_wgrad_factors_requested(weight: Any) -> bool:
    """Return whether any extra wgrad factor was requested on a weight."""

    if weight is None:
        return False
    return _flags_value(weight) != 0


def has_feature_gram_requested(weight: Any) -> bool:
    """Return whether FEATURE_GRAM was requested on a weight."""

    if weight is None:
        return False
    return bool(_flags_value(weight) & _FEATURE_GRAM_FLAG)


def assert_no_extra_wgrad_factors_requested(weight: Any, module_name: str) -> None:
    """Fail closed for TE modules that do not implement extra wgrad factors."""

    if has_extra_wgrad_factors_requested(weight):
        raise NotImplementedError(
            f"{module_name} does not support extra wgrad factors yet. "
            "Per-weight FEATURE_GRAM collection must be implemented before enabling "
            "matrix optimizers for this module."
        )


def validate_extra_wgrad_factor_flags(weight: Any, module_name: str) -> None:
    """Reject unsupported or internally inconsistent extra-factor requests."""

    flags = _flags_value(weight)
    if flags & ~_SUPPORTED_FLAGS:
        raise NotImplementedError(
            f"{module_name} received unsupported extra wgrad factor bits: "
            f"{flags & ~_SUPPORTED_FLAGS}."
        )
    if (flags & _FEATURE_SUM_FLAG) and not (flags & _FEATURE_GRAM_FLAG):
        raise NotImplementedError(
            f"{module_name} FEATURE_SUM requires FEATURE_GRAM in the same collection window."
        )


def validate_extra_wgrad_factors_need_fused_main_grad(
    weight: Any,
    module_name: str,
    *,
    fuse_wgrad_accumulation: bool,
    requires_wgrad: bool,
) -> None:
    """Ensure TE extra factors follow the existing fused-main-grad contract."""

    if not requires_wgrad or not has_extra_wgrad_factors_requested(weight):
        return
    validate_extra_wgrad_factor_flags(weight, module_name)
    if not fuse_wgrad_accumulation:
        raise NotImplementedError(
            f"{module_name} extra wgrad factors require fuse_wgrad_accumulation=True, "
            "matching the existing main_grad accumulation path."
        )


def _recipe_field(recipe: Any, name: str, default: Any = None) -> Any:
    return _value(getattr(recipe, name, default))


def _is_quantized_tensor(tensor: Any) -> bool:
    return isinstance(tensor, (QuantizedTensor, QuantizedTensorStorage))


def _dequantize_feature_tensor(tensor: Any, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(tensor, QuantizedTensorStorage):
        return tensor.dequantize(dtype=dtype)
    if isinstance(tensor, QuantizedTensor):
        return tensor.dequantize(dtype=dtype)
    raise NotImplementedError(
        "FEATURE_GRAM source_dtype='fp8_dequant' requires a TE quantized wgrad "
        "feature tensor."
    )


def _cast_feature_input(inputmat: Any, recipe: Any) -> torch.Tensor:
    source_dtype = _recipe_field(recipe, "source_dtype")
    accumulation_dtype = getattr(recipe, "accumulation_dtype", torch.float32)

    if source_dtype not in ("bf16_saved", "fp32_cast", "fp8_dequant"):
        raise ValueError(f"Unsupported FEATURE_GRAM source dtype: {source_dtype!r}")
    if source_dtype == "fp8_dequant":
        return _dequantize_feature_tensor(inputmat, accumulation_dtype)
    if _is_quantized_tensor(inputmat):
        raise NotImplementedError(
            "FEATURE_GRAM source_dtype must be 'fp8_dequant' when the TE wgrad "
            "feature tensor is quantized."
        )
    if source_dtype == "fp32_cast":
        return inputmat.to(torch.float32)
    return inputmat.to(accumulation_dtype)


def _accumulate_diag_feature_gram(gram: torch.Tensor, x: torch.Tensor) -> None:
    if gram.is_cuda and x.is_cuda:
        try:
            from emerging_optimizers.triton_kernels.feature_gram import diag_feature_gram_reduce

            diag_feature_gram_reduce(x, out=gram, accumulate=True)
            return
        except Exception:
            pass
    gram.add_(torch.sum(x * x, dim=0))


def _accumulate_block_diag_feature_gram(gram: torch.Tensor, x: torch.Tensor, block_size: int) -> None:
    if gram.ndim != 3 or gram.shape[-1] != gram.shape[-2]:
        raise RuntimeError("block_diag FEATURE_GRAM buffer must have shape [num_blocks, b, b].")
    num_blocks = gram.shape[0]
    if gram.shape[-1] != block_size:
        raise RuntimeError(
            f"block_diag FEATURE_GRAM buffer block size {gram.shape[-1]} does not match "
            f"recipe block size {block_size}."
        )
    padded_dim = num_blocks * block_size
    if x.shape[-1] > padded_dim:
        raise RuntimeError(
            f"block_diag FEATURE_GRAM buffer covers {padded_dim} features, got {x.shape[-1]}."
        )
    if x.shape[-1] < padded_dim:
        x = torch.nn.functional.pad(x, (0, padded_dim - x.shape[-1]))
    x_blocks = x.reshape(x.shape[0], num_blocks, block_size).transpose(0, 1)
    gram.add_(torch.bmm(x_blocks.transpose(1, 2), x_blocks))


@torch.no_grad()
def maybe_accumulate_feature_gram(weight: Any, inputmat: Any) -> None:
    """Accumulate ``X.T @ X`` for a TE linear weight when requested.

    The caller must pass the same logical 2D feature matrix used for the
    ordinary wgrad GEMM, after TE's required gather/reshape/usage decisions.
    This function stores raw sums plus row count; normalization is optimizer-side.
    """

    if has_extra_wgrad_factors_requested(weight):
        validate_extra_wgrad_factor_flags(weight, "TransformerEngine")
    if not has_feature_gram_requested(weight):
        return
    if not getattr(weight, "_feature_gram_active", True):
        return

    recipe = getattr(weight, "_feature_gram_recipe", None)
    if recipe is None:
        raise RuntimeError("FEATURE_GRAM requested without a FeatureGramRecipe.")
    if not hasattr(weight, "main_grad_feature_gram"):
        raise RuntimeError("FEATURE_GRAM requested without a main_grad_feature_gram buffer.")
    if not hasattr(weight, "main_grad_feature_count"):
        raise RuntimeError("FEATURE_GRAM requested without a main_grad_feature_count buffer.")

    x = _cast_feature_input(inputmat, recipe)
    if x.dim() != 2:
        x = x.reshape(-1, x.shape[-1])

    token_sample_size = _recipe_field(recipe, "token_sample_size")
    if token_sample_size is not None:
        remaining = int(token_sample_size) - int(weight.main_grad_feature_count.item())
        if remaining <= 0:
            return
        if x.shape[0] > remaining:
            x = x[:remaining]

    approximation = _recipe_field(recipe, "approximation")
    gram = weight.main_grad_feature_gram
    if approximation == "full":
        gram.add_(x.t().matmul(x))
    elif approximation == "diag":
        _accumulate_diag_feature_gram(gram, x)
    elif approximation == "block_diag":
        _accumulate_block_diag_feature_gram(
            gram, x, int(_recipe_field(recipe, "block_size", 128))
        )
    else:
        raise NotImplementedError(
            f"FEATURE_GRAM approximation {approximation!r} is not implemented in TE yet."
        )

    weight.main_grad_feature_count.add_(float(x.shape[0]))
    weight._feature_gram_finalized = not getattr(
        weight, "_feature_gram_finalization_required", False
    )

    if _flags_value(weight) & _FEATURE_SUM_FLAG:
        if not hasattr(weight, "main_grad_feature_sum"):
            raise RuntimeError("FEATURE_SUM requested without a main_grad_feature_sum buffer.")
        weight.main_grad_feature_sum.add_(x.sum(dim=0))
