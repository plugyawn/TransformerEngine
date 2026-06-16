# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the extra wgrad factor accumulation surface.

Covers the [Common] CUDA kernels (diag and block_diag), the typed
ExtraWgradRequest contract, and end-to-end integration with the four
affine TE modules (Linear, LayerNormLinear, LayerNormMLP, GroupedLinear).
"""

from __future__ import annotations

import pytest
import torch

# Import transformer_engine.pytorch first; its __init__ globally loads
# libtransformer_engine.so with RTLD_GLOBAL so that subsequently importing
# transformer_engine_torch can resolve typeinfo and other symbols against
# the common library.
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.module.extra_wgrad import (
    FACTOR_FEATURE_GRAM,
    FACTOR_FEATURE_SUM,
    FACTOR_GRAD_GRAM,
    ExtraWgradRequest,
    FeatureGramRecipe,
    GradGramRecipe,
    get_extra_wgrad_request,
    maybe_accumulate_wgrad_factors,
)


CUDA_AVAILABLE = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA required")


# ---------------------------------------------------------------------------
# Kernel correctness: nvte_feature_gram_diag / nvte_feature_gram_block_diag.
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(64, 128), (512, 1024), (37, 257)])
def test_diag_kernel_matches_reference(dtype: torch.dtype, shape):
    torch.manual_seed(0)
    M, N = shape
    x = torch.randn(M, N, dtype=dtype, device="cuda")
    out = torch.zeros(N, dtype=torch.float32, device="cuda")

    tex.feature_gram_diag(x.contiguous(), out)

    ref = (x.float() * x.float()).sum(dim=0)
    tol = {torch.bfloat16: 5e-2, torch.float16: 5e-3, torch.float32: 1e-4}[dtype]
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


@requires_cuda
def test_diag_kernel_accumulates():
    torch.manual_seed(1)
    M, N = 256, 192
    x1 = torch.randn(M, N, dtype=torch.float32, device="cuda")
    x2 = torch.randn(M, N, dtype=torch.float32, device="cuda")
    out = torch.zeros(N, dtype=torch.float32, device="cuda")

    tex.feature_gram_diag(x1, out)
    tex.feature_gram_diag(x2, out)

    ref = (x1 * x1).sum(dim=0) + (x2 * x2).sum(dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@requires_cuda
@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_block_diag_kernel_matches_reference(block_size: int):
    torch.manual_seed(2)
    M = 200
    num_blocks = 3
    N = num_blocks * block_size
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(num_blocks, block_size, block_size, dtype=torch.float32, device="cuda")

    tex.feature_gram_block_diag(x.contiguous(), out, block_size)

    x_blocks = x.float().reshape(M, num_blocks, block_size).transpose(0, 1)  # [B, M, b]
    ref = torch.bmm(x_blocks.transpose(1, 2), x_blocks)  # [B, b, b]
    torch.testing.assert_close(out, ref, rtol=5e-2, atol=5e-2)


@requires_cuda
def test_block_diag_kernel_handles_padding():
    torch.manual_seed(3)
    M = 128
    block_size = 32
    num_blocks = 3
    N_actual = num_blocks * block_size - 7  # last block is short
    x = torch.randn(M, N_actual, dtype=torch.float32, device="cuda")
    out = torch.zeros(num_blocks, block_size, block_size, dtype=torch.float32, device="cuda")

    tex.feature_gram_block_diag(x.contiguous(), out, block_size)

    padded = torch.nn.functional.pad(x, (0, num_blocks * block_size - N_actual))
    x_blocks = padded.reshape(M, num_blocks, block_size).transpose(0, 1)
    ref = torch.bmm(x_blocks.transpose(1, 2), x_blocks)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_block_diag_kernel_accumulates():
    torch.manual_seed(4)
    M, B = 64, 32
    num_blocks = 2
    N = num_blocks * B
    x1 = torch.randn(M, N, dtype=torch.float32, device="cuda")
    x2 = torch.randn(M, N, dtype=torch.float32, device="cuda")
    out = torch.zeros(num_blocks, B, B, dtype=torch.float32, device="cuda")

    tex.feature_gram_block_diag(x1, out, B)
    tex.feature_gram_block_diag(x2, out, B)

    def ref_block(x):
        xb = x.reshape(M, num_blocks, B).transpose(0, 1)
        return torch.bmm(xb.transpose(1, 2), xb)

    torch.testing.assert_close(out, ref_block(x1) + ref_block(x2), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Python contract: ExtraWgradRequest + FeatureGramRecipe validation.
# ---------------------------------------------------------------------------


def test_recipe_rejects_bad_approximation():
    with pytest.raises(ValueError, match="approximation"):
        FeatureGramRecipe(approximation="sketch")


def test_recipe_block_diag_requires_block_size():
    with pytest.raises(ValueError, match="block_size"):
        FeatureGramRecipe(approximation="block_diag", block_size=0)


def test_request_rejects_sum_without_gram():
    with pytest.raises(ValueError, match="FEATURE_SUM requires FEATURE_GRAM"):
        ExtraWgradRequest(
            factors=FACTOR_FEATURE_SUM,
            feature_gram=None,
            gram_buffer=None,
            count_buffer=None,
            sum_buffer=torch.zeros(4, dtype=torch.float32),
        )


def test_request_rejects_missing_gram_buffer():
    with pytest.raises(ValueError, match="gram_buffer"):
        ExtraWgradRequest(
            factors=FACTOR_FEATURE_GRAM,
            feature_gram=FeatureGramRecipe(approximation="diag"),
            gram_buffer=None,
            count_buffer=torch.zeros((), dtype=torch.float32),
        )


def test_request_rejects_non_fp32_buffer():
    with pytest.raises(ValueError, match="float32"):
        ExtraWgradRequest(
            factors=FACTOR_FEATURE_GRAM,
            feature_gram=FeatureGramRecipe(approximation="diag"),
            gram_buffer=torch.zeros(8, dtype=torch.float16),
            count_buffer=torch.zeros((), dtype=torch.float32),
        )


def test_get_extra_wgrad_request_returns_none_when_unset():
    class W:
        pass

    assert get_extra_wgrad_request(W()) is None
    assert get_extra_wgrad_request(None) is None


# ---------------------------------------------------------------------------
# Helper accumulation: maybe_accumulate_wgrad_factors.
# ---------------------------------------------------------------------------


def _alloc_gram(*, approximation, block_size=0, dim):
    if approximation == "diag":
        return torch.zeros(dim, dtype=torch.float32, device="cuda")
    if approximation == "block_diag":
        num_blocks = (dim + block_size - 1) // block_size
        return torch.zeros(num_blocks, block_size, block_size, dtype=torch.float32, device="cuda")
    if approximation == "full":
        return torch.zeros(dim, dim, dtype=torch.float32, device="cuda")
    raise ValueError(approximation)


def _attach_request(
    weight,
    *,
    approximation,
    block_size=0,
    with_sum=False,
    with_grad=False,
    dim=None,
    grad_dim=None,
):
    dim = dim if dim is not None else weight.shape[0] if hasattr(weight, "shape") else 64
    grad_dim = grad_dim if grad_dim is not None else weight.shape[0] if hasattr(weight, "shape") else dim
    gram = _alloc_gram(approximation=approximation, block_size=block_size, dim=dim)
    grad_gram = (
        _alloc_gram(approximation=approximation, block_size=block_size, dim=grad_dim)
        if with_grad
        else None
    )
    factors = FACTOR_FEATURE_GRAM | (FACTOR_FEATURE_SUM if with_sum else 0)
    if with_grad:
        factors |= FACTOR_GRAD_GRAM
    request = ExtraWgradRequest(
        factors=factors,
        feature_gram=FeatureGramRecipe(approximation=approximation, block_size=block_size),
        gram_buffer=gram,
        count_buffer=torch.zeros((), dtype=torch.float32, device="cuda"),
        grad_gram=(
            GradGramRecipe(approximation=approximation, block_size=block_size)
            if with_grad
            else None
        ),
        grad_gram_buffer=grad_gram,
        grad_count_buffer=(
            torch.zeros((), dtype=torch.float32, device="cuda") if with_grad else None
        ),
        sum_buffer=(torch.zeros(dim, dtype=torch.float32, device="cuda") if with_sum else None),
    )
    weight._te_extra_wgrad = request
    return request


def _attach_grad_request(weight, *, approximation, block_size=0, grad_dim=None):
    grad_dim = grad_dim if grad_dim is not None else weight.shape[0] if hasattr(weight, "shape") else 64
    grad_gram = _alloc_gram(approximation=approximation, block_size=block_size, dim=grad_dim)
    request = ExtraWgradRequest(
        factors=FACTOR_GRAD_GRAM,
        feature_gram=None,
        gram_buffer=None,
        count_buffer=None,
        grad_gram=GradGramRecipe(approximation=approximation, block_size=block_size),
        grad_gram_buffer=grad_gram,
        grad_count_buffer=torch.zeros((), dtype=torch.float32, device="cuda"),
    )
    weight._te_extra_wgrad = request
    return request


@requires_cuda
def test_helper_diag_matches_reference():
    weight = torch.nn.Parameter(torch.empty(64, 32, device="cuda"))
    request = _attach_request(weight, approximation="diag", dim=128)
    x = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn(256, 64, dtype=torch.bfloat16, device="cuda")
    maybe_accumulate_wgrad_factors(weight, x, dy)

    ref = (x.float() ** 2).sum(dim=0)
    torch.testing.assert_close(request.gram_buffer, ref, rtol=5e-2, atol=5e-2)
    assert float(request.count_buffer.item()) == 256.0
    assert request.feature_rows == 256


@requires_cuda
def test_helper_full_matches_reference():
    weight = torch.nn.Parameter(torch.empty(64, 32, device="cuda"))
    request = _attach_request(weight, approximation="full", dim=48)
    x = torch.randn(200, 48, dtype=torch.float32, device="cuda")
    dy = torch.randn(200, 64, dtype=torch.float32, device="cuda")
    maybe_accumulate_wgrad_factors(weight, x, dy)

    ref = x.t() @ x
    torch.testing.assert_close(request.gram_buffer, ref, rtol=1e-4, atol=1e-4)
    assert request.feature_rows == 200


@requires_cuda
def test_helper_block_diag_matches_reference():
    weight = torch.nn.Parameter(torch.empty(64, 32, device="cuda"))
    B = 16
    request = _attach_request(weight, approximation="block_diag", block_size=B, dim=B * 3)
    x = torch.randn(128, B * 3, dtype=torch.float32, device="cuda")
    dy = torch.randn(128, 64, dtype=torch.float32, device="cuda")
    maybe_accumulate_wgrad_factors(weight, x, dy)

    xb = x.reshape(128, 3, B).transpose(0, 1)
    ref = torch.bmm(xb.transpose(1, 2), xb)
    torch.testing.assert_close(request.gram_buffer, ref, rtol=1e-4, atol=1e-4)
    assert request.feature_rows == 128


@requires_cuda
def test_helper_feature_sum_accumulates():
    weight = torch.nn.Parameter(torch.empty(64, 32, device="cuda"))
    request = _attach_request(weight, approximation="diag", with_sum=True, dim=64)
    x = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    dy = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    maybe_accumulate_wgrad_factors(weight, x, dy)

    torch.testing.assert_close(
        request.sum_buffer, x.float().sum(dim=0), rtol=5e-2, atol=5e-2
    )


@requires_cuda
def test_helper_no_request_is_noop():
    weight = torch.nn.Parameter(torch.empty(64, 32, device="cuda"))
    # No exception, no side effect.
    maybe_accumulate_wgrad_factors(
        weight, torch.randn(8, 32, device="cuda"), torch.randn(8, 64, device="cuda")
    )


@requires_cuda
def test_helper_inactive_is_noop():
    weight = torch.nn.Parameter(torch.empty(64, 32, device="cuda"))
    request = _attach_request(weight, approximation="diag", dim=32)
    request.active = False
    maybe_accumulate_wgrad_factors(
        weight, torch.randn(8, 32, device="cuda"), torch.randn(8, 64, device="cuda")
    )
    assert torch.all(request.gram_buffer == 0).item()
    assert request.feature_rows == 0


@requires_cuda
@pytest.mark.parametrize("approximation", ["diag", "full", "block_diag"])
def test_helper_grad_gram_matches_reference(approximation):
    weight = torch.nn.Parameter(torch.empty(64, 32, device="cuda"))
    block_size = 16 if approximation == "block_diag" else 0
    grad_dim = 48
    request = _attach_grad_request(
        weight, approximation=approximation, block_size=block_size, grad_dim=grad_dim
    )
    x = torch.randn(128, 32, dtype=torch.float32, device="cuda")
    dy = torch.randn(128, grad_dim, dtype=torch.float32, device="cuda")
    maybe_accumulate_wgrad_factors(weight, x, dy)

    if approximation == "diag":
        ref = (dy * dy).sum(dim=0)
    elif approximation == "full":
        ref = dy.t() @ dy
    else:
        padded_dim = ((grad_dim + block_size - 1) // block_size) * block_size
        padded = torch.nn.functional.pad(dy, (0, padded_dim - grad_dim))
        dy_blocks = padded.reshape(128, -1, block_size).transpose(0, 1)
        ref = torch.bmm(dy_blocks.transpose(1, 2), dy_blocks)
    torch.testing.assert_close(request.grad_gram_buffer, ref, rtol=1e-4, atol=1e-4)
    assert float(request.grad_count_buffer.item()) == 128.0
    assert request.grad_rows == 128


# ---------------------------------------------------------------------------
# Module integration.
# ---------------------------------------------------------------------------


def _wgrad_buffer_setup(module, weight_attr="weight", *, recipe, dim=None):
    weight = getattr(module, weight_attr)
    weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
    return _attach_request(weight, approximation=recipe[0], block_size=recipe[1], dim=dim)


@requires_cuda
def test_linear_diag_accumulates_at_wgrad_site():
    torch.manual_seed(10)
    in_features, out_features = 128, 64
    module = te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=True,
    ).cuda()
    module.weight.main_grad = torch.zeros_like(module.weight, dtype=torch.float32)
    request = _attach_request(
        module.weight, approximation="diag", with_grad=True, dim=in_features, grad_dim=out_features
    )

    x = torch.randn(64, in_features, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = module(x)
    y.sum().backward()

    ref = (x.detach().float() ** 2).sum(dim=0)
    torch.testing.assert_close(request.gram_buffer, ref, rtol=5e-2, atol=5e-2)
    assert float(request.count_buffer.item()) == 64.0
    torch.testing.assert_close(
        request.grad_gram_buffer,
        torch.full((out_features,), 64.0, dtype=torch.float32, device="cuda"),
        rtol=1e-4,
        atol=1e-4,
    )
    assert float(request.grad_count_buffer.item()) == 64.0
    assert request.feature_rows == 64
    assert request.grad_rows == 64


@requires_cuda
def test_linear_diag_accumulates_without_fused_main_grad():
    torch.manual_seed(13)
    in_features, out_features = 32, 32
    module = te.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=False,
    ).cuda()
    request = _attach_request(
        module.weight, approximation="diag", with_grad=True, dim=in_features, grad_dim=out_features
    )

    x = torch.randn(16, in_features, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = module(x)
    y.sum().backward()

    ref = (x.detach().float() ** 2).sum(dim=0)
    torch.testing.assert_close(request.gram_buffer, ref, rtol=5e-2, atol=5e-2)
    assert float(request.count_buffer.item()) == 16.0
    torch.testing.assert_close(
        request.grad_gram_buffer,
        torch.full((out_features,), 16.0, dtype=torch.float32, device="cuda"),
        rtol=1e-4,
        atol=1e-4,
    )
    assert float(request.grad_count_buffer.item()) == 16.0
    assert request.feature_rows == 16
    assert request.grad_rows == 16
    assert module.weight.grad is not None


@requires_cuda
def test_layernorm_linear_diag_accumulates():
    torch.manual_seed(11)
    hidden, out_features = 64, 96
    module = te.LayerNormLinear(
        in_features=hidden,
        out_features=out_features,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=True,
    ).cuda()
    request = _wgrad_buffer_setup(module, recipe=("diag", 0), dim=hidden)

    x = torch.randn(32, hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    module(x).sum().backward()

    # Post-LN feature is what the wgrad GEMM sees; we cannot easily compute a
    # closed-form reference here, so just assert the buffer was populated with
    # a strictly positive accumulation (all diag entries are sums of squares).
    assert torch.all(request.gram_buffer >= 0).item()
    assert request.gram_buffer.sum().item() > 0.0
    assert request.feature_rows == 32


@requires_cuda
def test_layernorm_mlp_accumulates_both_fc1_and_fc2():
    torch.manual_seed(12)
    hidden, ffn = 64, 256
    module = te.LayerNormMLP(
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        bias=False,
        params_dtype=torch.bfloat16,
        fuse_wgrad_accumulation=True,
    ).cuda()
    fc1_request = _wgrad_buffer_setup(
        module, weight_attr="fc1_weight", recipe=("diag", 0), dim=hidden
    )
    fc2_request = _wgrad_buffer_setup(
        module, weight_attr="fc2_weight", recipe=("diag", 0), dim=ffn
    )

    x = torch.randn(48, hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    module(x).sum().backward()

    assert fc1_request.gram_buffer.sum().item() > 0.0
    assert fc2_request.gram_buffer.sum().item() > 0.0
    assert fc1_request.feature_rows == 48
    assert fc2_request.feature_rows == 48


@requires_cuda
def test_grouped_linear_fail_closed():
    num_gemms = 2
    in_features, out_features = 32, 32
    module = te.GroupedLinear(
        num_gemms=num_gemms,
        in_features=in_features,
        out_features=out_features,
        bias=False,
        params_dtype=torch.bfloat16,
    ).cuda()
    weight = getattr(module, "weight0")
    _attach_request(weight, approximation="diag", dim=in_features)

    x = torch.randn(16, in_features, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="GroupedLinear"):
        module(x, [8, 8])


# ---------------------------------------------------------------------------
# CUDA-graph capture / replay parity.
# ---------------------------------------------------------------------------


@requires_cuda
def test_diag_kernel_cuda_graph_safe():
    """Capture and replay the diag accumulation; verify the buffer compounds."""
    torch.manual_seed(20)
    M, N = 256, 128
    out = torch.zeros(N, dtype=torch.float32, device="cuda")
    x = torch.randn(M, N, dtype=torch.float32, device="cuda")

    # Warm up.
    tex.feature_gram_diag(x, out)
    out.zero_()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            tex.feature_gram_diag(x, out)
    torch.cuda.current_stream().wait_stream(s)

    g.replay()
    g.replay()
    g.replay()
    torch.cuda.synchronize()

    ref = 3 * (x * x).sum(dim=0)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
