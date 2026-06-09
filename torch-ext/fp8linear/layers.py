from __future__ import annotations

import torch
from torch import nn

from ._ops import ops, add_op_namespace_prefix

E4M3_MAX = 448.0


# torch.compile support: fake/meta impls so Dynamo can trace shapes through the
# ops without a graph break.
@torch.library.register_fake(add_op_namespace_prefix("fp8_linear"))
def _fp8_linear_fake(x, wq, w_scale, bias=None):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.float16)


@torch.library.register_fake(add_op_namespace_prefix("quantize_fp8"))
def _quantize_fp8_fake(x):
    xq = x.new_empty(x.shape, dtype=torch.float8_e4m3fn)
    scale = x.new_empty((x.shape[0],), dtype=torch.float32)
    return xq, scale


@torch.library.register_fake(add_op_namespace_prefix("fp8_gemm"))
def _fp8_gemm_fake(xq, x_scale, wq, w_scale, bias=None):
    return xq.new_empty((xq.shape[0], wq.shape[0]), dtype=torch.float16)


@torch.library.register_fake(add_op_namespace_prefix("quantize_mxfp8"))
def _quantize_mxfp8_fake(x):
    xq = x.new_empty(x.shape, dtype=torch.float8_e4m3fn)
    # Tiled cuBLASLt scale layout: 512B tiles over ceil(M/128) x ceil((K/32)/4).
    nrb = (x.shape[0] + 127) // 128
    ncb = (x.shape[1] // 32 + 3) // 4
    scales = x.new_empty((nrb * ncb * 512,), dtype=torch.uint8)
    return xq, scales


@torch.library.register_fake(add_op_namespace_prefix("mxfp8_gemm"))
def _mxfp8_gemm_fake(xq, x_scales, wq, w_scales, bias=None):
    return xq.new_empty((xq.shape[0], wq.shape[0]), dtype=torch.float16)


@torch.library.register_fake(add_op_namespace_prefix("mxfp8_linear"))
def _mxfp8_linear_fake(x, wq, w_scales, bias=None):
    return x.new_empty((*x.shape[:-1], wq.shape[0]), dtype=torch.float16)


MX_BLOCK = 32


def quantize_weight_mxfp8(weight: torch.Tensor):
    """MXFP8 block-quantize a [N, K] weight (offline) via the CUDA kernel.

    Returns (wq[N,K] e4m3, scales uint8/e8m0) with the scales already in the tiled
    cuBLASLt scale-factor layout that `mxfp8_gemm` consumes.
    """
    w = weight.detach()
    if w.dtype not in (torch.float16, torch.bfloat16):
        w = w.to(torch.bfloat16)
    return ops.quantize_mxfp8(w.contiguous())


def mxfp8_linear(x, wq, w_scales, bias=None):
    """MXFP8 linear (Blackwell sm_120+). `wq`/`w_scales` from quantize_weight_mxfp8."""
    return ops.mxfp8_linear(x, wq, w_scales, bias)


def quantize_weight(weight: torch.Tensor):
    """Arch-aware weight quantization (the choice lives in C++ `ops.quantize_weight`).

    Returns (wq, w_scale). On Blackwell (sm_100+) w_scale is uint8/e8m0 block scales
    (MXFP8); otherwise fp32 per-channel scales. `fp8_linear` dispatches on that dtype.
    """
    return ops.quantize_weight(weight)


def fp8_linear(x, wq, w_scale, bias=None):
    """FP8 linear op for pre-quantized weights (`wq`/`w_scale` from quantize_weight)."""
    return ops.fp8_linear(x, wq, w_scale, bias)


@torch.no_grad()
def quantize_(module: nn.Module) -> int:
    """In-place: quantize every eligible nn.Linear weight in `module` to e4m3.

    The quantized weight replaces `linear.weight` (as a buffer) and the per-channel
    scale is stored as `linear.weight_scale`. Run this ONCE before `kernelize()` so
    the stateless `Fp8Linear` layer below never re-quantizes the weight per forward.
    Returns the number of layers converted.
    """
    n = 0
    for child in module.modules():
        if (
            isinstance(child, nn.Linear)
            and child.in_features % 16 == 0
            and getattr(child, "weight", None) is not None
            and child.weight.dtype != torch.float8_e4m3fn
        ):
            wq, scale = quantize_weight(child.weight.data)
            del child._parameters["weight"]
            child.register_buffer("weight", wq)
            child.register_buffer("weight_scale", scale)
            n += 1
    return n


class Fp8Linear(nn.Module):
    """Stateless kernel layer: replaces an `nn.Linear`'s forward with an FP8 matmul.

    Per the kernels layer contract this layer is pure -- no constructor, no state of
    its own. `kernelize()` grafts this `forward` onto a module that has already been
    quantized by `quantize_()`, so it reads a pre-quantized e4m3 `weight` and the
    per-channel `weight_scale` from the host (declared below as type annotations).
    Quantizing the weight once -- rather than per forward -- is what keeps the kernel
    a net speedup; activations are still quantized per token by the fused op.
    """

    # Member variables expected from the (quantized) host module.
    weight: torch.Tensor  # e4m3 [N, K]
    weight_scale: torch.Tensor  # fp32 [N]
    bias: torch.Tensor | None

    # Allowed class-variable exceptions to the no-state rule.
    has_backward: bool = False
    can_torch_compile: bool = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = ops.fp8_linear(x, self.weight, self.weight_scale, self.bias)
        return out.to(x.dtype)
