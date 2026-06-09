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


def quantize_weight(weight: torch.Tensor):
    """Per-channel (per-output-row) quantize a [N, K] weight to e4m3.

    Returns (wq, w_scale) with weight[n] ~= wq[n] * w_scale[n]; w_scale is [N] fp32.
    """
    amax = weight.detach().abs().amax(dim=1).float()  # [N]
    scale = (amax / E4M3_MAX).clamp_min(1e-12)  # [N]
    wq = (
        (weight.float() / scale[:, None])
        .clamp_(-E4M3_MAX, E4M3_MAX)
        .to(torch.float8_e4m3fn)
    )
    return wq, scale


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
