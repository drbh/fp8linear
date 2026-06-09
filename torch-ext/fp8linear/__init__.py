from . import layers
from ._ops import ops
from .layers import Fp8Linear, fp8_linear, quantize_weight

quantize_fp8 = ops.quantize_fp8
fp8_gemm = ops.fp8_gemm

__all__ = [
    "ops",
    "layers",
    "Fp8Linear",
    "fp8_linear",
    "quantize_weight",
    "quantize_fp8",
    "fp8_gemm",
]
