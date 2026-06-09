from . import layers
from ._ops import ops
from .layers import (
    Fp8Linear,
    fp8_linear,
    mxfp8_linear,
    quantize_,
    quantize_weight,
    quantize_weight_mxfp8,
)

quantize_fp8 = ops.quantize_fp8
fp8_gemm = ops.fp8_gemm
quantize_mxfp8 = ops.quantize_mxfp8
mxfp8_gemm = ops.mxfp8_gemm

__all__ = [
    "ops",
    "layers",
    "Fp8Linear",
    "fp8_linear",
    "quantize_",
    "quantize_weight",
    "quantize_fp8",
    "fp8_gemm",
    # Blackwell MXFP8 path
    "mxfp8_linear",
    "quantize_weight_mxfp8",
    "quantize_mxfp8",
    "mxfp8_gemm",
]
