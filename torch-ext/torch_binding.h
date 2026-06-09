#pragma once

#include <torch/torch.h>

// Dynamic per-tensor quantization of a fp16/bf16 activation tensor to
// float8_e4m3fn. Returns (xq, scale) where x ~= xq * scale.
//   x     : [..., K] fp16 or bf16, contiguous
//   xq    : same shape, float8_e4m3fn
//   scale : scalar fp32 tensor (amax / 448)
std::tuple<torch::Tensor, torch::Tensor> quantize_fp8(torch::Tensor x);

// Arch-aware weight quantization (offline). Default is per-channel e4m3 + fp32 [N]
// scale (the validated path). On Blackwell sm_100+ with FP8LINEAR_USE_MX=1 it emits
// MXFP8 (e4m3 + e8m0 [N,K/32] uint8 scales) instead -- experimental/unvalidated.
// The returned scale dtype tells fp8_linear which path to take.
std::tuple<torch::Tensor, torch::Tensor> quantize_weight(torch::Tensor weight);

// FP8 matmul with per-tensor dequant scales, computed on tensor cores:
//   out[M, N] = (xq @ wq^T) * x_scale * w_scale (+ bias)
//   xq      : [M, K] float8_e4m3fn, row-major
//   x_scale : scalar fp32
//   wq      : [N, K] float8_e4m3fn, row-major (i.e. transposed weight)
//   w_scale : scalar fp32
//   bias    : optional [N] fp16
//   out     : [M, N] fp16
torch::Tensor fp8_gemm(torch::Tensor xq,
                       torch::Tensor x_scale,
                       torch::Tensor wq,
                       torch::Tensor w_scale,
                       std::optional<torch::Tensor> bias);

// Fused dynamic-quant FP8 linear: quantizes x on the fly, then runs the FP8
// matmul against an already-quantized weight. This is the op that recovers the
// FP8 speedup that a naive (multi-pass) quantize would give back.
//   x       : [M, K] fp16/bf16
//   wq      : [N, K] float8_e4m3fn (pre-quantized weight)
//   w_scale : scalar fp32 weight scale
//   bias    : optional [N] fp16
torch::Tensor fp8_linear(torch::Tensor x,
                         torch::Tensor wq,
                         torch::Tensor w_scale,
                         std::optional<torch::Tensor> bias);

// ===== Blackwell (sm_120) MXFP8 microscaling path =====
// Block-scaled FP8 consumed natively by Blackwell tensor cores: e8m0 scale per
// 32-element block, applied in-matmul -> fp16 direct, no dequant epilogue.

// Returns (xq[M,K] e4m3, scales[M, K/32] uint8/e8m0).
std::tuple<torch::Tensor, torch::Tensor> quantize_mxfp8(torch::Tensor x);

// out[M,N] = (xq @ wq^T) with per-32-block e8m0 dequant applied in-core, fp16 out.
//   x_scales : [M, K/32] uint8 (e8m0)   w_scales : [N, K/32] uint8 (e8m0)
torch::Tensor mxfp8_gemm(torch::Tensor xq,
                         torch::Tensor x_scales,
                         torch::Tensor wq,
                         torch::Tensor w_scales,
                         std::optional<torch::Tensor> bias);

// Fused MXFP8 linear: dynamic block-scaled quant of x, then mxfp8_gemm.
torch::Tensor mxfp8_linear(torch::Tensor x,
                           torch::Tensor wq,
                           torch::Tensor w_scales,
                           std::optional<torch::Tensor> bias);

// ===== Blackwell NVFP4 path (e2m1 values, e4m3 scale per 16-element block) =====
// 2x the FP8 tensor-core rate on sm_100+/sm_120. Values are packed two-per-byte;
// scales use the same tiled cuBLASLt layout (scale-column count K/16).

// Returns (xq[M, K/2] uint8 packed e2m1, scales uint8/e4m3, tiled+padded).
std::tuple<torch::Tensor, torch::Tensor> quantize_nvfp4(torch::Tensor x);

// out[M,N] = (xq @ wq^T) with per-16 e4m3 dequant applied in-core, fp16 out.
torch::Tensor nvfp4_gemm(torch::Tensor xq,
                         torch::Tensor x_scales,
                         torch::Tensor wq,
                         torch::Tensor w_scales,
                         std::optional<torch::Tensor> bias);

// Fused NVFP4 linear: dynamic block-scaled quant of x, then nvfp4_gemm.
torch::Tensor nvfp4_linear(torch::Tensor x,
                           torch::Tensor wq,
                           torch::Tensor w_scales,
                           std::optional<torch::Tensor> bias);
