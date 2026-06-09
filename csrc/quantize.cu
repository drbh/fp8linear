// csrc/quantize.cu
//
// Fused per-row (per-token) dynamic quantization of fp16/bf16 activations to
// float8_e4m3fn. One CUDA kernel, one block per row: it computes the row's amax
// and writes the quantized row + its scale in a single launch -- no separate
// reduction pass and no host sync. Per-row scaling is both cheaper (no global
// reduction) and far more accurate than per-tensor for activations with outliers.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdlib>
#include <cublas_v2.h>  // for CUBLAS_VERSION (MX path gated to cuBLAS >= 12.8)
#include <cuda_fp8.h>
#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> quantize_fp8(torch::Tensor x);
std::tuple<torch::Tensor, torch::Tensor> quantize_mxfp8(torch::Tensor x);
std::tuple<torch::Tensor, torch::Tensor> quantize_weight(torch::Tensor weight);

namespace {

constexpr float kE4M3Max = 448.0f;
constexpr int kThreads = 256;
constexpr int kMxBlock = 32;  // MXFP8 block size (OCP microscaling)

__device__ __forceinline__ float block_reduce_max(float v) {
  __shared__ float s[kThreads / 32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
  if (lane == 0)
    s[wid] = v;
  __syncthreads();
  v = (threadIdx.x < (blockDim.x + 31) / 32) ? s[lane] : 0.0f;
  if (wid == 0) {
    for (int o = 16; o > 0; o >>= 1)
      v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
  }
  return v;
}

template <typename scalar_t>
__global__ void quant_rowwise_kernel(const scalar_t *__restrict__ x,
                                     __nv_fp8_storage_t *__restrict__ out,
                                     float *__restrict__ scale, int K) {
  const int64_t row = blockIdx.x;
  const int64_t base = row * K;

  float amax = 0.0f;
  for (int j = threadIdx.x; j < K; j += blockDim.x)
    amax = fmaxf(amax, fabsf(static_cast<float>(x[base + j])));
  amax = block_reduce_max(amax);

  __shared__ float s_inv;
  __shared__ float s_scale;
  if (threadIdx.x == 0) {
    float sc = fmaxf(amax / kE4M3Max, 1e-12f);
    s_scale = sc;
    s_inv = 1.0f / sc;
    scale[row] = sc;
  }
  __syncthreads();

  const float inv = s_inv;
  for (int j = threadIdx.x; j < K; j += blockDim.x) {
    float v = static_cast<float>(x[base + j]) * inv;
    v = fminf(fmaxf(v, -kE4M3Max), kE4M3Max);
    out[base + j] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  }
}

// MXFP8: per-32-element block quantization to e4m3 with a shared power-of-two
// (e8m0) scale per block, the OCP microscaling format Blackwell tensor cores
// consume natively. One block per row; each thread owns 32-element groups.
template <typename scalar_t>
__global__ void quant_mxfp8_kernel(const scalar_t *__restrict__ x,
                                   __nv_fp8_storage_t *__restrict__ out,
                                   uint8_t *__restrict__ scales, int K,
                                   int nblk) {
  const int64_t row = blockIdx.x;
  for (int g = threadIdx.x; g < nblk; g += blockDim.x) {
    const int64_t base = row * K + (int64_t)g * kMxBlock;
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kMxBlock; ++i)
      amax = fmaxf(amax, fabsf(static_cast<float>(x[base + i])));

    int e8;       // biased (bias 127) exponent of the block scale X = 2^(Xexp)
    float x_inv;  // 1 / X, maps the block into e4m3 range
    if (amax > 0.0f) {
      // X = 2^(floor(log2(amax)) - 8); e4m3 max normal exponent is 8.
      int xexp = (int)floorf(log2f(amax)) - 8;
      e8 = xexp + 127;
      e8 = e8 < 0 ? 0 : (e8 > 254 ? 254 : e8);
      x_inv = exp2f(-(float)(e8 - 127));
    } else {
      e8 = 127;  // X = 1, values are zero anyway
      x_inv = 1.0f;
    }
    scales[row * nblk + g] = (uint8_t)e8;
#pragma unroll
    for (int i = 0; i < kMxBlock; ++i) {
      float q = static_cast<float>(x[base + i]) * x_inv;
      q = fminf(fmaxf(q, -kE4M3Max), kE4M3Max);
      out[base + i] = __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
    }
  }
}

} // namespace

// Returns (xq[M,K] e4m3, x_scale[M] fp32) with x[m] ~= xq[m] * x_scale[m].
std::tuple<torch::Tensor, torch::Tensor> quantize_fp8(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kHalf ||
                  x.scalar_type() == torch::kBFloat16,
              "x must be fp16 or bf16");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
  auto xc = x.contiguous();
  const int64_t M = xc.size(0);
  const int64_t K = xc.size(1);

  auto out = torch::empty_like(xc, xc.options().dtype(torch::kFloat8_e4m3fn));
  auto scale = torch::empty({M}, xc.options().dtype(torch::kFloat));
  if (M == 0)
    return {out, scale};

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_SWITCH(
      xc.scalar_type(), "quantize_fp8",
      AT_DISPATCH_CASE(torch::kHalf,
                       [&] {
                         quant_rowwise_kernel<scalar_t>
                             <<<M, kThreads, 0, stream>>>(
                                 xc.data_ptr<scalar_t>(),
                                 reinterpret_cast<__nv_fp8_storage_t *>(
                                     out.data_ptr()),
                                 scale.data_ptr<float>(), (int)K);
                       })
          AT_DISPATCH_CASE(torch::kBFloat16, [&] {
            quant_rowwise_kernel<scalar_t><<<M, kThreads, 0, stream>>>(
                xc.data_ptr<scalar_t>(),
                reinterpret_cast<__nv_fp8_storage_t *>(out.data_ptr()),
                scale.data_ptr<float>(), (int)K);
          }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, scale};
}

// Returns (xq[M,K] e4m3, scales[M, K/32] uint8/e8m0) in MXFP8 microscaling format.
std::tuple<torch::Tensor, torch::Tensor> quantize_mxfp8(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be on CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kHalf ||
                  x.scalar_type() == torch::kBFloat16,
              "x must be fp16 or bf16");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [M, K]");
  auto xc = x.contiguous();
  const int64_t M = xc.size(0);
  const int64_t K = xc.size(1);
  TORCH_CHECK(K % kMxBlock == 0, "K must be a multiple of 32 for MXFP8, got ", K);
  const int64_t nblk = K / kMxBlock;

  auto out = torch::empty_like(xc, xc.options().dtype(torch::kFloat8_e4m3fn));
  auto scales = torch::empty({M, nblk}, xc.options().dtype(torch::kByte));
  if (M == 0)
    return {out, scales};

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_SWITCH(
      xc.scalar_type(), "quantize_mxfp8",
      AT_DISPATCH_CASE(torch::kHalf,
                       [&] {
                         quant_mxfp8_kernel<scalar_t><<<M, kThreads, 0, stream>>>(
                             xc.data_ptr<scalar_t>(),
                             reinterpret_cast<__nv_fp8_storage_t *>(out.data_ptr()),
                             scales.data_ptr<uint8_t>(), (int)K, (int)nblk);
                       })
          AT_DISPATCH_CASE(torch::kBFloat16, [&] {
            quant_mxfp8_kernel<scalar_t><<<M, kThreads, 0, stream>>>(
                xc.data_ptr<scalar_t>(),
                reinterpret_cast<__nv_fp8_storage_t *>(out.data_ptr()),
                scales.data_ptr<uint8_t>(), (int)K, (int)nblk);
          }));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, scales};
}

// Arch-aware weight quantization (run once, offline). On Blackwell (sm_100+) it
// emits MXFP8 (e4m3 + e8m0 block scales) so fp8_linear takes the native block-
// scaled path; otherwise per-channel e4m3 + fp32 scale. The returned scale's dtype
// (uint8 vs fp32) is what fp8_linear later dispatches on. Set FP8LINEAR_NO_MX=1 to
// force the per-channel path even on Blackwell (escape hatch while MX is validated).
std::tuple<torch::Tensor, torch::Tensor> quantize_weight(torch::Tensor weight) {
  TORCH_CHECK(weight.is_cuda() && weight.dim() == 2, "weight must be a 2D CUDA tensor");

#if CUBLAS_VERSION >= 120800
  // MXFP8 only on Blackwell AND only in cu128+ builds (the MX matmul needs it).
  const int major = at::cuda::getCurrentDeviceProperties()->major;
  const bool force_fp8 = std::getenv("FP8LINEAR_NO_MX") != nullptr;
  if (major >= 10 && !force_fp8) {
    auto w = (weight.scalar_type() == torch::kFloat) ? weight.to(torch::kBFloat16)
                                                     : weight;
    return quantize_mxfp8(w);
  }
#endif

  // Per-channel (per-output-row) e4m3.
  auto amax = weight.detach().abs().amax(1).to(torch::kFloat);
  auto scale = (amax / kE4M3Max).clamp_min(1e-12);
  auto wq = (weight.to(torch::kFloat) / scale.unsqueeze(1))
                .clamp(-kE4M3Max, kE4M3Max)
                .to(torch::kFloat8_e4m3fn);
  return {wq, scale};
}
