// csrc/quantize.cu
//
// Fused per-row (per-token) dynamic quantization of fp16/bf16 activations to
// float8_e4m3fn. One CUDA kernel, one block per row: it computes the row's amax
// and writes the quantized row + its scale in a single launch -- no separate
// reduction pass and no host sync. Per-row scaling is both cheaper (no global
// reduction) and far more accurate than per-tensor for activations with outliers.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <algorithm>
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

// cuBLASLt block-scale-factor layout (Blackwell tcgen05): scales live in 512-byte
// tiles covering 128 rows x 4 scale-columns. Within a tile, scale (r, c) sits at
// (r%32)*16 + ((r%128)/32)*4 + (c%4); tiles are K-major (column-tile stride 512,
// row-tile stride 512*ncb). `ncb` = ceil(nblk/4) column tiles. Row/col padding to
// the tile grid is required; pad bytes are prefilled with 127 (scale = 1.0).
__device__ __forceinline__ int64_t sf_offset(int64_t r, int64_t c, int64_t ncb) {
  return (r >> 7) * (ncb * 512) + (c >> 2) * 512 + (r & 31) * 16 +
         ((r & 127) >> 5) * 4 + (c & 3);
}

// MXFP8: per-32-element block quantization to e4m3 with a shared power-of-two
// (e8m0) scale per block, the OCP microscaling format Blackwell tensor cores
// consume natively. One WARP per 32-element group: lane l owns element l, so
// loads/stores are fully coalesced and the block amax is a warp-shuffle
// reduction. (The previous thread-per-group serial loop was 5-10x slower than
// the rowwise fp8 quantizer and erased the MX matmul win on large-K shapes.)
// Scales are written directly in the tiled cuBLASLt layout described above.
template <typename scalar_t>
__global__ void quant_mxfp8_kernel(const scalar_t *__restrict__ x,
                                   __nv_fp8_storage_t *__restrict__ out,
                                   uint8_t *__restrict__ scales, int64_t M,
                                   int K, int nblk, int ncb) {
  const int64_t total_groups = M * (int64_t)nblk;
  const int warps_per_block = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int64_t gi = (int64_t)blockIdx.x * warps_per_block + warp;
       gi < total_groups; gi += (int64_t)gridDim.x * warps_per_block) {
    const int64_t row = gi / nblk;
    const int g = (int)(gi % nblk);
    const float val = static_cast<float>(x[row * K + (int64_t)g * kMxBlock + lane]);

    float amax = fabsf(val);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));

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
    if (lane == 0)
      scales[sf_offset(row, g, ncb)] = (uint8_t)e8;

    float q = fminf(fmaxf(val * x_inv, -kE4M3Max), kE4M3Max);
    out[row * K + (int64_t)g * kMxBlock + lane] =
        __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
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

// Returns (xq[M,K] e4m3, scales uint8/e8m0) in MXFP8 microscaling format. The
// scales tensor is 1-D, in cuBLASLt's tiled scale-factor layout padded to the
// 128-row x 4-col tile grid: ceil(M/128) * ceil((K/32)/4) * 512 bytes.
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
  const int64_t nrb = (M + 127) / 128;   // 128-row tiles
  const int64_t ncb = (nblk + 3) / 4;    // 4-wide scale-column tiles

  auto out = torch::empty_like(xc, xc.options().dtype(torch::kFloat8_e4m3fn));
  // Prefill padding with 127 (e8m0 scale = 1.0); 0xFF would be NaN.
  auto scales = torch::full({nrb * ncb * 512}, 127,
                            xc.options().dtype(torch::kByte));
  if (M == 0)
    return {out, scales};

  // One warp per 32-element group, grid-stride over all groups.
  const int64_t total_groups = M * nblk;
  const int warps_per_block = kThreads / 32;
  const int blocks = (int)std::min<int64_t>(
      (total_groups + warps_per_block - 1) / warps_per_block, 32768);

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_SWITCH(
      xc.scalar_type(), "quantize_mxfp8",
      AT_DISPATCH_CASE(torch::kHalf,
                       [&] {
                         quant_mxfp8_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
                             xc.data_ptr<scalar_t>(),
                             reinterpret_cast<__nv_fp8_storage_t *>(out.data_ptr()),
                             scales.data_ptr<uint8_t>(), M, (int)K, (int)nblk,
                             (int)ncb);
                       })
          AT_DISPATCH_CASE(torch::kBFloat16, [&] {
            quant_mxfp8_kernel<scalar_t><<<blocks, kThreads, 0, stream>>>(
                xc.data_ptr<scalar_t>(),
                reinterpret_cast<__nv_fp8_storage_t *>(out.data_ptr()),
                scales.data_ptr<uint8_t>(), M, (int)K, (int)nblk, (int)ncb);
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
  // MXFP8 is OPT-IN (FP8LINEAR_USE_MX=1) on Blackwell: the cuBLASLt MX scale-factor
  // layout is not yet validated -- on an RTX PRO 6000 it produced wrong output
  // (PSNR ~8 dB) and no speedup. Default stays on the working per-channel FP8 path.
  const int major = at::cuda::getCurrentDeviceProperties()->major;
  const bool use_mx = std::getenv("FP8LINEAR_USE_MX") != nullptr;
  if (major >= 10 && use_mx) {
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
