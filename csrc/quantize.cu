// csrc/quantize.cu
//
// Fused per-row (per-token) dynamic quantization of fp16/bf16 activations to
// float8_e4m3fn. One CUDA kernel, one block per row: it computes the row's amax
// and writes the quantized row + its scale in a single launch -- no separate
// reduction pass and no host sync. Per-row scaling is both cheaper (no global
// reduction) and far more accurate than per-tensor for activations with outliers.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp8.h>
#include <torch/torch.h>

std::tuple<torch::Tensor, torch::Tensor> quantize_fp8(torch::Tensor x);

namespace {

constexpr float kE4M3Max = 448.0f;
constexpr int kThreads = 256;

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
