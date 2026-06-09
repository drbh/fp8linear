// csrc/fp8_gemm.cu
//
// Rowwise FP8 (e4m3) linear on tensor cores. CUDA 12.8 cuBLASLt only exposes
// per-tensor (scalar) or MX block scaling -- no fp32 outer-vector mode -- so we
// get per-token x per-channel scaling by running the FP8 matmul *unscaled* into
// an fp32 accumulator, then applying the dequant (x_scale[m] * w_scale[n]) and
// bias in a single fused epilogue kernel that writes fp16.
//
// out[M,N] = xq[M,K] @ wq[N,K]^T  (TN layout, see below); the matmul desc /
// layouts / heuristic algorithm are cached per shape.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/torch.h>

#include <mutex>
#include <unordered_map>

std::tuple<torch::Tensor, torch::Tensor> quantize_fp8(torch::Tensor x);
torch::Tensor fp8_gemm(torch::Tensor xq, torch::Tensor x_scale,
                       torch::Tensor wq, torch::Tensor w_scale,
                       std::optional<torch::Tensor> bias);
torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor wq,
                         torch::Tensor w_scale,
                         std::optional<torch::Tensor> bias);

namespace {

#define CUBLAS_CHECK(expr)                                                     \
  do {                                                                         \
    cublasStatus_t status_ = (expr);                                           \
    TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS,                              \
                "cuBLASLt error ", (int)status_, " at " #expr);                \
  } while (0)

constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;

cublasLtHandle_t get_lt_handle() {
  static cublasLtHandle_t handle = nullptr;
  if (handle == nullptr)
    CUBLAS_CHECK(cublasLtCreate(&handle));
  return handle;
}

// out[m,n] = acc[m,n] * x_scale[m] * w_scale[n] (+ bias[n]); acc is row-major
// [M,N] (the col-major [N,M] the matmul produced aliases the same memory).
__global__ void dequant_bias_kernel(const float *__restrict__ acc,
                                    const float *__restrict__ x_scale,
                                    const float *__restrict__ w_scale,
                                    const __half *__restrict__ bias,
                                    __half *__restrict__ out, int64_t M,
                                    int64_t N) {
  const int64_t total = M * N;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += gridDim.x * blockDim.x) {
    const int64_t m = i / N, n = i % N;
    float v = acc[i] * x_scale[m] * w_scale[n];
    if (bias != nullptr)
      v += static_cast<float>(bias[n]);
    out[i] = __float2half(v);
  }
}

struct Plan {
  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
};
struct Key {
  int M, N, K;
  bool operator==(const Key &o) const {
    return M == o.M && N == o.N && K == o.K;
  }
};
struct KeyHash {
  size_t operator()(const Key &k) const {
    return ((size_t)k.M * 73856093) ^ ((size_t)k.N * 19349663) ^
           ((size_t)k.K * 83492791);
  }
};
std::mutex g_mtx;
std::unordered_map<Key, Plan, KeyHash> g_cache;

const Plan &get_plan(int M, int N, int K) {
  Key key{M, N, K};
  auto it = g_cache.find(key);
  if (it != g_cache.end())
    return it->second;

  Plan p;
  const int m = N, n = M, k = K; // Dc[N,M] = wq(opT)[N,K] @ xq(opN)[K,M]
  CUBLAS_CHECK(
      cublasLtMatmulDescCreate(&p.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      p.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      p.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.a_desc, CUDA_R_8F_E4M3, k, m, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.b_desc, CUDA_R_8F_E4M3, k, n, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.c_desc, CUDA_R_32F, m, n, m));

  cublasLtMatmulPreference_t pref = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t ws = kWorkspaceBytes;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(get_lt_handle(), p.op_desc,
                                              p.a_desc, p.b_desc, p.c_desc,
                                              p.c_desc, pref, 1, &p.heuristic,
                                              &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  TORCH_CHECK(returned > 0, "no cuBLASLt FP8 algorithm for ", m, "x", n, "x", k);
  return g_cache.emplace(key, p).first->second;
}

} // namespace

torch::Tensor fp8_gemm(torch::Tensor xq,
                       torch::Tensor x_scale,
                       torch::Tensor wq,
                       torch::Tensor w_scale,
                       std::optional<torch::Tensor> bias) {
  TORCH_CHECK(xq.is_cuda() && wq.is_cuda(), "inputs must be on CUDA");
  TORCH_CHECK(xq.scalar_type() == torch::kFloat8_e4m3fn &&
                  wq.scalar_type() == torch::kFloat8_e4m3fn,
              "xq and wq must be float8_e4m3fn");
  TORCH_CHECK(xq.dim() == 2 && wq.dim() == 2, "xq, wq must be 2D");
  TORCH_CHECK(xq.size(1) == wq.size(1), "K mismatch: xq[M,K], wq[N,K]");
  auto xqc = xq.contiguous();
  auto wqc = wq.contiguous();

  const int64_t M = xqc.size(0), K = xqc.size(1), N = wqc.size(0);
  TORCH_CHECK(K % 16 == 0, "K must be a multiple of 16 for FP8 matmul, got ", K);
  auto xs = x_scale.contiguous();
  auto ws = w_scale.contiguous();
  TORCH_CHECK(xs.scalar_type() == torch::kFloat && ws.scalar_type() == torch::kFloat,
              "scales must be fp32");
  TORCH_CHECK(xs.numel() == M, "x_scale must have M entries (per-token)");
  TORCH_CHECK(ws.numel() == N, "w_scale must have N entries (per-channel)");

  auto acc = torch::empty({M, N}, xqc.options().dtype(torch::kFloat));
  auto out = torch::empty({M, N}, xqc.options().dtype(torch::kHalf));
  auto workspace = torch::empty(
      {(int64_t)kWorkspaceBytes},
      torch::TensorOptions().dtype(torch::kByte).device(xqc.device()));

  const __half *bias_ptr = nullptr;
  torch::Tensor bias_t;
  if (bias.has_value()) {
    bias_t = bias->to(torch::kHalf).contiguous();
    bias_ptr = reinterpret_cast<const __half *>(bias_t.data_ptr());
  }

  const float alpha = 1.0f, beta = 0.0f;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  {
    std::lock_guard<std::mutex> guard(g_mtx);
    const Plan &p = get_plan((int)M, (int)N, (int)K);
    CUBLAS_CHECK(cublasLtMatmul(get_lt_handle(), p.op_desc, &alpha,
                                wqc.data_ptr(), p.a_desc, xqc.data_ptr(),
                                p.b_desc, &beta, acc.data_ptr(), p.c_desc,
                                acc.data_ptr(), p.c_desc, &p.heuristic.algo,
                                workspace.data_ptr(), kWorkspaceBytes, stream));
  }

  const int threads = 256;
  const int blocks = std::min<int64_t>((M * N + threads - 1) / threads, 8192);
  dequant_bias_kernel<<<blocks, threads, 0, stream>>>(
      acc.data_ptr<float>(), xs.data_ptr<float>(), ws.data_ptr<float>(),
      bias_ptr, reinterpret_cast<__half *>(out.data_ptr()), M, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor fp8_linear(torch::Tensor x,
                         torch::Tensor wq,
                         torch::Tensor w_scale,
                         std::optional<torch::Tensor> bias) {
  auto x2d = x.reshape({-1, x.size(-1)});
  auto qr = quantize_fp8(x2d);
  auto out = fp8_gemm(std::get<0>(qr), std::get<1>(qr), wq, w_scale, bias);
  auto out_shape = x.sizes().vec();
  out_shape.back() = wq.size(0);
  return out.reshape(out_shape);
}
