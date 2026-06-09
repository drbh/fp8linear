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
#include <cublas_v2.h>  // for CUBLAS_VERSION
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
std::tuple<torch::Tensor, torch::Tensor> quantize_mxfp8(torch::Tensor x);
torch::Tensor mxfp8_gemm(torch::Tensor xq, torch::Tensor x_scales,
                         torch::Tensor wq, torch::Tensor w_scales,
                         std::optional<torch::Tensor> bias);
torch::Tensor mxfp8_linear(torch::Tensor x, torch::Tensor wq,
                           torch::Tensor w_scales,
                           std::optional<torch::Tensor> bias);
std::tuple<torch::Tensor, torch::Tensor> quantize_nvfp4(torch::Tensor x);
torch::Tensor nvfp4_gemm(torch::Tensor xq, torch::Tensor x_scales,
                         torch::Tensor wq, torch::Tensor w_scales,
                         std::optional<torch::Tensor> bias);
torch::Tensor nvfp4_linear(torch::Tensor x, torch::Tensor wq,
                           torch::Tensor w_scales,
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
  torch::Tensor out;
  // Dispatch on the weight format produced by quantize_weight:
  //   packed-uint8 weight + uint8 scales -> NVFP4 (e2m1, per-16 e4m3 scales)
  //   e4m3 weight + uint8 scales        -> MXFP8 (per-32 e8m0 scales)
  //   e4m3 weight + fp32 scales         -> standard per-channel FP8
  if (w_scale.scalar_type() == torch::kByte) {
    if (wq.scalar_type() == torch::kByte) {
      auto qr = quantize_nvfp4(x2d);
      out = nvfp4_gemm(std::get<0>(qr), std::get<1>(qr), wq, w_scale, bias);
    } else {
      auto qr = quantize_mxfp8(x2d);
      out = mxfp8_gemm(std::get<0>(qr), std::get<1>(qr), wq, w_scale, bias);
    }
  } else {
    auto qr = quantize_fp8(x2d);
    out = fp8_gemm(std::get<0>(qr), std::get<1>(qr), wq, w_scale, bias);
  }
  auto out_shape = x.sizes().vec();
  out_shape.back() = wq.size(0);
  return out.reshape(out_shape);
}

// ===== Blackwell (sm_120) MXFP8 path =====
// Block-scaled (microscaling) FP8: the e8m0 per-32 scales are consumed *inside*
// the matmul by Blackwell's tensor cores (VEC32_UE8M0 scale mode), so the result
// is written fp16 directly -- no fp32 accumulator and no separate dequant epilogue,
// which is the overhead that caps the standard FP8 path on fast-bf16 GPUs.
//
// NOTE: requires sm_100+/sm_120 tensor cores to run; cannot be exercised on Ada.
// Scale tensors are in cuBLASLt's tiled scale-factor layout (512-byte tiles of
// 128 rows x 4 scale-cols, K-major tile order, padded grid) as produced by
// quantize_mxfp8 -- a flat row-major layout here yields NaN garbage (verified
// the hard way on an RTX PRO 6000).
torch::Tensor mxfp8_gemm(torch::Tensor xq,
                         torch::Tensor x_scales,
                         torch::Tensor wq,
                         torch::Tensor w_scales,
                         std::optional<torch::Tensor> bias) {
  // The MX (VEC32_UE8M0) scale modes were added in cuBLAS 12.8; the build matrix
  // also compiles cu126 variants where they don't exist, so guard on the version.
#if CUBLAS_VERSION >= 120800
  TORCH_CHECK(xq.is_cuda() && wq.is_cuda(), "inputs must be on CUDA");
  TORCH_CHECK(xq.scalar_type() == torch::kFloat8_e4m3fn &&
                  wq.scalar_type() == torch::kFloat8_e4m3fn,
              "xq and wq must be float8_e4m3fn");
  auto xqc = xq.contiguous();
  auto wqc = wq.contiguous();
  auto xs = x_scales.contiguous();
  auto ws = w_scales.contiguous();
  const int64_t M = xqc.size(0), K = xqc.size(1), N = wqc.size(0);
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32 for MXFP8, got ", K);
  // Scales must be in the tiled cuBLASLt layout from quantize_mxfp8: 512-byte
  // tiles over a padded grid of ceil(rows/128) x ceil((K/32)/4).
  const int64_t ncb = ((K / 32) + 3) / 4;
  TORCH_CHECK(xs.scalar_type() == torch::kByte && ws.scalar_type() == torch::kByte,
              "MX scales must be uint8 (e8m0)");
  TORCH_CHECK(xs.numel() == ((M + 127) / 128) * ncb * 512,
              "x_scales has wrong size for tiled MX layout (expected ",
              ((M + 127) / 128) * ncb * 512, ", got ", xs.numel(), ")");
  TORCH_CHECK(ws.numel() == ((N + 127) / 128) * ncb * 512,
              "w_scales has wrong size for tiled MX layout (expected ",
              ((N + 127) / 128) * ncb * 512, ", got ", ws.numel(), ")");

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

  // Dc[N,M] = wq(opT)[N,K] @ xq(opN)[K,M], fp16 direct.
  const int m = (int)N, n = (int)M, k = (int)K;
  cublasLtMatmulDesc_t op = nullptr;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  cublasLtMatmulMatrixScale_t mx = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mx, sizeof(mx)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mx, sizeof(mx)));
  void *a_scale = ws.data_ptr();  // A <- wq, e8m0 [N, K/32]
  void *b_scale = xs.data_ptr();  // B <- xq, e8m0 [M, K/32]
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
  if (bias_ptr != nullptr) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }

  cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_8F_E4M3, k, m, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_8F_E4M3, k, n, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, m, n, m));

  cublasLtMatmulPreference_t pref;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsbytes = kWorkspaceBytes;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsbytes, sizeof(wsbytes)));
  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(get_lt_handle(), op, a_desc, b_desc,
                                              c_desc, c_desc, pref, 1, &heur, &returned));
  TORCH_CHECK(returned > 0, "no cuBLASLt MXFP8 algorithm for ", m, "x", n, "x", k,
              " (needs Blackwell sm_100+/sm_120 and a cuBLASLt build with MX support)");

  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasLtMatmul(
      get_lt_handle(), op, &alpha, wqc.data_ptr(), a_desc, xqc.data_ptr(), b_desc,
      &beta, out.data_ptr(), c_desc, out.data_ptr(), c_desc, &heur.algo,
      workspace.data_ptr(), kWorkspaceBytes,
      at::cuda::getCurrentCUDAStream().stream()));

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatmulDescDestroy(op);
  return out;
#else
  TORCH_CHECK(false,
              "mxfp8_gemm requires a cuBLAS >= 12.8 build (this variant has ",
              CUBLAS_VERSION, "); use a cu128+ build for the Blackwell MXFP8 path");
  return torch::Tensor();
#endif
}

torch::Tensor mxfp8_linear(torch::Tensor x,
                           torch::Tensor wq,
                           torch::Tensor w_scales,
                           std::optional<torch::Tensor> bias) {
  auto x2d = x.reshape({-1, x.size(-1)});
  auto qr = quantize_mxfp8(x2d);
  auto out = mxfp8_gemm(std::get<0>(qr), std::get<1>(qr), wq, w_scales, bias);
  auto out_shape = x.sizes().vec();
  out_shape.back() = wq.size(0);
  return out.reshape(out_shape);
}

// ===== Blackwell (sm_100+/sm_120) NVFP4 path =====
// e2m1 4-bit operands (two per byte) with an e4m3 (ue4m3) scale per 16-element
// block, consumed natively by Blackwell tensor cores at 2x the FP8 rate. Same
// tiled scale-factor layout as MXFP8 (VEC16 -> scale-column count is K/16).
// Operands arrive packed as uint8 [rows, K/2] from quantize_nvfp4.
torch::Tensor nvfp4_gemm(torch::Tensor xq,
                         torch::Tensor x_scales,
                         torch::Tensor wq,
                         torch::Tensor w_scales,
                         std::optional<torch::Tensor> bias) {
#if CUBLAS_VERSION >= 120800
  TORCH_CHECK(xq.is_cuda() && wq.is_cuda(), "inputs must be on CUDA");
  TORCH_CHECK(xq.scalar_type() == torch::kByte && wq.scalar_type() == torch::kByte,
              "xq and wq must be packed uint8 (two e2m1 per byte)");
  auto xqc = xq.contiguous();
  auto wqc = wq.contiguous();
  auto xs = x_scales.contiguous();
  auto ws = w_scales.contiguous();
  const int64_t M = xqc.size(0), K = xqc.size(1) * 2, N = wqc.size(0);
  TORCH_CHECK(wqc.size(1) * 2 == K, "K mismatch: xq[M,K/2], wq[N,K/2]");
  TORCH_CHECK(K % 32 == 0, "K must be a multiple of 32 for NVFP4, got ", K);
  const int64_t ncb = ((K / 16) + 3) / 4;
  TORCH_CHECK(xs.scalar_type() == torch::kByte && ws.scalar_type() == torch::kByte,
              "NVFP4 scales must be uint8 (e4m3)");
  TORCH_CHECK(xs.numel() == ((M + 127) / 128) * ncb * 512,
              "x_scales has wrong size for tiled NVFP4 layout");
  TORCH_CHECK(ws.numel() == ((N + 127) / 128) * ncb * 512,
              "w_scales has wrong size for tiled NVFP4 layout");

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

  // Dc[N,M] = wq(opT)[N,K] @ xq(opN)[K,M], fp16 direct; ld is in ELEMENTS.
  const int m = (int)N, n = (int)M, k = (int)K;
  cublasLtMatmulDesc_t op = nullptr;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t op_t = CUBLAS_OP_T, op_n = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &op_t, sizeof(op_t)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

  cublasLtMatmulMatrixScale_t v16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &v16, sizeof(v16)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &v16, sizeof(v16)));
  void *a_scale = ws.data_ptr();  // A <- wq scales (per 16, tiled)
  void *b_scale = xs.data_ptr();  // B <- xq scales (per 16, tiled)
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
  if (bias_ptr != nullptr) {
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
  }

  cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_4F_E2M1, k, m, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_4F_E2M1, k, n, k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, m, n, m));

  cublasLtMatmulPreference_t pref;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  size_t wsbytes = kWorkspaceBytes;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsbytes, sizeof(wsbytes)));
  cublasLtMatmulHeuristicResult_t heur{};
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(get_lt_handle(), op, a_desc, b_desc,
                                              c_desc, c_desc, pref, 1, &heur, &returned));
  TORCH_CHECK(returned > 0, "no cuBLASLt NVFP4 algorithm for ", m, "x", n, "x", k,
              " (needs Blackwell sm_100+/sm_120 with FP4 tensor cores)");

  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasLtMatmul(
      get_lt_handle(), op, &alpha, wqc.data_ptr(), a_desc, xqc.data_ptr(), b_desc,
      &beta, out.data_ptr(), c_desc, out.data_ptr(), c_desc, &heur.algo,
      workspace.data_ptr(), kWorkspaceBytes,
      at::cuda::getCurrentCUDAStream().stream()));

  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatmulDescDestroy(op);
  return out;
#else
  TORCH_CHECK(false,
              "nvfp4_gemm requires a cuBLAS >= 12.8 build (this variant has ",
              CUBLAS_VERSION, "); use a cu128+ build for the Blackwell NVFP4 path");
  return torch::Tensor();
#endif
}

torch::Tensor nvfp4_linear(torch::Tensor x,
                           torch::Tensor wq,
                           torch::Tensor w_scales,
                           std::optional<torch::Tensor> bias) {
  auto x2d = x.reshape({-1, x.size(-1)});
  auto qr = quantize_nvfp4(x2d);
  auto out = nvfp4_gemm(std::get<0>(qr), std::get<1>(qr), wq, w_scales, bias);
  auto out_shape = x.sizes().vec();
  out_shape.back() = wq.size(0);
  return out.reshape(out_shape);
}
