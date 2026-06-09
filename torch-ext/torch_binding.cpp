#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("quantize_fp8(Tensor x) -> (Tensor, Tensor)");
  ops.impl("quantize_fp8", torch::kCUDA, &quantize_fp8);

  // Arch-aware weight quantization: MXFP8 on Blackwell, per-channel otherwise.
  ops.def("quantize_weight(Tensor weight) -> (Tensor, Tensor)");
  ops.impl("quantize_weight", torch::kCUDA, &quantize_weight);

  ops.def("fp8_gemm("
          "Tensor xq, "
          "Tensor x_scale, "
          "Tensor wq, "
          "Tensor w_scale, "
          "Tensor? bias) -> Tensor");
  ops.impl("fp8_gemm", torch::kCUDA, &fp8_gemm);

  ops.def("fp8_linear("
          "Tensor x, "
          "Tensor wq, "
          "Tensor w_scale, "
          "Tensor? bias) -> Tensor");
  ops.impl("fp8_linear", torch::kCUDA, &fp8_linear);

  // Blackwell (sm_120) MXFP8 microscaling path.
  ops.def("quantize_mxfp8(Tensor x) -> (Tensor, Tensor)");
  ops.impl("quantize_mxfp8", torch::kCUDA, &quantize_mxfp8);

  ops.def("mxfp8_gemm("
          "Tensor xq, "
          "Tensor x_scales, "
          "Tensor wq, "
          "Tensor w_scales, "
          "Tensor? bias) -> Tensor");
  ops.impl("mxfp8_gemm", torch::kCUDA, &mxfp8_gemm);

  ops.def("mxfp8_linear("
          "Tensor x, "
          "Tensor wq, "
          "Tensor w_scales, "
          "Tensor? bias) -> Tensor");
  ops.impl("mxfp8_linear", torch::kCUDA, &mxfp8_linear);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
