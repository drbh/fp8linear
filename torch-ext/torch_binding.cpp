#include <torch/library.h>

#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("quantize_fp8(Tensor x) -> (Tensor, Tensor)");
  ops.impl("quantize_fp8", torch::kCUDA, &quantize_fp8);

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
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
