---
license: mit
tags:
  - kernel
---

# fp8linear

A fused dynamic-quant **FP8 linear** kernel for Ada (sm_89), built with
[`kernel-builder`](https://github.com/huggingface/kernels).

It quantizes activations to e4m3 per-token in a single pass and runs the matmul on
FP8 tensor cores (cuBLASLt), applying per-token × per-channel dequant + bias in a
fused epilogue. The drop-in replacement for the bulk linears in diffusion
transformers like FLUX.

### Kernel Hub

```python
from kernels import get_kernel
fp8linear = get_kernel("drbh/fp8linear", revision="v1")
```
