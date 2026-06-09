# /// script
# requires-python = ">=3.10"
# dependencies = ["torch==2.11.*", "numpy", "kernels==0.14.0"]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128" }]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
"""
Validate the Blackwell NVFP4 path (drbh/fp8linear@v1) on real hardware: correctness
vs bf16 and speed vs bf16 / standard FP8 / MXFP8 on the FLUX.1 GEMM shapes.
Requires Blackwell (sm_100+) FP4 tensor cores; skips otherwise.

Accuracy context from the real-weight simulations: nvfp4 W4A4 logits err ~0.046 and
image-level clean; expect per-GEMM rel-err ~0.08-0.12 on random data (e2m1 floor).

Run "Build Kernel" first so the Hub build has the nvfp4 ops.
"""

import torch
import torch.nn.functional as F
from kernels import get_kernel

k = get_kernel("drbh/fp8linear", revision="v1", trust_remote_code=True)

SHAPES = [
    ("single.qkv", 4608, 3072, 9216),
    ("single.proj_mlp", 4608, 3072, 12288),
    ("single.proj_out", 4608, 15360, 3072),
    ("double.ff_img_in", 4096, 3072, 12288),
    ("double.ff_img_out", 4096, 12288, 3072),
]


def time_fn(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    cap = torch.cuda.get_device_capability(0)
    print(f"device={torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  torch={torch.__version__}")
    if cap < (10, 0):
        print(f"SKIP: NVFP4 needs Blackwell sm_100+; got sm_{cap[0]}{cap[1]}")
        return

    print(f"\n{'shape':<20}{'nv_relerr':>10}{'bf16_us':>9}{'fp8_us':>8}{'mx_us':>8}{'nv_us':>8}"
          f"{'nv/bf16':>9}{'nv/fp8':>8}")
    for label, M, K, N in SHAPES:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
        ref = F.linear(x, w)

        wq8, ws8 = k.quantize_weight(w)               # per-channel fp8 (default)
        wqm, wsm = k.quantize_weight_mxfp8(w)         # MXFP8
        wqn, wsn = k.quantize_weight_nvfp4(w)         # NVFP4
        got = k.nvfp4_linear(x, wqn, wsn, None)
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()

        t_bf16 = time_fn(lambda: F.linear(x, w))
        t_fp8 = time_fn(lambda: k.fp8_linear(x, wq8, ws8, None))
        t_mx = time_fn(lambda: k.mxfp8_linear(x, wqm, wsm, None))
        t_nv = time_fn(lambda: k.nvfp4_linear(x, wqn, wsn, None))
        print(f"{label:<20}{rel:>10.4f}{t_bf16:>9.1f}{t_fp8:>8.1f}{t_mx:>8.1f}{t_nv:>8.1f}"
              f"{t_bf16 / t_nv:>8.2f}x{t_fp8 / t_nv:>7.2f}x")

    print("\nnv/bf16 > 1: NVFP4 beats bf16. nv/fp8 > 1: beats the standard FP8 path.")
    print("(rel-err here is per-GEMM W4A4 on random data; end-to-end quality was gated"
          " separately on real weights/images.)")


if __name__ == "__main__":
    main()
