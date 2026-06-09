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
        print(f"SKIP: MXFP8 needs Blackwell sm_100+/sm_120; got sm_{cap[0]}{cap[1]}")
        return

    print(f"\n{'shape':<20}{'mx_relerr':>10}{'bf16_us':>10}{'fp8_us':>9}{'mxfp8_us':>10}"
          f"{'mx/bf16':>9}{'mx/fp8':>8}")
    for label, M, K, N in SHAPES:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
        ref = F.linear(x, w)

        # standard FP8 (per-row/channel)
        wq8, ws8 = k.quantize_weight(w)
        # MXFP8 (block-scaled)
        wqm, wsm = k.quantize_weight_mxfp8(w)
        got = k.mxfp8_linear(x, wqm, wsm, None)
        rel = ((got.float() - ref.float()).norm() / ref.float().norm()).item()

        t_bf16 = time_fn(lambda: F.linear(x, w))
        t_fp8 = time_fn(lambda: k.fp8_linear(x, wq8, ws8, None))
        t_mx = time_fn(lambda: k.mxfp8_linear(x, wqm, wsm, None))
        print(f"{label:<20}{rel:>10.4f}{t_bf16:>10.1f}{t_fp8:>9.1f}{t_mx:>10.1f}"
              f"{t_bf16 / t_mx:>8.2f}x{t_fp8 / t_mx:>7.2f}x")

    print("\nmx/bf16 > 1 means MXFP8 beats bf16; mx/fp8 > 1 means it beats the standard FP8 path.")


if __name__ == "__main__":
    main()
