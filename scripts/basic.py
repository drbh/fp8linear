# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "diffusers",
#   "transformers",
#   "accelerate",
#   "sentencepiece",
#   "protobuf",
#   "numpy",
#   "Pillow",
#   "huggingface_hub",
#   "kernels==0.14.0",
#   "torch==2.11.0",
# ]
#
# [tool.uv.sources]
# torch = [{ index = "pytorch-cu128" }]
#
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# ///
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image
from kernels import get_kernel
from huggingface_hub import HfApi
from diffusers import FluxPipeline

# Pull the fp8linear kernel from the Hub.
fp8 = get_kernel("drbh/fp8linear", revision="v1")

OUT = Path(__file__).resolve().parent.parent / "basic_out"
OUT.mkdir(exist_ok=True)
# Dataset to publish the generated samples to (override with FP8_SAMPLES_DATASET).
DATASET_REPO = os.environ.get("FP8_SAMPLES_DATASET", "drbh/fp8linear-samples")
PROMPT = "a photograph of an astronaut riding a horse on the moon, dramatic lighting"
H = W = 512
STEPS = 18
SEED = 0


class Fp8Linear(nn.Module):
    """nn.Linear replacement backed by the Hub fp8linear kernel."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        wq, w_scale = fp8.quantize_weight(lin.weight.data.to("cuda"))
        self.register_buffer("wq", wq)
        self.register_buffer("w_scale", w_scale)
        self.register_buffer(
            "bias", lin.bias.data.to(torch.float16) if lin.bias is not None else None
        )

    def forward(self, x):
        return fp8.ops.fp8_linear(x, self.wq, self.w_scale, self.bias).to(x.dtype)


def swap_linears(module: nn.Module) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.in_features % 16 == 0:
            setattr(module, name, Fp8Linear(child))
            n += 1
        else:
            n += swap_linears(child)
    return n


def generate(use_kernel: bool) -> tuple[Image.Image, float]:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    if use_kernel:
        # swap the transformer-block linears (the bulk) for the FP8 kernel
        n = swap_linears(pipe.transformer.transformer_blocks)
        n += swap_linears(pipe.transformer.single_transformer_blocks)
        print(f"  swapped {n} Linear layers -> fp8linear")
        torch.cuda.empty_cache()
    pipe.to("cuda")  # H200 has plenty of VRAM -> no CPU offload needed

    def run():
        return pipe(
            PROMPT,
            height=H,
            width=W,
            num_inference_steps=STEPS,
            guidance_scale=3.5,
            generator=torch.Generator("cpu").manual_seed(SEED),
        ).images[0]

    run()  # warmup (first step pays kernel autotune / allocator costs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return img, elapsed


def upload_to_dataset(info: dict):
    """Push basic_out/ to an HF dataset under a timestamped run dir. No-op without a token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("(no HF_TOKEN -> skipping dataset upload)")
        return
    try:
        run = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        (OUT / "info.json").write_text(json.dumps(info, indent=2))
        api = HfApi(token=token)
        api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(OUT),
            repo_id=DATASET_REPO,
            repo_type="dataset",
            path_in_repo=f"samples/{run}",
            commit_message=f"fp8linear samples {run}",
        )
        print(
            f"uploaded -> https://huggingface.co/datasets/{DATASET_REPO}/tree/main/samples/{run}"
        )
    except Exception as e:
        print(f"(dataset upload failed: {type(e).__name__}: {str(e)[:160]})")


def main():
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    cap = torch.cuda.get_device_capability(0)
    print(
        f"device={torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  "
        f"FLUX.1-dev {H}x{W} {STEPS} steps seed={SEED}\n"
    )

    print("WITHOUT kernel (bf16 baseline)...")
    base, t_base = generate(use_kernel=False)
    base.save(OUT / "without_kernel.png")

    print("WITH fp8linear kernel...")
    fp8_img, t_fp8 = generate(use_kernel=True)
    fp8_img.save(OUT / "with_kernel.png")

    speedup = t_base / t_fp8 if t_fp8 else float("nan")
    print(
        f"\nlatency ({STEPS} steps): bf16 {t_base:.2f}s -> fp8 {t_fp8:.2f}s  "
        f"({speedup:.2f}x faster)"
    )

    a = np.asarray(base, np.float64)
    b = np.asarray(fp8_img, np.float64)
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)
    Image.fromarray(
        np.concatenate([np.asarray(base), np.asarray(fp8_img)], axis=1)
    ).save(OUT / "side_by_side.png")

    mean_dpix = float(np.abs(a - b).mean())
    print(f"\nbf16 vs fp8: PSNR {psnr:.2f} dB, mean |dpix| {mean_dpix:.2f}/255")
    print(f"saved: {OUT}/ (without_kernel.png, with_kernel.png, side_by_side.png)")

    upload_to_dataset(
        {
            "device": torch.cuda.get_device_name(0),
            "prompt": PROMPT,
            "height": H,
            "width": W,
            "steps": STEPS,
            "seed": SEED,
            "psnr_db": round(psnr, 3),
            "mean_pixel_diff": round(mean_dpix, 3),
            "latency_bf16_s": round(t_base, 3),
            "latency_fp8_s": round(t_fp8, 3),
            "speedup": round(speedup, 3),
        }
    )


if __name__ == "__main__":
    main()
