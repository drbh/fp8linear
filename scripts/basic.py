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
"""
Generate the same FLUX.1-dev image with all four precisions -- bf16, fp8 (per-channel),
mxfp8 (e8m0/32 block scales), nvfp4 (e2m1 + e4m3/16 block scales) -- and stitch them
into a labeled 2x2 grid. Each tile carries the kernel name + runtime burned into the
bottom-right corner so the quality/speed trade-off is visible at a glance.

The block-scaled schemes are selected via the kernel's own env switches
(FP8LINEAR_USE_MX / FP8LINEAR_USE_NVFP4) around the one-time weight quantization;
they need Blackwell (sm_100+) and are skipped elsewhere.
"""

import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
from kernels import (
    LayerRepository,
    Mode,
    get_kernel,
    kernelize,
    replace_kernel_forward_from_hub,
    use_kernel_mapping,
)
from huggingface_hub import HfApi
from diffusers import FluxPipeline

# Pull the kernel from the Hub for the one-time weight-quantization helper.
# trust_remote_code=True: drbh is not a default-trusted publisher.
fp8 = get_kernel("drbh/fp8linear", revision="v1", trust_remote_code=True)

# Make nn.Linear extensible and map it to the kernel's stateless Fp8Linear layer.
# kernelize() grafts the layer's forward onto each nn.Linear in place; the layer
# reads the quantized weight + scale that fp8.quantize_() stored on the module
# (the weight/scale dtypes select fp8 vs mxfp8 vs nvfp4 inside the op).
replace_kernel_forward_from_hub(nn.Linear, "Fp8Linear")
FP8_MAPPING = {
    "Fp8Linear": {
        "cuda": LayerRepository(
            repo_id="drbh/fp8linear",
            revision="v1",
            layer_name="Fp8Linear",
            trust_remote_code=True,
        )
    }
}

OUT = Path(__file__).resolve().parent.parent / "basic_out"
OUT.mkdir(exist_ok=True)
# Dataset to publish the generated samples to (override with FP8_SAMPLES_DATASET).
DATASET_REPO = os.environ.get("FP8_SAMPLES_DATASET", "drbh/fp8linear-samples")
PROMPT = "a photograph of an astronaut riding a horse on the moon, dramatic lighting"
# 1024px (seq ~4608) is FLUX's native resolution and where the quantized GEMMs are
# compute-bound enough to win over fast bf16.
H = W = int(os.environ.get("FP8_RES", "1024"))
STEPS = int(os.environ.get("FP8_STEPS", "18"))
SEED = 0

# scheme -> env flag understood by the kernel's C++ quantize_weight (None = default fp8)
SCHEME_ENV = {"fp8": None, "mxfp8": "FP8LINEAR_USE_MX", "nvfp4": "FP8LINEAR_USE_NVFP4"}


def quantize_and_kernelize(pipe, scheme: str) -> int:
    """Quantize transformer-block weights once (format per `scheme`), then kernelize."""
    for flag in ("FP8LINEAR_USE_MX", "FP8LINEAR_USE_NVFP4"):
        os.environ.pop(flag, None)
    flag = SCHEME_ENV[scheme]
    if flag:
        os.environ[flag] = "1"
    try:
        blocks = (
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        )
        n = sum(fp8.quantize_(b) for b in blocks)
        with use_kernel_mapping(FP8_MAPPING):
            for b in blocks:
                kernelize(b, mode=Mode.INFERENCE, device="cuda")
        return n
    finally:
        if flag:
            os.environ.pop(flag, None)


def generate(scheme: str) -> tuple[Image.Image, float]:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")
    if scheme != "bf16":
        n = quantize_and_kernelize(pipe, scheme)
        print(f"  quantized + kernelized {n} Linear layers -> {scheme}")

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


def label(img: Image.Image, text: str) -> Image.Image:
    """Burn `text` into the bottom-right corner (white on translucent black)."""
    img = img.convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default(size=max(18, img.height // 32))
    except TypeError:  # older Pillow: no size param
        font = ImageFont.load_default()
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    pad, margin = 10, 14
    bx1, by1 = img.width - margin, img.height - margin
    bx0, by0 = bx1 - tw - 2 * pad, by1 - th - 2 * pad
    d.rectangle([bx0, by0, bx1, by1], fill=(0, 0, 0, 170))
    d.text((bx0 + pad, by0 + pad - y0), text, fill=(255, 255, 255, 255), font=font)
    return img


def psnr_vs(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a, np.float64)
    y = np.asarray(b, np.float64)
    mse = float(((x - y) ** 2).mean())
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


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
    blackwell = cap >= (10, 0)
    schemes = ["bf16", "fp8"] + (["mxfp8", "nvfp4"] if blackwell else [])
    print(
        f"device={torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}  "
        f"FLUX.1-dev {H}x{W} {STEPS} steps seed={SEED}  schemes={schemes}"
    )
    if not blackwell:
        print("(mxfp8/nvfp4 need Blackwell sm_100+ -> skipped)\n")

    images, times = {}, {}
    for scheme in schemes:
        print(f"{scheme}...")
        images[scheme], times[scheme] = generate(scheme)
        images[scheme].save(OUT / f"{scheme}.png")

    base, t_base = images["bf16"], times["bf16"]
    info = {
        "device": torch.cuda.get_device_name(0),
        "prompt": PROMPT,
        "height": H, "width": W, "steps": STEPS, "seed": SEED,
        "schemes": {},
    }
    labeled = []
    print(f"\n{'scheme':<8}{'latency':>9}{'speedup':>9}{'psnr_vs_bf16':>14}")
    for scheme in schemes:
        t = times[scheme]
        speed = t_base / t if t else float("nan")
        p = psnr_vs(images[scheme], base) if scheme != "bf16" else float("inf")
        tag = f"{scheme}  {t:.2f}s  " + ("(baseline)" if scheme == "bf16" else f"({speed:.2f}x)")
        labeled.append(np.asarray(label(images[scheme], tag)))
        info["schemes"][scheme] = {
            "latency_s": round(t, 3),
            "speedup": round(speed, 3),
            "psnr_db_vs_bf16": None if scheme == "bf16" else round(p, 3),
        }
        print(f"{scheme:<8}{t:>8.2f}s{speed:>8.2f}x"
              + (f"{p:>13.2f} dB" if scheme != "bf16" else f"{'—':>14}"))

    # stitch: 2x2 grid when we have 4, else a horizontal strip
    if len(labeled) == 4:
        grid = np.concatenate(
            [np.concatenate(labeled[:2], axis=1), np.concatenate(labeled[2:], axis=1)], axis=0)
    else:
        grid = np.concatenate(labeled, axis=1)
    Image.fromarray(grid).save(OUT / "grid.png")
    print(f"saved: {OUT}/ ({', '.join(s + '.png' for s in schemes)}, grid.png)")

    upload_to_dataset(info)


if __name__ == "__main__":
    main()
