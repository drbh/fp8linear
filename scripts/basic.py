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
from kernels import (
    LayerRepository,
    Mode,
    kernelize,
    replace_kernel_forward_from_hub,
    use_kernel_mapping,
)
from huggingface_hub import HfApi
from diffusers import FluxPipeline

# Make nn.Linear extensible and map it to the kernel's stateless Fp8Linear layer.
# kernelize() grafts the layer's forward onto each nn.Linear in place (keeping the
# module's own weight/bias), pulling the kernel from the Hub.
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
H = W = 512
STEPS = 18
SEED = 0


def generate(use_kernel: bool) -> tuple[Image.Image, float]:
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")  # H200 has plenty of VRAM -> no CPU offload needed
    if use_kernel:
        # Kernelize only the transformer blocks (the bulk of the compute); the
        # embedders / output head stay bf16.
        with use_kernel_mapping(FP8_MAPPING):
            kernelize(
                pipe.transformer.transformer_blocks, mode=Mode.INFERENCE, device="cuda"
            )
            kernelize(
                pipe.transformer.single_transformer_blocks,
                mode=Mode.INFERENCE,
                device="cuda",
            )
        print("  kernelized transformer blocks -> fp8linear:Fp8Linear")

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
