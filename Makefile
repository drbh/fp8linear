# Restricting to a single variant + single capability keeps the Nix build fast.

VARIANT ?= torch211-cxx11-cu128-x86_64-linux

dev:
	nix run github:huggingface/kernels#kernel-builder -- build --variant $(VARIANT) --max-jobs 4 --cores 4
