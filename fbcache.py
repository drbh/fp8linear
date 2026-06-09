"""
First-Block Cache (FBCache / TeaCache-style step caching) for Flux.

The transformer output evolves slowly across denoising steps. FBCache runs only
the *first* transformer block each step; if its output residual changed little
vs the previous step (relative L1 below `threshold`), it skips the remaining
blocks and reuses the cached aggregate residual. Otherwise it runs the full
stack and refreshes the cache.

This is the mechanism behind the step-caching speedup measured in
`bench_flux_real.py` (skipping ~40-50% of steps -> ~1.4-2x on top of compute,
the published quality-preserving range for Flux). It composes with the FP8
kernel + torch.compile (which accelerate the steps that *are* computed).

Integration sketch (inside a FluxTransformer2DModel.forward override):

    cache = FBCache(threshold=0.12)
    # per step, after embeddings:
    hidden, enc = blocks[0](hidden, enc, temb, rope)        # always run block 0
    def run_rest(h, e):
        for blk in double_blocks[1:]:
            e, h = blk(h, e, temb, rope)
        ... single blocks ...
        return h, e
    hidden, enc = cache.apply(first_block_residual=hidden, run_rest=run_rest,
                              rest_inputs=(hidden, enc))
"""

from __future__ import annotations

import torch


class FBCache:
    def __init__(self, threshold: float = 0.12):
        self.threshold = threshold
        self._prev_first = None  # previous step's first-block output
        self._cached_residual = (
            None  # (full_output - first_block_output), reused on skip
        )
        self.computed = 0
        self.skipped = 0

    def relative_change(self, cur: torch.Tensor) -> float:
        if self._prev_first is None:
            return float("inf")
        denom = self._prev_first.abs().mean().clamp_min(1e-6)
        return ((cur - self._prev_first).abs().mean() / denom).item()

    def apply(self, first_out, run_rest, rest_inputs):
        """first_out: tensor used for the change metric (block-0 output).
        run_rest(*rest_inputs) -> full output(s). Returns the (possibly cached) full output."""
        change = self.relative_change(first_out)
        reuse = change < self.threshold and self._cached_residual is not None

        if reuse:
            self.skipped += 1
            out = first_out + self._cached_residual
        else:
            self.computed += 1
            out = run_rest(*rest_inputs)
            primary = out[0] if isinstance(out, tuple) else out
            self._cached_residual = primary - first_out

        self._prev_first = first_out.detach().clone()
        return out

    def summary(self) -> str:
        total = self.computed + self.skipped
        r = self.skipped / total if total else 0.0
        return (
            f"FBCache: {self.computed} computed, {self.skipped} skipped ({r:.0%} skip)"
        )
