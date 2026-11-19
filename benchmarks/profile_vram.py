"""
================================================================================
PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
BENCHMARK: Python Heap Memory Profiler (CPU / No-GPU)
COPYRIGHT (C) 2026 Ayushman Singh Rathore
================================================================================

Measures REAL Python heap allocations using tracemalloc.
Reports peak memory in MB for Tensor allocation and forward-pass operations.

NOTE: This benchmark runs on CPU only. It measures Python-side heap allocations,
NOT VRAM. GPU VRAM profiling requires physical NVIDIA hardware with pynvml or
torch.cuda.memory_stats(), which are unavailable in this environment.

All numbers below are real measurements from tracemalloc — no formulas, no
hardcoded values, no assertions on specific percentages.
"""

import sys
import os
import tracemalloc
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import velograd as vg
from velograd.tensor import Tensor, randn, zeros
import velograd.nn as nn
import velograd.optim as optim


def measure_alloc(label: str, fn):
    """Run fn() under tracemalloc and return (result, peak_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    print(f"  {label:<45} peak={peak_mb:>8.3f} MB   time={elapsed*1000:>8.2f} ms")
    return result, peak_mb


def main():
    print("=" * 80)
    print("      VELOGRAD CPU HEAP MEMORY PROFILER (tracemalloc — real measurements)")
    print("=" * 80)
    print(f"  velograd version: {vg.__version__}")
    print(f"  Environment:      CPU-only (no CUDA hardware present)")
    print()

    results = {}

    # --- Tensor Allocation ---
    print("[ Tensor Allocation ]")

    _, mb = measure_alloc("randn(1, 512)  — small tensor",
                          lambda: randn(1, 512))
    results["randn_small"] = mb

    _, mb = measure_alloc("randn(16, 2048) — medium tensor",
                          lambda: randn(16, 2048))
    results["randn_medium"] = mb

    _, mb = measure_alloc("randn(64, 4096) — large tensor",
                          lambda: randn(64, 4096))
    results["randn_large"] = mb

    # --- Linear Layer Forward + Backward ---
    print()
    print("[ Linear Layer Forward + Backward ]")

    def linear_fwd_bwd():
        model = nn.Linear(128, 64)
        x = randn(8, 128, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        return loss

    _, mb = measure_alloc("Linear(128, 64) fwd+bwd  bs=8", linear_fwd_bwd)
    results["linear_fwd_bwd"] = mb

    def linear_large_fwd_bwd():
        model = nn.Linear(512, 256)
        x = randn(16, 512, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        return loss

    _, mb = measure_alloc("Linear(512, 256) fwd+bwd bs=16", linear_large_fwd_bwd)
    results["linear_large"] = mb

    # --- MLP Forward + Backward ---
    print()
    print("[ MLP Forward + Backward ]")

    def mlp_fwd_bwd():
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)
                self._submodules.extend([self.fc1, self.fc2])
            def forward(self, x):
                import velograd.functional as F
                return self.fc2(F.relu(self.fc1(x)))

        model = MLP()
        x = randn(4, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        return loss

    _, mb = measure_alloc("MLP(64→128→64) fwd+bwd bs=4", mlp_fwd_bwd)
    results["mlp"] = mb

    # --- AdamW Optimizer Step ---
    print()
    print("[ AdamW Optimizer Step ]")

    def adamw_step():
        model = nn.Linear(64, 32)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        x = randn(4, 64, requires_grad=True)
        for _ in range(3):
            opt.zero_grad()
            out = model(x)
            loss = out.sum()
            loss.backward()
            opt.step()
        return loss

    _, mb = measure_alloc("AdamW 3-step loop on Linear(64,32)", adamw_step)
    results["adamw"] = mb

    # --- Summary ---
    print()
    print("=" * 80)
    print("  SUMMARY (real tracemalloc peak allocations)")
    print("=" * 80)
    for k, v in results.items():
        print(f"  {k:<40}  {v:>8.3f} MB")

    print()
    print("NOTE: These are Python heap measurements. GPU VRAM profiling requires")
    print("      physical NVIDIA hardware (pynvml / torch.cuda) — not available here.")
    print("      Do NOT interpret these numbers as GPU VRAM consumption.")


if __name__ == "__main__":
    main()
