"""
================================================================================
PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
BENCHMARK: Head-to-Head Compute Kernel Benchmarking (VeloGrad vs PyTorch)
COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
================================================================================
Empirical execution benchmarking comparing VeloGrad fused compute kernels
directly against PyTorch native operations across GEMM, Normalization, RoPE,
Cross-Entropy, and end-to-end model execution.
"""

import sys
import os
import time
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

import velograd as vg
import velograd.nn as nn
import velograd.functional as F
from velograd.tensor import Tensor, randn, zeros
import velograd.kernels as kernels

# Check for PyTorch availability
try:
    import torch
    HAS_TORCH = True
    TORCH_VERSION = torch.__version__
except ImportError:
    HAS_TORCH = False
    TORCH_VERSION = None

def benchmark_gemm(m: int = 128, k: int = 256, n: int = 256, iters: int = 20):
    a = randn(m, k)
    b = randn(k, n)

    # Warmup
    for _ in range(3):
        c = a.matmul(b)

    t0 = time.perf_counter()
    for _ in range(iters):
        c = a.matmul(b)
    t1 = time.perf_counter()

    vg_ms = ((t1 - t0) / iters) * 1000.0
    flops = 2.0 * m * k * n
    vg_gflops = (flops / (vg_ms / 1000.0)) / 1e9

    torch_ms = None
    if HAS_TORCH:
        t_a = torch.randn(m, k)
        t_b = torch.randn(k, n)
        for _ in range(3):
            t_c = torch.matmul(t_a, t_b)
        t0 = time.perf_counter()
        for _ in range(iters):
            t_c = torch.matmul(t_a, t_b)
        t1 = time.perf_counter()
        torch_ms = ((t1 - t0) / iters) * 1000.0

    return vg_ms, vg_gflops, torch_ms

def benchmark_rmsnorm(dim: int = 4096, seq_len: int = 512, iters: int = 20):
    x = randn(seq_len, dim)
    weight = randn(dim)

    # Warmup
    for _ in range(3):
        out = kernels.fused_rmsnorm_residual(x, None, weight, 1e-5)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = kernels.fused_rmsnorm_residual(x, None, weight, 1e-5)
    t1 = time.perf_counter()

    vg_ms = ((t1 - t0) / iters) * 1000.0

    torch_ms = None
    if HAS_TORCH:
        t_x = torch.randn(seq_len, dim)
        t_w = torch.randn(dim)
        for _ in range(3):
            var = t_x.pow(2).mean(-1, keepdim=True)
            t_out = t_x * torch.rsqrt(var + 1e-5) * t_w
        t0 = time.perf_counter()
        for _ in range(iters):
            var = t_x.pow(2).mean(-1, keepdim=True)
            t_out = t_x * torch.rsqrt(var + 1e-5) * t_w
        t1 = time.perf_counter()
        torch_ms = ((t1 - t0) / iters) * 1000.0

    return vg_ms, torch_ms

def benchmark_rope(seq_len: int = 512, num_heads: int = 16, head_dim: int = 64, iters: int = 20):
    dim = num_heads * head_dim
    x = randn(seq_len, dim)

    for _ in range(3):
        out = kernels.fused_rope(x, head_dim=head_dim)

    t0 = time.perf_counter()
    for _ in range(iters):
        out = kernels.fused_rope(x, head_dim=head_dim)
    t1 = time.perf_counter()

    vg_ms = ((t1 - t0) / iters) * 1000.0
    return vg_ms

def benchmark_streaming_cross_entropy(batch: int = 32, vocab: int = 4096, iters: int = 20):
    logits = randn(batch, vocab)
    targets = [i % vocab for i in range(batch)]

    for _ in range(3):
        loss, grad = kernels.fused_cross_entropy_streaming(logits, targets)

    t0 = time.perf_counter()
    for _ in range(iters):
        loss, grad = kernels.fused_cross_entropy_streaming(logits, targets)
    t1 = time.perf_counter()

    vg_ms = ((t1 - t0) / iters) * 1000.0

    torch_ms = None
    if HAS_TORCH:
        t_logits = torch.randn(batch, vocab)
        t_targets = torch.tensor(targets, dtype=torch.long)
        for _ in range(3):
            t_loss = torch.nn.functional.cross_entropy(t_logits, t_targets)
        t0 = time.perf_counter()
        for _ in range(iters):
            t_loss = torch.nn.functional.cross_entropy(t_logits, t_targets)
        t1 = time.perf_counter()
        torch_ms = ((t1 - t0) / iters) * 1000.0

    return vg_ms, torch_ms

def benchmark_end_to_end_step(dim: int = 128, seq_len: int = 64, iters: int = 5):
    model = nn.Sequential(
        nn.Linear(dim, dim),
        nn.RMSNorm(dim),
        nn.GELU(),
        nn.Linear(dim, dim)
    )
    optimizer = vg.optim.AdamW(model.parameters(), lr=1e-3)
    x = randn(seq_len, dim)
    target = randn(seq_len, dim)

    t0 = time.perf_counter()
    for _ in range(iters):
        optimizer.zero_grad()
        out = model(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
    t1 = time.perf_counter()

    vg_step_ms = ((t1 - t0) / iters) * 1000.0
    vg_tokens_per_sec = (seq_len * iters) / (t1 - t0)

    torch_step_ms = None
    if HAS_TORCH:
        t_model = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim)
        )
        t_opt = torch.optim.AdamW(t_model.parameters(), lr=1e-3)
        t_x = torch.randn(seq_len, dim)
        t_target = torch.randn(seq_len, dim)
        for _ in range(2):
            t_opt.zero_grad()
            t_out = t_model(t_x)
            t_loss = torch.nn.functional.mse_loss(t_out, t_target)
            t_loss.backward()
            t_opt.step()
        t0 = time.perf_counter()
        for _ in range(iters):
            t_opt.zero_grad()
            t_out = t_model(t_x)
            t_loss = torch.nn.functional.mse_loss(t_out, t_target)
            t_loss.backward()
            t_opt.step()
        t1 = time.perf_counter()
        torch_step_ms = ((t1 - t0) / iters) * 1000.0

    return vg_step_ms, vg_tokens_per_sec, torch_step_ms

def main():
    print("=" * 80)
    print("      VELOGRAD vs PYTORCH COMPUTE ENGINE BENCHMARK SUITE")
    print("      Evaluating Fused Compute Kernels & Step Latencies")
    print("=" * 80)
    print(f"  VeloGrad Version: {vg.__version__}")
    if HAS_TORCH:
        print(f"  PyTorch Version : {TORCH_VERSION} (Live Head-to-Head Active)")
    else:
        print("  PyTorch Version : Not installed (Running standalone VeloGrad benchmarks)")
    print("-" * 80)

    print("\n[1/5] Benchmarking MatMul (GEMM 128x256x256):")
    gemm_ms, gflops, torch_gemm = benchmark_gemm(m=128, k=256, n=256)
    print(f"      VeloGrad Latency : {gemm_ms:.2f} ms | Throughput: {gflops:.2f} GFLOPS")
    if torch_gemm:
        print(f"      PyTorch Latency  : {torch_gemm:.2f} ms")

    print("\n[2/5] Benchmarking Fused RMSNorm + Residual Add (512x4096):")
    rmsnorm_ms, torch_rmsnorm = benchmark_rmsnorm(dim=4096, seq_len=512)
    print(f"      VeloGrad Latency : {rmsnorm_ms:.2f} ms (fused single-pass)")
    if torch_rmsnorm:
        print(f"      PyTorch Latency  : {torch_rmsnorm:.2f} ms (unfused RMSNorm)")

    print("\n[3/5] Benchmarking Fused Rotary Position Embeddings (RoPE):")
    rope_ms = benchmark_rope(seq_len=512, num_heads=16, head_dim=64)
    print(f"      VeloGrad Latency : {rope_ms:.2f} ms (512-token in-place rotation)")

    print("\n[4/5] Benchmarking Streaming Single-Pass Cross-Entropy (32x4096):")
    ce_ms, torch_ce = benchmark_streaming_cross_entropy(batch=32, vocab=4096)
    print(f"      VeloGrad Latency : {ce_ms:.2f} ms (zero logit tensor materialization)")
    if torch_ce:
        print(f"      PyTorch Latency  : {torch_ce:.2f} ms")

    print("\n[5/5] Benchmarking End-to-End Training Step (Fwd + Bwd + AdamW):")
    step_ms, tok_s, torch_step = benchmark_end_to_end_step(dim=128, seq_len=64)
    print(f"      VeloGrad Latency : {step_ms:.2f} ms | Throughput: {tok_s:.1f} tokens/sec")
    if torch_step:
        print(f"      PyTorch Latency  : {torch_step:.2f} ms")

    print("\n" + "=" * 80)
    print(">>> Benchmark execution completed successfully.")
    print("=" * 80)

if __name__ == "__main__":
    main()
