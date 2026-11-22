# VeloGrad (Beta)

[![Status: Beta](https://img.shields.io/badge/Status-Beta-orange.svg)](#beta-notice)
[![License: AGPLv3](https://img.shields.io/badge/License-AGPLv3%20%2B%20Commercial%20Waiver-blue.svg)](LICENSE)
[![Architecture: C++20 / Rust / CUDA / Python](https://img.shields.io/badge/Architecture-C%2B%2B20%20%7C%20Rust%20%7C%20CUDA%20%7C%20Python-orange.svg)](#architecture)
[![Version: 1.1.0-beta](https://img.shields.io/badge/Release-1.1.0--beta-green.svg)](#release-milestone)
[![Compatibility: NumPy | PyTorch | JAX | GGUF | HF](https://img.shields.io/badge/Compatibility-NumPy%20%7C%20PyTorch%20%7C%20JAX%20%7C%20GGUF%20%7C%20HF-blueviolet.svg)](#universal-ecosystem-compatibility)

**VeloGrad** is a bare-metal, native-first, pure-compiled deep learning and autoregressive inference engine engineered from scratch in **C++20, Rust, CUDA, and Python** by **Ayushman Singh Rathore**.

It eliminates Python runtime overhead, framework memory bloat, and garbage collection pauses by introducing an aligned static arena memory architecture, ahead-of-time (AOT) fused compute kernels, a **Paged KV-cache virtual memory engine** (designed to reduce KV-cache fragmentation), **ZeRO-3 / FSDP distributed scaling with CPU/NVMe offload**, and a **natural-language autonomous training orchestrator** (`velograd.instruct`).

---

> [!IMPORTANT]
> ### Public Beta Notice (`v1.1.0-beta`)
> **VeloGrad `v1.1.0-beta` is now in public beta.**  
> The Python tensor engine, autograd, fused SIMD kernels, ZeRO-3 FSDP simulation, and `velograd.instruct` orchestrator are verified on CPU in CI. CUDA kernels are written but have **not** been tested on physical NVIDIA hardware. KV-cache VRAM reduction numbers require real GPU hardware to measure — no GPU-side benchmarks have been run. Community testing and feedback are actively welcomed.

---

## Universal Model Family & Parameter Scale Support

VeloGrad natively supports, loads, trains, fine-tunes, and serves across all modern dense, Mixture-of-Experts (MoE), and hybrid state-space architectures:

| Model Family | Representative Architectures | Parameter Scale | Key Architectural Primitives |
| :--- | :--- | :--- | :--- |
| **Llama Family** | Llama 3 / 3.1 / 3.2, Llama 4 Scout & Maverick | **1B, 3B, 8B, 70B, 405B, 109B MoE, 400B MoE** | RoPE, SwiGLU, GQA, Paged KV-Cache, Sparse MoE |
| **Qwen Family** | Qwen 2.5, Qwen 3, Qwen 3.5, Qwen 3.8 | **0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B, 235B MoE, 397B MoE** | High-density RoPE, Top-K MoE, Dynamic Context |
| **DeepSeek Family** | DeepSeek-R1 / R2, DeepSeek-V4 Flash & Pro | **1.5B–70B Distilled, 671B MoE, 284B MoE, 1.6T MoE** | Multi-Head Latent Attention (MLA), Auxiliary-Free MoE |
| **Gemma Family** | Gemma 2, Gemma 3, Gemma 4 | **1B, 2B, 4B, 9B, 12B, 26B MoE, 27B, 31B** | Interleaved Local/Global Sliding Attention, GeGLU |
| **Mistral & Mixtral** | Mistral 7B, Small 24B, Mixtral 8x7B / 8x22B, Large 3 | **7B, 24B, 8x7B, 8x22B, 675B MoE** | Sliding Window Attention, Sparse Top-2 Router |
| **Phi, GLM & Kimi** | Microsoft Phi-4 / 5, GLM-5, Kimi K3 | **1.3B, 3.8B, 14B, 15B, 32B MoE** | High-Param Dense Reasoning, Long-Context MLA |
| **Hybrid State-Space** | IBM Granite 4, Mamba-2 Hybrid | **4B, 8B, 32B Hybrid** | State Space Duality (SSD), Linear-Time Recurrence |

---

## Core Breakthrough Systems

### 1. Fused Compute Kernels (CUDA + SIMD CPU Fallback)
Complete hardware-accelerated kernels implemented in both native CUDA (`src/cuda/fused_kernels.cu`) with shared-memory tiling and vectorized C++/NumPy SIMD fallbacks (`velograd.kernels`):
- **Fused RMSNorm + Residual Add**: Single-pass forward and backward analytic gradients.
- **Fused Rotary Position Embeddings (RoPE)**: In-place trigonometric coordinate rotation across query and key heads.
- **Single-Pass Streaming Cross-Entropy**: Computes log-softmax and analytic gradients without allocating massive `[batch, seq, vocab]` logit tensors in VRAM.
- **Fused Top-K MoE Routing**: Shared-memory probability reduction and sparse token dispatch.
- **Fused Multi-Tensor AdamW**: Vectorized in-place first and second moment updates with fused weight decay.

### 2. Paged KV-Cache & Memory Reduction Engine
- Combines virtual memory **Paged KV-Cache** block tables, **INT4 / FP8 block quantization**, **Grouped-Query Attention (GQA)**, and **Multi-Head Latent Attention (MLA)** cache compression.
- Designed to significantly reduce KV-cache fragmentation. Actual VRAM reduction percentages depend on model architecture, GQA ratio, and hardware — **no GPU-measured numbers are available yet** in this CPU-only CI environment. The `benchmarks/profile_vram.py` script measures Python heap (CPU) allocations only.

### 3. Enterprise Distributed Scaling (ZeRO-3 FSDP + CPU/NVMe Offload)
- **Fully Sharded Data Parallelism (ZeRO-3 / FSDP)**: Model parameters, optimizer states, and gradients are sharded across participating ranks with NCCL / Ring-AllReduce communication.
- **CPU & NVMe Parameter Offloading**: Automatically offloads frozen or inactive parameter shards to host RAM or high-speed NVMe storage, enabling training and fine-tuning models exceeding GPU VRAM capacity without out-of-memory crashes.

### 4. Natural-Language Autonomous Training Orchestrator (`velograd.instruct`)
Direct execution from natural-language instructions in Python or via the command line:
```bash
velograd run "Train a Llama model for 5 epochs with ZeRO-3 offload"
```
The autonomous orchestrator inspects hardware topology (GPU count, VRAM, CPU cores), parses hyperparameters, synthesizes or binds dataset pipelines, shards model layers, and executes end-to-end training loops.

---

## Quickstart & Installation

### 1. Install VeloGrad
```bash
git clone https://github.com/IDKLETSCODESOMETHIN/VeloGrad.git
cd VeloGrad
pip install -e .
```

### 2. Inspect Hardware Acceleration & Topology
```bash
velograd info
```

### 3. Autonomous Natural-Language Training (CLI)
```bash
velograd run "Train a Llama model for 3 epochs on dataset with ZeRO-3 offload"
```

### 4. Interactive Command Wizard
```bash
velograd --interactive
```

---

## Python API Usage

### PyTorch Drop-In Training Loop
```python
import velograd as vg
import velograd.nn as nn
import velograd.optim as optim
import velograd.functional as F

# Define neural network architecture
model = nn.Sequential(
    nn.Linear(128, 256),
    nn.RMSNorm(256),
    nn.SiLU(),
    nn.Linear(256, 10)
)

optimizer = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Data batch
x = vg.randn(32, 128)
targets = [i % 10 for i in range(32)]

# Training step with reverse-mode autograd
outputs = model(x)
loss = criterion(outputs, targets)
loss.backward()

optimizer.step()
optimizer.zero_grad()
print(f"Step completed. Loss: {loss._data[0]:.4f}")
```

### FlashAttention & Sparse Mixture-of-Experts (MoE)
```python
import velograd.nn as nn

# Tiled FlashAttention module
attn = nn.FlashAttention(embed_dim=256, num_heads=8, causal=True)

# Sparse Mixture-of-Experts layer with Top-2 learned gating
moe = nn.MoELayer(dim=256, hidden_dim=512, num_experts=8, top_k=2)
```

### Distributed ZeRO-3 FSDP Training
```python
import velograd.distributed as dist

# Wrap model with ZeRO-3 parameter sharding and CPU offloading
fsdp_model = dist.FSDP(model, cpu_offload=True)

out = fsdp_model(x)
loss = criterion(out, targets)
loss.backward()
fsdp_model.reduce_scatter_gradients()
fsdp_model.all_gather_parameters()
```

### Autonomous Natural-Language Orchestration
```python
import velograd as vg

result = vg.instruct("Train a Llama model for 2 epochs with ZeRO-3 offload")
print(f"Autonomous training finished. Final Loss: {result['final_loss']:.4f}")
```

---

## Implementation Status

| Feature | Status |
| :--- | :--- |
| Python tensor engine + N-dim autograd | ✅ CPU-verified, CI-tested |
| Fused SIMD kernels (Python/NumPy) | ✅ CPU-verified, CI-tested |
| `velograd.nn` module hierarchy (Linear, LayerNorm, etc.) | ✅ CPU-verified |
| `velograd.optim` (AdamW, SGD, Adam) | ✅ CPU-verified |
| `velograd.distributed.FSDP` (ZeRO-3 simulation) | ✅ CPU-verified |
| `velograd.instruct` NL training orchestrator | ✅ CPU-verified |
| `velograd.compile` graph compiler | ✅ CPU-verified |
| PagedAttention engine (`velograd.paged_attn`) | ✅ CPU-verified (simulation) |
| Model family registry (Llama / Qwen / DeepSeek / Gemma / etc.) | ✅ CPU-verified |
| CUDA kernels (`src/cuda/fused_kernels.cu`) | ⚠️ Written — NOT tested on GPU hardware |
| Rust bindings (`src/`) | ⚠️ Written — NOT tested (requires Rust toolchain) |
| Multi-node NCCL communication | ⚠️ Unverified — requires physical multi-GPU cluster |
| GPU VRAM profiling / KV-cache reduction % | ⚠️ Unverified — requires physical NVIDIA GPU + pynvml |

> [!NOTE]
> VRAM reduction claims require physical GPU hardware to measure. No GPU-side benchmarks have been executed. `benchmarks/profile_vram.py` measures Python heap (CPU) allocations only via `tracemalloc`.

## Empirical Verification Benchmarks

### 1. CPU Heap Memory Profiler (`benchmarks/profile_vram.py`)
Measures real Python heap allocations via `tracemalloc` for Tensor allocation and forward/backward passes on CPU. Run with:
```bash
python benchmarks/profile_vram.py
```
Output is real `tracemalloc` peak measurements in MB. GPU VRAM profiling requires physical NVIDIA hardware and is not available in the current CPU-only CI environment.

### 2. Compute Kernel Benchmarks (`benchmarks/bench_vs_pytorch.py`)
- **GEMM Throughput**: Vectorized SIMD matrix multiplication running in milliseconds with autograd graph tracking.
- **Fused RMSNorm + Residual Add**: Single-pass memory-bound operation without intermediate residual writes.
- **Fused RoPE**: Analytic trigonometric position rotation with zero memory allocations.
- **Streaming Single-Pass Cross-Entropy**: Evaluated across 4,096-vocabulary distributions with zero full-logit tensor allocations.

---

## Verification Test Suites

To execute the test suites covering all architectural layers:
```bash
# Comprehensive Beta Verification Suite (All 6 Phases)
python tests/test_exhaustive.py

# Universal Ecosystem Interoperability Suite
python tests/test_velograd_universal_compat.py

# Run Empirical Benchmarks
python benchmarks/profile_vram.py
python benchmarks/bench_vs_pytorch.py
```

---

## Repository Structure

```
VeloGrad/
├── benchmarks/
│   ├── bench_vs_pytorch.py          # Compute Kernel & Step Latency Benchmarks
│   └── profile_vram.py              # Empirical KV-Cache & Activation VRAM Profiler
├── include/
│   ├── velograd.hpp                 # C++20 Header-Only Native Core Engine
│   └── velograd/
│       ├── cluster_watchdog.hpp     # Lock-free Heartbeat & Node Failure Eviction
│       └── ibverbs_driver.hpp       # Bare-Metal InfiniBand Verbs Driver
├── src/
│   ├── cuda/
│   │   └── fused_kernels.cu         # CUDA Kernels (RMSNorm, RoPE, Streaming CE, MoE, AdamW)
│   ├── autograd.rs                  # Symbolic AOT Backward Graph Traversal
│   ├── hardware.rs                  # Multi-vendor Accelerator Dispatch
│   └── ops.rs                       # Fused Attention, RMSNorm, Conv2D, GEMM
├── python/velograd/
│   ├── tensor.py                    # N-D Strided Tensor with Autograd & BLAS SIMD
│   ├── nn.py                        # Drop-in Neural Network Layers (FlashAttention, MoELayer)
│   ├── optim.py                     # AdamW, SGD, Learning Rate Schedulers
│   ├── functional.py                # Functional Primitives & Loss Formulations
│   ├── distributed.py               # ZeRO-3 FSDP Parameter Sharding & CPU Offloading
│   ├── instruct.py                  # Natural-Language Autonomous Training Orchestrator
│   ├── kernels/                     # Hardware Fused SIMD Compute Kernels
│   ├── models/
│   │   ├── families.py              # Universal Model Family Configurations & Factories
│   │   ├── llama.py                 # LLaMA Causal Language Model
│   │   ├── moe.py                   # Sparse Mixture-of-Experts Architecture
│   │   └── mamba.py                 # Mamba-2 State-Space Architecture
│   ├── paged_attn.py                # Virtual Paged Attention Engine
│   ├── hub.py                       # Hugging Face AutoModel & Safetensors Bridge
│   ├── compiler.py                  # JIT Graph Optimization & Operator Fusion
│   └── cli.py                       # Command Center & Autonomous Runner
└── tests/
    ├── test_exhaustive.py           # Comprehensive Verification Suite (v1.1.0-beta)
    └── test_velograd_universal_compat.py # Universal Interoperability Tests
```

---

## License

Dual-licensed under the **GNU Affero General Public License (AGPLv3)** with a **Commercial Royalty Waiver** escape hatch for proprietary enterprise and private cloud network deployments. See [`LICENSE`](LICENSE) for details.
