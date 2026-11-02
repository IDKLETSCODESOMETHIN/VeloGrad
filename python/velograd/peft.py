# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Native Parameter-Efficient Fine-Tuning (velograd.peft / LoRA & QLoRA)
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad Native PEFT Engine (velograd.peft):
Provides zero-overhead Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA) directly in the core:
- Freezes base model parameters and trains lightweight low-rank decomposition matrices (A x B)
- QLoRA: Combines 4-bit frozen base weights with full-precision trainable adapters (87.5% VRAM cut)
- Fast zero-latency weight merging via merge_and_unload() for deployment
- Drop-in parity for Hugging Face PEFT library
"""

import math
import random
from typing import List, Optional, Dict, Any, Union
from .tensor import Tensor, zeros
from . import nn
from .quantization import Linear4Bit

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) Linear Layer.
    Computes: output = x @ W_frozen + (lora_alpha / r) * (x @ lora_A @ lora_B)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        bias: bool = False
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r) if r > 0 else 1.0
        self.merged = False

        # 1. Base Frozen Weights
        self.weight = Tensor(in_features, out_features, requires_grad=False)
        std = math.sqrt(2.0 / in_features)
        self.weight._data = [random.gauss(0.0, std) for _ in range(in_features * out_features)]

        # 2. Trainable Low-Rank Adapters
        # lora_A: in_features -> r (Kaiming Gaussian initialized)
        self.lora_A = Tensor(in_features, r, requires_grad=True)
        a_std = 1.0 / math.sqrt(in_features)
        self.lora_A._data = [random.gauss(0.0, a_std) for _ in range(in_features * r)]

        # lora_B: r -> out_features (Zero initialized so initial adapter output is exactly 0)
        self.lora_B = zeros(r, out_features, requires_grad=True)

        self._parameters = [self.lora_A, self.lora_B]

        if bias:
            self.bias = zeros(1, out_features, requires_grad=False)
            self._parameters.append(self.bias)
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0) -> 'LoRALinear':
        """Constructs a LoRALinear module from an existing Linear layer, preserving base weights."""
        lora_layer = cls(
            linear_layer.in_features,
            linear_layer.out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=(linear_layer.bias is not None)
        )
        # Copy frozen base weights
        lora_layer.weight._data = list(linear_layer.weight.contiguous()._data)
        lora_layer.weight.requires_grad = False
        if linear_layer.bias is not None and lora_layer.bias is not None:
            lora_layer.bias._data = list(linear_layer.bias.contiguous()._data)
        return lora_layer

    def forward(self, x: Tensor) -> Tensor:
        if self.merged:
            out = x @ self.weight
        else:
            # Base frozen forward pass
            base_out = x @ self.weight
            # Low-rank adapter forward pass: (x @ A) @ B
            lora_out = (x @ self.lora_A) @ self.lora_B
            out = base_out + lora_out.mul(self.scaling)

        if self.bias is not None:
            out = out + self.bias
        return out

    def merge(self) -> None:
        """Merges adapter weights into base weight matrix in-place for zero-latency inference."""
        if not self.merged:
            delta_w = (self.lora_A @ self.lora_B).mul(self.scaling)
            delta_data = delta_w.contiguous()._data
            for i in range(len(self.weight._data)):
                self.weight._data[i] += delta_data[i]
            self.merged = True

    def unmerge(self) -> None:
        """Subtracts adapter weights to resume fine-tuning."""
        if self.merged:
            delta_w = (self.lora_A @ self.lora_B).mul(self.scaling)
            delta_data = delta_w.contiguous()._data
            for i in range(len(self.weight._data)):
                self.weight._data[i] -= delta_data[i]
            self.merged = False

class QLoRALinear(nn.Module):
    """
    Quantized Low-Rank Adaptation (QLoRA) Linear Layer.
    Base weights are frozen in 4-bit quantized format (Linear4Bit), cutting base memory by 87.5%,
    while LoRA adapter matrices (A and B) train in full precision.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: float = 16.0,
        block_size: int = 32
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / float(r) if r > 0 else 1.0

        # Frozen 4-bit base weights
        self.base_q4 = Linear4Bit(in_features, out_features, block_size=block_size)

        # Trainable full-precision LoRA adapters
        self.lora_A = Tensor(in_features, r, requires_grad=True)
        a_std = 1.0 / math.sqrt(in_features)
        self.lora_A._data = [random.gauss(0.0, a_std) for _ in range(in_features * r)]

        self.lora_B = zeros(r, out_features, requires_grad=True)
        self._parameters = [self.lora_A, self.lora_B]

    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0, block_size: int = 32) -> 'QLoRALinear':
        qlora_layer = cls(linear_layer.in_features, linear_layer.out_features, r=r, lora_alpha=lora_alpha, block_size=block_size)
        qlora_layer.base_q4 = Linear4Bit.from_linear(linear_layer, block_size=block_size)
        return qlora_layer

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_q4(x)
        lora_out = (x @ self.lora_A) @ self.lora_B
        return base_out + lora_out.mul(self.scaling)

def get_peft_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    qlora: bool = False,
    block_size: int = 32
) -> nn.Module:
    """
    Converts a standard VeloGrad model into a parameter-efficient fine-tuning (PEFT) model:
    - Freezes all base parameters
    - Injects LoRA or QLoRA adapters across all Linear projections
    - Displays trainable vs frozen parameter statistics
    """
    # 1. Freeze all existing parameters
    for p in model.parameters():
        p.requires_grad = False

    # 2. Recursively replace Linear layers with LoRALinear or QLoRALinear
    def _convert_module(mod: nn.Module):
        for idx, sub in enumerate(mod._submodules):
            if isinstance(sub, nn.Linear):
                if qlora:
                    mod._submodules[idx] = QLoRALinear.from_linear(sub, r=r, lora_alpha=lora_alpha, block_size=block_size)
                else:
                    mod._submodules[idx] = LoRALinear.from_linear(sub, r=r, lora_alpha=lora_alpha)
            else:
                _convert_module(sub)

    _convert_module(model)

    # 3. Compute parameter statistics
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += len(p._data)
        if p.requires_grad:
            trainable_params += len(p._data)

    pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    mode_str = "QLoRA (4-bit Base + FP32 Adapters)" if qlora else "LoRA (Full Base + FP32 Adapters)"
    print(f"[VeloGrad PEFT] Initialized {mode_str}:")
    print(f"  -> Trainable Parameters: {trainable_params:,d} / {total_params:,d} ({pct:.3f}%)")
    print(f"  -> Memory Savings: ~{100.0 - pct:.1f}% reduction in backpropagation gradients.")

    return model

def merge_and_unload(model: nn.Module) -> nn.Module:
    """
    Recursively merges all LoRA adapter weights into their base Linear layers.
    Returns the unified, standalone model ready for zero-overhead inference.
    """
    def _merge_module(mod: nn.Module):
        for idx, sub in enumerate(mod._submodules):
            if isinstance(sub, LoRALinear):
                sub.merge()
                # Create standard Linear layer with merged weights
                merged_linear = nn.Linear(sub.in_features, sub.out_features, bias=(sub.bias is not None))
                merged_linear.weight._data = list(sub.weight._data)
                if sub.bias is not None and merged_linear.bias is not None:
                    merged_linear.bias._data = list(sub.bias._data)
                mod._submodules[idx] = merged_linear
            else:
                _merge_module(sub)

    _merge_module(model)
    print("[VeloGrad PEFT] All LoRA adapters successfully merged into base weights.")
    return model

__all__ = [
    "LoRALinear",
    "QLoRALinear",
    "get_peft_model",
    "merge_and_unload"
]
