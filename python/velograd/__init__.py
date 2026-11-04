# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE 4: Zero-Friction PyTorch Drop-In API Layer
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad: The Native-First, Bare-Metal Machine Learning Engine.
Zero-friction drop-in replacement for PyTorch with dynamic backward graph traversal,
N-dimensional strided tensor core, finite-difference verification (gradcheck), and live hardware detection.
"""

import sys
import os
import math
from typing import Tuple, Optional, List, Callable, Any

from .tensor import Tensor, zeros, ones, randn, tensor
from . import autograd
from . import backend
from . import nn
from . import optim
from . import functional
from . import models
from . import data
from . import utils
from . import distributed
from . import hub
from . import quantization
from . import peft
from . import export
from . import compiler
from . import paged_attn
from . import disaggregated
from . import telemetry
from . import triton_kernels
from . import checkpoint
from . import ragged
from . import coexecution
from . import rdma
from . import numerical_stability
from . import compat
from . import converter
from .trainer import AutoTrainer

__version__ = "1.0.0-enterprise-production"

dist = distributed
compile = compiler.compile
mps = backend.mps
AutoModelForCausalLM = hub.AutoModelForCausalLM
AutoTokenizer = hub.AutoTokenizer
from_numpy = compat.from_array
to_numpy = compat.to_numpy
from_torch = compat.from_torch
to_torch = compat.to_torch

__all__ = [
    "__version__", "Tensor", "zeros", "ones", "randn", "tensor", "matmul", "save", "load",
    "nn", "optim", "functional", "models", "data", "utils", "autograd", "backend", "cuda", "mps", "AutoTrainer",
    "distributed", "dist", "hub", "quantization", "peft", "export", "compiler", "compile",
    "paged_attn", "disaggregated", "telemetry", "triton_kernels", "checkpoint", "ragged", "coexecution",
    "rdma", "numerical_stability", "compat", "converter", "AutoModelForCausalLM", "AutoTokenizer",
    "from_numpy", "to_numpy", "from_torch", "to_torch"
]

def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication of two tensors."""
    return a.matmul(b)

def save(obj: Any, f: str):
    """Saves an object (e.g. model state_dict or tensor) to disk."""
    import json
    with open(f, "w", encoding="utf-8") as file:
        json.dump(obj, file)

def load(f: str) -> Any:
    """Loads an object (e.g. model state_dict) from disk."""
    import json
    with open(f, "r", encoding="utf-8") as file:
        return json.load(file)

# Expose cuda namespace mirroring torch.cuda
class _CudaNamespace:
    @staticmethod
    def is_available() -> bool:
        return backend.is_available()

    @staticmethod
    def device_count() -> int:
        return backend.device_count()

    @staticmethod
    def get_device_name(device_id: int = 0) -> str:
        return backend.get_device_name(device_id)

cuda = _CudaNamespace()
