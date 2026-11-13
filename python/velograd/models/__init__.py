# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# PRODUCTION MODELS REPOSITORY
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""VeloGrad Standard Production Models (LLaMA, ResNet, MoE, Multimodal VLM)."""

from .llama import LLaMAModel, LLaMATransformerBlock, LLaMAFeedForward
from .resnet import ResNet18, BasicBlock
from .moe import MoEModel, MoETransformerBlock, MoEFeedForward, MoEGate
from .multimodal import VisionLanguageModel, VisionTransformer, PatchEmbedding, MultimodalProjector
from .mamba import Mamba2Model, Mamba2Block
from .families import MODEL_REGISTRY, build_model_from_family

__all__ = [
    "LLaMAModel", "LLaMATransformerBlock", "LLaMAFeedForward", "ResNet18", "BasicBlock",
    "MoEModel", "MoETransformerBlock", "MoEFeedForward", "MoEGate",
    "VisionLanguageModel", "VisionTransformer", "PatchEmbedding", "MultimodalProjector",
    "Mamba2Model", "Mamba2Block", "MODEL_REGISTRY", "build_model_from_family"
]
