# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Natural Language Training Parser (Plain-English Zero-Code Intent Engine)
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad Natural Language Intent Parser:
Translates human plain-English instructions into fully configured, executable neural network
training pipelines with zero code required from the user.
"""

import re
from typing import Dict, Any

def parse_training_prompt(prompt: str) -> Dict[str, Any]:
    """
    Parses a plain-English user prompt and extracts model architecture and hyperparameters.
    
    Examples:
      - "Train a 4-layer LLaMA model with 64 dimensions for 10 epochs at 0.002 learning rate"
      - "Fine-tune ResNet-18 for 8 epochs with batch size 32"
      - "Train an MLP with 64 units for 12 epochs"
    """
    p_lower = prompt.lower().strip()

    # 1. Detect Architecture
    model_type = "llama"
    if any(k in p_lower for k in ["moe", "mixture of experts", "expert", "sparse"]):
        model_type = "moe"
    elif any(k in p_lower for k in ["vlm", "multimodal", "vision language"]):
        model_type = "vlm"
    elif any(k in p_lower for k in ["resnet", "vision", "conv", "cnn", "image"]):
        model_type = "resnet"
    elif any(k in p_lower for k in ["mlp", "linear", "perceptron", "dense", "feedforward"]):
        model_type = "mlp"
    elif any(k in p_lower for k in ["llama", "transformer", "attention", "causal", "llm"]):
        model_type = "llama"

    # Defaults
    epochs = 5
    dim = 32 if model_type != "resnet" else 3
    layers = 2
    heads = 2
    batch_size = 16
    lr = 0.001

    # 2. Extract Epochs
    epoch_match = re.search(r'\b(\d+)\s*(?:epochs?|rounds?|iters?)\b', p_lower)
    if epoch_match:
        epochs = max(1, min(500, int(epoch_match.group(1))))

    # 3. Extract Layers
    layer_match = re.search(r'\b(\d+)[-\s]*(?:layers?|blocks?|depth)\b', p_lower)
    if layer_match:
        layers = max(1, min(64, int(layer_match.group(1))))

    # 4. Extract Dimension
    dim_match = re.search(r'\b(\d+)[-\s]*(?:dims?|dimensions?|units?|features?|hidden)\b', p_lower)
    if dim_match:
        dim = max(8, min(4096, int(dim_match.group(1))))

    # 5. Extract Learning Rate
    lr_match = re.search(r'\b(?:lr|learning\s*rate|at\s*rate)\s*[:=]?\s*([0-9\.]+(?:e-?\d+)?)\b', p_lower)
    if lr_match:
        try:
            lr = float(lr_match.group(1))
        except ValueError:
            pass

    # 6. Extract Batch Size
    bs_match = re.search(r'\b(?:batch\s*size|batch|bs)\s*[:=]?\s*(\d+)\b', p_lower)
    if bs_match:
        batch_size = max(1, min(1024, int(bs_match.group(1))))

    if model_type == "llama":
        heads = max(1, min(8, dim // 16)) if dim >= 16 else 1

    return {
        "model": model_type,
        "epochs": epochs,
        "layers": layers,
        "dim": dim,
        "heads": heads,
        "batch_size": batch_size,
        "lr": lr,
        "prompt": prompt,
        "parsed_summary": f"Architecture: {model_type.upper()} | {layers} Layers | Dim: {dim} | {epochs} Epochs | Batch: {batch_size} | LR: {lr}"
    }

__all__ = ["parse_training_prompt"]
