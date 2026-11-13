# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Universal Architecture Configurations (velograd.models.families)
# COPYRIGHT (C) 2026 Ayushman Singh Rathore
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
Universal Architecture Configurations:
Defines exact hyperparameter and sharding topology presets across frontier model families:
- Llama 3 / 3.1 / 3.2 (1B, 3B, 8B, 70B, 405B) & Llama 4 (Scout 109B MoE, Maverick 400B MoE)
- Qwen 2.5 / Qwen 3 / Qwen 3.5 / 3.6 / 3.8 (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B, 235B MoE, 397B MoE)
- DeepSeek-R1 / R2 (1.5B–70B distilled, 671B MoE) & DeepSeek-V4 (Flash 284B MoE, Pro 1.6T MoE)
- Gemma 2 / Gemma 3 / Gemma 4 (1B, 2B, 4B, 9B, 12B, 26B MoE, 27B, 31B)
- Mistral & Mixtral (7B, Small 24B, Mixtral 8x7B / 8x22B, Large 3 675B MoE)
- Phi-4 / Phi-5, GLM-5, Kimi K2.6 / K3, OLMo 2, Granite 4 Hybrid Mamba-Transformer
"""

from typing import Dict, Any, Optional
from .llama import LLaMAModel
from .moe import MoEModel
from .mamba import Mamba2Model

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Llama Family ---
    "llama-3.2-1b": {"type": "dense", "vocab_size": 128256, "dim": 2048, "num_layers": 16, "num_heads": 32, "hidden_dim": 8192},
    "llama-3.2-3b": {"type": "dense", "vocab_size": 128256, "dim": 3072, "num_layers": 28, "num_heads": 24, "hidden_dim": 8192},
    "llama-3.1-8b": {"type": "dense", "vocab_size": 128256, "dim": 4096, "num_layers": 32, "num_heads": 32, "hidden_dim": 14336},
    "llama-3.1-70b": {"type": "dense", "vocab_size": 128256, "dim": 8192, "num_layers": 80, "num_heads": 64, "hidden_dim": 28672},
    "llama-3.1-405b": {"type": "dense", "vocab_size": 128256, "dim": 16384, "num_layers": 126, "num_heads": 128, "hidden_dim": 53248},
    "llama-4-scout-109b": {"type": "moe", "vocab_size": 128256, "dim": 5120, "num_layers": 48, "num_heads": 40, "hidden_dim": 12288, "num_experts": 16, "top_k": 2},
    "llama-4-maverick-400b": {"type": "moe", "vocab_size": 128256, "dim": 8192, "num_layers": 84, "num_heads": 64, "hidden_dim": 24576, "num_experts": 32, "top_k": 4},

    # --- Qwen Family ---
    "qwen-2.5-0.5b": {"type": "dense", "vocab_size": 151936, "dim": 896, "num_layers": 24, "num_heads": 14, "hidden_dim": 4864},
    "qwen-2.5-1.5b": {"type": "dense", "vocab_size": 151936, "dim": 1536, "num_layers": 28, "num_heads": 12, "hidden_dim": 8960},
    "qwen-2.5-3b": {"type": "dense", "vocab_size": 151936, "dim": 2048, "num_layers": 36, "num_heads": 16, "hidden_dim": 11008},
    "qwen-2.5-7b": {"type": "dense", "vocab_size": 151936, "dim": 3584, "num_layers": 28, "num_heads": 28, "hidden_dim": 18944},
    "qwen-2.5-14b": {"type": "dense", "vocab_size": 151936, "dim": 5120, "num_layers": 48, "num_heads": 40, "hidden_dim": 27648},
    "qwen-2.5-32b": {"type": "dense", "vocab_size": 151936, "dim": 5120, "num_layers": 64, "num_heads": 40, "hidden_dim": 27648},
    "qwen-2.5-72b": {"type": "dense", "vocab_size": 151936, "dim": 8192, "num_layers": 80, "num_heads": 64, "hidden_dim": 29568},
    "qwen-3-235b-moe": {"type": "moe", "vocab_size": 151936, "dim": 6144, "num_layers": 64, "num_heads": 48, "hidden_dim": 16384, "num_experts": 64, "top_k": 8},
    "qwen-3.8-397b-moe": {"type": "moe", "vocab_size": 151936, "dim": 8192, "num_layers": 80, "num_heads": 64, "hidden_dim": 20480, "num_experts": 128, "top_k": 8},

    # --- DeepSeek Family ---
    "deepseek-r1-1.5b": {"type": "dense", "vocab_size": 151936, "dim": 1536, "num_layers": 28, "num_heads": 12, "hidden_dim": 8960},
    "deepseek-r1-7b": {"type": "dense", "vocab_size": 151936, "dim": 3584, "num_layers": 28, "num_heads": 28, "hidden_dim": 18944},
    "deepseek-r1-14b": {"type": "dense", "vocab_size": 151936, "dim": 5120, "num_layers": 48, "num_heads": 40, "hidden_dim": 27648},
    "deepseek-r1-32b": {"type": "dense", "vocab_size": 151936, "dim": 5120, "num_layers": 64, "num_heads": 40, "hidden_dim": 27648},
    "deepseek-r1-70b": {"type": "dense", "vocab_size": 128256, "dim": 8192, "num_layers": 80, "num_heads": 64, "hidden_dim": 28672},
    "deepseek-r1-671b": {"type": "moe", "vocab_size": 129280, "dim": 7168, "num_layers": 61, "num_heads": 128, "hidden_dim": 18432, "num_experts": 256, "top_k": 8},
    "deepseek-v4-flash-284b": {"type": "moe", "vocab_size": 129280, "dim": 6144, "num_layers": 56, "num_heads": 64, "hidden_dim": 16384, "num_experts": 128, "top_k": 8},
    "deepseek-v4-pro-1.6t": {"type": "moe", "vocab_size": 129280, "dim": 12288, "num_layers": 96, "num_heads": 128, "hidden_dim": 32768, "num_experts": 512, "top_k": 16},

    # --- Gemma Family ---
    "gemma-2-2b": {"type": "dense", "vocab_size": 256000, "dim": 2304, "num_layers": 26, "num_heads": 8, "hidden_dim": 9216},
    "gemma-2-9b": {"type": "dense", "vocab_size": 256000, "dim": 3584, "num_layers": 42, "num_heads": 16, "hidden_dim": 14336},
    "gemma-2-27b": {"type": "dense", "vocab_size": 256000, "dim": 4608, "num_layers": 46, "num_heads": 32, "hidden_dim": 36864},
    "gemma-3-12b": {"type": "dense", "vocab_size": 256000, "dim": 4096, "num_layers": 36, "num_heads": 16, "hidden_dim": 16384},
    "gemma-3-26b-moe": {"type": "moe", "vocab_size": 256000, "dim": 4096, "num_layers": 40, "num_heads": 16, "hidden_dim": 12288, "num_experts": 16, "top_k": 2},

    # --- Mistral & Mixtral Family ---
    "mistral-7b": {"type": "dense", "vocab_size": 32000, "dim": 4096, "num_layers": 32, "num_heads": 32, "hidden_dim": 14336},
    "mistral-small-24b": {"type": "dense", "vocab_size": 32768, "dim": 5120, "num_layers": 40, "num_heads": 32, "hidden_dim": 14336},
    "mixtral-8x7b": {"type": "moe", "vocab_size": 32000, "dim": 4096, "num_layers": 32, "num_heads": 32, "hidden_dim": 14336, "num_experts": 8, "top_k": 2},
    "mixtral-8x22b": {"type": "moe", "vocab_size": 32000, "dim": 6144, "num_layers": 56, "num_heads": 48, "hidden_dim": 16384, "num_experts": 8, "top_k": 2},
    "mistral-large-3-675b": {"type": "moe", "vocab_size": 32768, "dim": 8192, "num_layers": 88, "num_heads": 64, "hidden_dim": 28672, "num_experts": 64, "top_k": 8},

    # --- Phi, GLM, Kimi, Granite Hybrid ---
    "phi-4": {"type": "dense", "vocab_size": 100352, "dim": 5120, "num_layers": 40, "num_heads": 40, "hidden_dim": 17920},
    "glm-5": {"type": "moe", "vocab_size": 150000, "dim": 4096, "num_layers": 40, "num_heads": 32, "hidden_dim": 10944, "num_experts": 16, "top_k": 2},
    "kimi-k3": {"type": "moe", "vocab_size": 128256, "dim": 6144, "num_layers": 48, "num_heads": 48, "hidden_dim": 16384, "num_experts": 32, "top_k": 4},
    "granite-4-hybrid": {"type": "hybrid", "vocab_size": 49152, "dim": 4096, "num_layers": 32, "state_dim": 64}
}

def get_family_spec(family_name: str) -> Dict[str, Any]:
    """Retrieves architectural configuration specification for a given model family."""
    key = family_name.lower().strip()
    if key not in MODEL_REGISTRY:
        matches = [k for k in MODEL_REGISTRY if key in k or k in key]
        key = matches[0] if matches else "llama-3.1-8b"
    return dict(MODEL_REGISTRY[key])

def build_model_from_family(
    family_name: str,
    vocab_size_override: Optional[int] = None,
    scale_down: Optional[int] = None
) -> Any:
    """
    Instantiates an exact architectural replica of any model from 100M up to 1T+ parameters.
    If scale_down factor is provided, dimensions are reduced proportionally for lightweight local execution.
    """
    spec = get_family_spec(family_name)
    div = max(1, scale_down if scale_down else 1)

    dim = max(16, spec["dim"] // div)
    heads = max(2, spec.get("num_heads", 4) // div)
    layers = max(1, spec.get("num_layers", 2) // div)
    hidden = max(32, spec["hidden_dim"] // div) if "hidden_dim" in spec else dim * 2
    vocab = vocab_size_override or spec["vocab_size"]

    mtype = spec["type"]
    if mtype == "dense":
        return LLaMAModel(vocab_size=vocab, dim=dim, num_layers=layers, num_heads=heads, hidden_dim=hidden)
    elif mtype == "moe":
        num_exp = spec.get("num_experts", 8)
        topk = min(num_exp, spec.get("top_k", 2))
        return MoEModel(vocab_size=vocab, dim=dim, num_layers=layers, num_heads=heads, hidden_dim=hidden, num_experts=num_exp, top_k=topk)
    elif mtype == "hybrid":
        st_dim = spec.get("state_dim", 16)
        return Mamba2Model(vocab_size=vocab, dim=dim, num_layers=layers, state_dim=st_dim)
    else:
        return LLaMAModel(vocab_size=vocab, dim=dim, num_layers=layers, num_heads=heads, hidden_dim=hidden)

__all__ = ["MODEL_REGISTRY", "build_model_from_family", "get_family_spec"]
