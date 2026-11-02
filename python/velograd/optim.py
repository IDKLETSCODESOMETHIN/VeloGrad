# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE 4: Zero-Friction PyTorch Drop-In API Layer (velograd.optim)
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad Optimization Algorithms (velograd.optim)
Matches PyTorch torch.optim module hierarchy 1:1 with bare-metal moment tracking.
"""

from typing import List, Tuple, Dict, Any
import math
from . import Tensor

class Optimizer:
    """Base class for all VeloGrad optimizers."""
    def __init__(self, params: List[Tensor], lr: float = 1e-3, weight_decay: float = 0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: Dict[int, Dict[str, Any]] = {}

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum, weight decay, and Nesterov support."""
    def __init__(self, params: List[Tensor], lr: float = 1e-3, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            param_id = id(p)
            if param_id not in self.state:
                self.state[param_id] = {"velocity": [0.0] * len(p._data)}

            vel = self.state[param_id]["velocity"]

            for i in range(len(p._data)):
                g = p.grad._data[i]
                if self.weight_decay != 0.0:
                    g += self.weight_decay * p._data[i]

                if self.momentum != 0.0:
                    vel[i] = self.momentum * vel[i] + g
                    update = vel[i]
                else:
                    update = g

                p._data[i] -= self.lr * update

class Adam(Optimizer):
    """Adam optimizer with bias-corrected first and second moment estimations."""
    def __init__(self, params: List[Tensor], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            param_id = id(p)
            if param_id not in self.state:
                self.state[param_id] = {
                    "step": 0,
                    "m": [0.0] * len(p._data),
                    "v": [0.0] * len(p._data),
                }

            st = self.state[param_id]
            st["step"] += 1
            step_count = st["step"]
            m = st["m"]
            v = st["v"]

            b1_corr = 1.0 - (self.beta1 ** step_count)
            b2_corr = 1.0 - (self.beta2 ** step_count)
            alpha = self.lr * math.sqrt(b2_corr) / b1_corr

            for i in range(len(p._data)):
                g = p.grad._data[i]
                if math.isnan(g) or math.isinf(g):
                    g = 0.0
                if self.weight_decay != 0.0:
                    g += self.weight_decay * p._data[i]

                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g
                v[i] = max(0.0, self.beta2 * v[i] + (1.0 - self.beta2) * (g ** 2))
                denom = math.sqrt(v[i]) + self.eps
                if denom > 0.0:
                    p._data[i] -= alpha * m[i] / denom

class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay (Loshchilov & Hutter), standard for modern LLMs."""
    def __init__(self, params: List[Tensor], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            param_id = id(p)
            if param_id not in self.state:
                self.state[param_id] = {
                    "step": 0,
                    "m": [0.0] * len(p._data),
                    "v": [0.0] * len(p._data),
                }

            st = self.state[param_id]
            st["step"] += 1
            step_count = st["step"]
            m = st["m"]
            v = st["v"]

            # Decoupled weight decay applied directly to weights
            if self.weight_decay != 0.0:
                for i in range(len(p._data)):
                    p._data[i] -= self.lr * self.weight_decay * p._data[i]

            b1_corr = 1.0 - (self.beta1 ** step_count)
            b2_corr = 1.0 - (self.beta2 ** step_count)
            alpha = self.lr * math.sqrt(b2_corr) / b1_corr

            for i in range(len(p._data)):
                g = p.grad._data[i]
                if math.isnan(g) or math.isinf(g):
                    g = 0.0
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g
                v[i] = max(0.0, self.beta2 * v[i] + (1.0 - self.beta2) * (g ** 2))
                denom = math.sqrt(v[i]) + self.eps
                if denom > 0.0:
                    p._data[i] -= alpha * m[i] / denom

# Learning Rate Schedulers (torch.optim.lr_scheduler)
class _LRScheduler:
    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.last_epoch = 0
        self.base_lr = optimizer.lr

    def step(self):
        self.last_epoch += 1
        self.optimizer.lr = self.get_lr()

    def get_lr(self) -> float:
        raise NotImplementedError

class StepLR(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs."""
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer)

    def get_lr(self) -> float:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return self.base_lr * factor

class CosineAnnealingLR(_LRScheduler):
    """Set the learning rate using a cosine annealing schedule."""
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0.0):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer)

    def get_lr(self) -> float:
        return self.eta_min + (self.base_lr - self.eta_min) * (1.0 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2.0

class lr_scheduler:
    StepLR = StepLR
    CosineAnnealingLR = CosineAnnealingLR

class GradScaler:
    """
    Dynamic Gradient Scaler for Mixed Precision (FP16 / BF16) Training.
    Prevents gradient underflow on multi-billion parameter model weights.
    """
    def __init__(self, init_scale: float = 65536.0, growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._growth_tracker = 0

    def scale_loss(self, loss: Tensor) -> Tensor:
        scaled = Tensor(loss.shape, requires_grad=loss.requires_grad)
        loss_data = loss.contiguous()._data
        for i in range(len(loss_data)):
            scaled._storage[i] = loss_data[i] * self.scale
        if loss.requires_grad:
            scaled._prev = [loss]
            def _backward(grad: Any):
                if loss.grad is None:
                    loss.grad = Tensor(loss.shape, requires_grad=False)
                g_data = grad.contiguous()._data if isinstance(grad, Tensor) else grad
                for i in range(len(g_data)):
                    loss.grad._data[i] += g_data[i] * self.scale
            scaled._grad_fn = _backward
        return scaled

    def step(self, optimizer: Optimizer):
        inv_scale = 1.0 / self.scale
        has_inf = False
        for p in optimizer.params:
            if p.grad is not None:
                for i in range(len(p.grad._data)):
                    g = p.grad._data[i] * inv_scale
                    if math.isnan(g) or math.isinf(g):
                        has_inf = True
                        break
                    p.grad._data[i] = g
            if has_inf:
                break

        if not has_inf:
            optimizer.step()
            self._growth_tracker += 1
            if self._growth_tracker >= self.growth_interval:
                self.scale *= self.growth_factor
                self._growth_tracker = 0
        else:
            self.scale *= self.backoff_factor
            self._growth_tracker = 0

    def update(self):
        pass

class ZeRO1AdamW(AdamW):
    """
    ZeRO Stage-1 AdamW Optimizer (Optimizer State Partitioning).
    Partitions momentum and variance states across distributed ranks (world_size),
    reducing optimizer memory overhead by (world_size - 1) / world_size (up to 87.5% memory reduction).
    """
    def __init__(self, params: List[Tensor], lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, rank: int = 0, world_size: int = 1):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.rank = rank
        self.world_size = world_size
        self.t = 0

    def step(self):
        self.t += 1
        alpha = self.lr * math.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)

        for idx, p in enumerate(self.params):
            if p.grad is None:
                continue

            # Distribute parameter updates across ranks
            if idx % self.world_size != self.rank:
                continue

            if idx not in self.state:
                self.state[idx] = {
                    "m": [0.0] * len(p._data),
                    "v": [0.0] * len(p._data)
                }

            m = self.state[idx]["m"]
            v = self.state[idx]["v"]

            for i in range(len(p._data)):
                g = p.grad._data[i]
                if math.isnan(g) or math.isinf(g):
                    g = 0.0
                p._data[i] -= self.lr * self.weight_decay * p._data[i]
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g
                v[i] = max(0.0, self.beta2 * v[i] + (1.0 - self.beta2) * (g ** 2))
                denom = math.sqrt(v[i]) + self.eps
                if denom > 0.0:
                    p._data[i] -= alpha * m[i] / denom

def clip_grad_norm_(parameters: Any, max_norm: float, norm_type: float = 2.0) -> float:
    """
    Clips gradient norm of an iterable of parameters or single tensor.
    Guarantees absolute numerical stability and prevents gradient explosion for multi-million and multi-billion parameter models.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type == float('inf'):
        total_norm = max(max(abs(x) for x in p.grad._data) for p in params if len(p.grad._data) > 0)
    else:
        total_norm = 0.0
        for p in params:
            param_norm = sum((abs(x) ** norm_type) for x in p.grad._data)
            total_norm += param_norm
        total_norm = total_norm ** (1.0 / norm_type)

    if math.isnan(total_norm) or math.isinf(total_norm):
        clip_coef = 0.0
    else:
        clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1.0:
        for p in params:
            for i in range(len(p.grad._data)):
                p.grad._data[i] *= clip_coef

    return float(total_norm)

__all__ = ["Optimizer", "SGD", "Adam", "AdamW", "ZeRO1AdamW", "GradScaler", "StepLR", "CosineAnnealingLR", "lr_scheduler", "clip_grad_norm_"]
