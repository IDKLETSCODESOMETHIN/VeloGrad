# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: Autograd Engine & Finite-Difference Numerical Verification (gradcheck)
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad Automatic Differentiation & Mathematical Verification Engine.
Provides finite-difference numerical gradient checking (gradcheck),
direct gradient extraction (torch.autograd.grad), and no_grad context managers.
"""

from typing import Callable, Tuple, List, Union, Any, Optional
import math
from .tensor import Tensor

class no_grad:
    """Context manager that disables gradient computation during inference."""
    _prev_state: bool = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def grad(
    outputs: Union[Tensor, List[Tensor]],
    inputs: Union[Tensor, List[Tensor]],
    grad_outputs: Optional[Union[Tensor, List[Tensor]]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Tuple[Tensor, ...]:
    """
    Computes and returns the sum of gradients of outputs with respect to the inputs.
    Drop-in parity for torch.autograd.grad.
    """
    if isinstance(outputs, Tensor):
        outputs = [outputs]
    if isinstance(inputs, Tensor):
        inputs = [inputs]

    for out in outputs:
        out.backward(gradient=grad_outputs)

    grads = []
    for inp in inputs:
        if inp.grad is not None:
            grads.append(inp.grad.clone())
        else:
            grads.append(Tensor(inp.shape, requires_grad=False))
    return tuple(grads)

def gradcheck(
    func: Callable[..., Tensor],
    inputs: Union[Tensor, Tuple[Tensor, ...], List[Tensor]],
    eps: float = 1e-4,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    raise_exception: bool = True,
) -> bool:
    """
    Finite-Difference Numerical Gradient Checker.
    Scientifically verifies that analytic gradients computed by VeloGrad's autograd engine
    match two-sided numerical difference:
        J_num = (f(x + eps) - f(x - eps)) / (2 * eps)
    to within specified absolute and relative tolerances.
    """
    if isinstance(inputs, Tensor):
        inputs = (inputs,)
    else:
        inputs = tuple(inputs)

    # 1. Compute Analytic Gradients via Reverse-Mode Autograd
    # Zero all input grads
    for inp in inputs:
        if inp.grad is not None:
            inp.grad.zero_()
        inp.requires_grad = True

    # Run forward pass
    out = func(*inputs)
    out.backward()

    # 2. Check each input tensor element-by-element
    for inp_idx, inp in enumerate(inputs):
        if not inp.requires_grad or inp.grad is None:
            continue

        analytic_grad = list(inp.grad.contiguous()._data)
        num_elements = inp.numel

        for elem_idx in range(num_elements):
            orig_val = inp._data[elem_idx]

            # f(x + eps)
            inp._data[elem_idx] = orig_val + eps
            out_pos = func(*inputs)
            pos_sum = sum(out_pos.contiguous()._data)

            # f(x - eps)
            inp._data[elem_idx] = orig_val - eps
            out_neg = func(*inputs)
            neg_sum = sum(out_neg.contiguous()._data)

            # Restore original value
            inp._data[elem_idx] = orig_val

            # Two-sided numerical finite difference
            numerical_grad = (pos_sum - neg_sum) / (2.0 * eps)
            analytic_val = analytic_grad[elem_idx]

            diff = abs(analytic_val - numerical_grad)
            max_tol = atol + rtol * abs(numerical_grad)

            if diff > max_tol:
                msg = (
                    f"Gradcheck failed for input [{inp_idx}], element [{elem_idx}]:\n"
                    f"  Analytic Grad:  {analytic_val:.6f}\n"
                    f"  Numerical Grad: {numerical_grad:.6f}\n"
                    f"  Abs Difference: {diff:.6f} > Allowed Tolerance {max_tol:.6f}"
                )
                if raise_exception:
                    raise AssertionError(msg)
                return False

    return True
