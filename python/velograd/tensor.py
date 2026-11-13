# =================================================================================================
# PROJECT: VeloGrad Engine Core (Bare-Metal Neural Compute Framework)
# MODULE: N-Dimensional Strided Tensor Core with Zero-Copy Views & Autograd
# COPYRIGHT (C) 2026 VeloGrad Architecture & Systems Core Team
# AGPLv3 + Commercial Royalty Waiver Dual License
# =================================================================================================

"""
VeloGrad N-Dimensional Strided Tensor Engine.
Provides arbitrary dimension tensors with zero-copy views, strided memory navigation,
NumPy/PyTorch compatible broadcasting, batched matrix multiplication, and dynamic reverse autograd.
"""

from typing import Tuple, List, Optional, Callable, Union, Any
import math
import random

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

def _compute_strides(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Computes contiguous C-order row-major strides for a given shape."""
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)

def _broadcast_shapes(s1: Tuple[int, ...], s2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Computes the broadcasted output shape following standard NumPy/PyTorch rules."""
    l1, l2 = len(s1), len(s2)
    max_len = max(l1, l2)
    pad1 = (1,) * (max_len - l1) + s1
    pad2 = (1,) * (max_len - l2) + s2

    out = []
    for d1, d2 in zip(pad1, pad2):
        if d1 == d2:
            out.append(d1)
        elif d1 == 1:
            out.append(d2)
        elif d2 == 1:
            out.append(d1)
        else:
            raise ValueError(f"Cannot broadcast dimensions: {s1} and {s2}")
    return tuple(out)

class Tensor:
    """
    VeloGrad N-Dimensional Hardware-Aligned Strided Tensor.
    Supports arbitrary dimensions, zero-copy views, in-place mutations,
    batched GEMM, and analytic reverse-mode autograd.
    """
    def __init__(
        self,
        *shape_args: Any,
        data: Optional[List[float]] = None,
        shape: Optional[Union[Tuple[int, ...], List[int]]] = None,
        strides: Optional[Tuple[int, ...]] = None,
        offset: int = 0,
        requires_grad: bool = False,
    ):
        # Support explicit shape kwarg: Tensor(data, shape=[...]) or Tensor(shape=[...], data=...)
        if shape is not None:
            self.shape: Tuple[int, ...] = tuple(int(x) for x in shape)
            if len(shape_args) == 1 and data is None:
                shape_val = shape_args[0]
                if hasattr(shape_val, "tolist"):
                    data = shape_val.tolist()
                elif isinstance(shape_val, (list, tuple)):
                    # Flatten nested list if provided
                    def _flat(lst):
                        res = []
                        for item in lst:
                            if isinstance(item, (list, tuple)):
                                res.extend(_flat(item))
                            else:
                                res.append(float(item))
                        return res
                    data = _flat(shape_val)
        elif len(shape_args) == 1:
            arg = shape_args[0]
            # Handle ndarray or nested list passed as first argument: Tensor([[1, 2], [3, 4]])
            if hasattr(arg, "shape") and hasattr(arg, "tolist"):
                self.shape = tuple(int(x) for x in arg.shape)
                data = arg.tolist()
            elif isinstance(arg, (tuple, list)) and len(arg) > 0 and isinstance(arg[0], (tuple, list)):
                def _get_shape_and_flat(lst):
                    shp = []
                    curr = lst
                    while isinstance(curr, (list, tuple)) and len(curr) > 0:
                        shp.append(len(curr))
                        curr = curr[0]
                    res = []
                    def _fl(items):
                        for it in items:
                            if isinstance(it, (list, tuple)):
                                _fl(it)
                            else:
                                res.append(float(it))
                    _fl(lst)
                    return tuple(shp), res
                self.shape, data = _get_shape_and_flat(arg)
            elif isinstance(arg, (tuple, list)):
                # Could be shape (e.g. [2, 3]) or 1D data if float/int
                # If data is provided, arg is the shape
                if data is not None:
                    self.shape = tuple(int(x) for x in arg)
                else:
                    # Treat as 1D data or shape: if elements are integers, could be shape,
                    # but if floats are present, treat as 1D data
                    has_float = any(isinstance(x, float) for x in arg)
                    if has_float:
                        self.shape = (len(arg),)
                        data = [float(x) for x in arg]
                    else:
                        self.shape = tuple(int(x) for x in arg)
            else:
                self.shape = (int(arg),)
        elif len(shape_args) > 1:
            self.shape = tuple(int(x) for x in shape_args)
        else:
            self.shape = (0,)

        self.ndim: int = len(self.shape)
        self.numel: int = 1
        for d in self.shape:
            self.numel *= d

        self.strides: Tuple[int, ...] = strides if strides is not None else _compute_strides(self.shape)
        self.offset: int = offset
        self.requires_grad: bool = requires_grad

        if data is not None:
            self._storage: List[float] = data
        else:
            self._storage = [0.0] * (self.offset + self.numel)

        self.grad: Optional['Tensor'] = None
        self._grad_fn: Optional[Callable[['Tensor'], None]] = None
        self._prev: List['Tensor'] = []

    @property
    def data(self) -> List[float]:
        """Provides direct access to flattened contiguous data for compatibility."""
        return self._data

    @data.setter
    def data(self, val: List[float]):
        self._data = val

    @property
    def _data(self) -> List[float]:
        """Flattened contiguous data slice representation."""
        if self.is_contiguous() and self.offset == 0 and len(self._storage) == self.numel:
            return self._storage
        return self.contiguous()._storage

    @_data.setter
    def _data(self, val: List[float]):
        if len(val) != self.numel:
            raise ValueError(f"Length mismatch: {len(val)} vs {self.numel}")
        if self.is_contiguous() and self.offset == 0 and len(self._storage) == self.numel:
            self._storage = list(val)
        elif self.is_contiguous():
            self._storage[self.offset:self.offset + self.numel] = list(val)
        else:
            for idx, linear in enumerate(self._iter_indices()):
                self._storage[linear] = val[idx]

    def is_contiguous(self) -> bool:
        """Returns True if memory is contiguous in standard C-order."""
        expected = _compute_strides(self.shape)
        return self.strides == expected

    def _iter_indices(self):
        """Generates linear storage indices according to current shape and strides."""
        if self.numel == 0:
            return
        indices = [0] * self.ndim
        for _ in range(self.numel):
            linear = self.offset
            for dim in range(self.ndim):
                linear += indices[dim] * self.strides[dim]
            yield linear

            # Increment coordinate
            for dim in range(self.ndim - 1, -1, -1):
                indices[dim] += 1
                if indices[dim] < self.shape[dim]:
                    break
                indices[dim] = 0

    def contiguous(self) -> 'Tensor':
        """Returns a contiguous in-memory copy if non-contiguous, or self if already contiguous."""
        if self.is_contiguous() and self.offset == 0:
            return self
        out_data = [self._storage[idx] for idx in self._iter_indices()]
        out = Tensor(self.shape, data=out_data, requires_grad=self.requires_grad)
        return out

    def clone(self) -> 'Tensor':
        """Deep copy of tensor memory."""
        out = Tensor(self.shape, data=list(self.contiguous()._data), requires_grad=self.requires_grad)
        return out

    # =========================================================================
    # Zero-Copy Transformations
    # =========================================================================

    def view(self, *new_shape: Any) -> 'Tensor':
        """
        Zero-copy view reshaping. Does NOT copy underlying buffer.
        """
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            target_shape = tuple(int(x) for x in new_shape[0])
        else:
            target_shape = tuple(int(x) for x in new_shape)

        # Handle -1 infer dimension
        if -1 in target_shape:
            known = 1
            infer_idx = -1
            for i, d in enumerate(target_shape):
                if d == -1:
                    infer_idx = i
                else:
                    known *= d
            inferred = self.numel // known
            target_shape_list = list(target_shape)
            target_shape_list[infer_idx] = inferred
            target_shape = tuple(target_shape_list)

        target_numel = 1
        for d in target_shape:
            target_numel *= d
        if target_numel != self.numel:
            raise ValueError(f"Cannot view shape {self.shape} into {target_shape} (numel {self.numel} vs {target_numel})")

        if not self.is_contiguous():
            return self.contiguous().view(*target_shape)

        new_strides = _compute_strides(target_shape)
        out = Tensor(target_shape, data=self._storage, strides=new_strides, offset=self.offset, requires_grad=self.requires_grad)

        if self.requires_grad:
            out._prev = [self]
            def _backward(grad: 'Tensor'):
                if self.grad is None:
                    self.grad = Tensor(self.shape, requires_grad=False)
                reshaped_grad = grad.view(*self.shape)
                for i in range(len(self.grad._data)):
                    self.grad._data[i] += reshaped_grad._data[i]
            out._grad_fn = _backward

        return out

    def reshape(self, *new_shape: Any) -> 'Tensor':
        """Reshapes tensor, returning a view if contiguous or a copy if non-contiguous."""
        return self.view(*new_shape)

    def slice(self, dim: int, start: int, end: int) -> 'Tensor':
        """Extracts a slice along specified dimension."""
        dim = dim if dim >= 0 else self.ndim + dim
        assert 0 <= dim < self.ndim, f"Dimension {dim} out of bounds for ndim {self.ndim}"
        start = max(0, start)
        end = min(self.shape[dim], end)
        new_dim_len = max(0, end - start)
        new_shape = list(self.shape)
        new_shape[dim] = new_dim_len

        out = Tensor(tuple(new_shape), requires_grad=self.requires_grad)
        outer = 1
        for d in self.shape[:dim]:
            outer *= d
        dim_sz = self.shape[dim]
        inner = 1
        for d in self.shape[dim + 1:]:
            inner *= d

        src_data = self.contiguous()._data
        out_data = out._storage

        for o in range(outer):
            for i in range(new_dim_len):
                src_pos = (o * dim_sz + (start + i)) * inner
                dst_pos = (o * new_dim_len + i) * inner
                out_data[dst_pos : dst_pos + inner] = src_data[src_pos : src_pos + inner]

        return out

    def transpose(self, dim0: int = -2, dim1: int = -1) -> 'Tensor':
        """Zero-copy transpose swapping two dimensions."""
        dim0 = dim0 if dim0 >= 0 else self.ndim + dim0
        dim1 = dim1 if dim1 >= 0 else self.ndim + dim1

        new_shape = list(self.shape)
        new_strides = list(self.strides)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]

        out = Tensor(tuple(new_shape), data=self._storage, strides=tuple(new_strides), offset=self.offset, requires_grad=self.requires_grad)

        if self.requires_grad:
            out._prev = [self]
            def _backward(grad: 'Tensor'):
                if self.grad is None:
                    self.grad = Tensor(self.shape, requires_grad=False)
                transposed_grad = grad.transpose(dim0, dim1).contiguous()
                for i in range(self.numel):
                    self.grad._data[i] += transposed_grad._data[i]
            out._grad_fn = _backward

        return out

    @property
    def T(self) -> 'Tensor':
        """2D transpose property."""
        if self.ndim < 2:
            return self
        return self.transpose(-2, -1)

    def permute(self, *dims: Any) -> 'Tensor':
        """Zero-copy arbitrary dimensional permutation."""
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            perm = tuple(int(x) for x in dims[0])
        else:
            perm = tuple(int(x) for x in dims)

        if len(perm) != self.ndim or sorted(perm) != list(range(self.ndim)):
            raise ValueError(f"Invalid permutation {perm} for {self.ndim}D tensor")

        new_shape = tuple(self.shape[p] for p in perm)
        new_strides = tuple(self.strides[p] for p in perm)

        out = Tensor(new_shape, data=self._storage, strides=new_strides, offset=self.offset, requires_grad=self.requires_grad)

        if self.requires_grad:
            out._prev = [self]
            inv_perm = [0] * len(perm)
            for i, p in enumerate(perm):
                inv_perm[p] = i
            def _backward(grad: 'Tensor'):
                if self.grad is None:
                    self.grad = Tensor(self.shape, requires_grad=False)
                inv_grad = grad.permute(*inv_perm).contiguous()
                for i in range(self.numel):
                    self.grad._data[i] += inv_grad._data[i]
            out._grad_fn = _backward

        return out

    def unsqueeze(self, dim: int) -> 'Tensor':
        """Zero-copy addition of a singleton dimension."""
        if dim < 0:
            dim = self.ndim + 1 + dim
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)

        new_strides = list(self.strides)
        stride_val = self.strides[dim] if dim < self.ndim else 1
        new_strides.insert(dim, stride_val)

        return Tensor(tuple(new_shape), data=self._storage, strides=tuple(new_strides), offset=self.offset, requires_grad=self.requires_grad)

    def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
        """Zero-copy removal of singleton dimensions."""
        if dim is not None:
            if dim < 0:
                dim = self.ndim + dim
            if self.shape[dim] != 1:
                return self
            new_shape = [d for i, d in enumerate(self.shape) if i != dim]
            new_strides = [s for i, s in enumerate(self.strides) if i != dim]
        else:
            new_shape = [d for d in self.shape if d != 1]
            new_strides = [s for i, s in enumerate(self.strides) if self.shape[i] != 1]

        if not new_shape:
            new_shape = [1]
            new_strides = [1]

        return Tensor(tuple(new_shape), data=self._storage, strides=tuple(new_strides), offset=self.offset, requires_grad=self.requires_grad)

    # =========================================================================
    # Arithmetic & Autograd Operations
    # =========================================================================

    def zero_(self) -> 'Tensor':
        """Zeroes all elements in-place."""
        for idx in self._iter_indices():
            self._storage[idx] = 0.0
        return self

    def fill_(self, val: float) -> 'Tensor':
        """Fills all elements with a constant in-place."""
        for idx in self._iter_indices():
            self._storage[idx] = float(val)
        return self

    def add(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """N-Dimensional broadcasting addition."""
        if not isinstance(other, Tensor):
            out = Tensor(self.shape, requires_grad=self.requires_grad)
            val = float(other)
            for i, idx in enumerate(self._iter_indices()):
                out._storage[i] = self._storage[idx] + val
            if self.requires_grad:
                out._prev = [self]
                def _backward(grad: 'Tensor'):
                    if self.grad is None:
                        self.grad = Tensor(self.shape, requires_grad=False)
                    for i in range(self.numel):
                        self.grad._data[i] += grad._data[i]
                out._grad_fn = _backward
            return out

        out_shape = _broadcast_shapes(self.shape, other.shape)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_shape, requires_grad=req_grad)

        s_flat = self.contiguous()._data
        o_flat = other.contiguous()._data

        if self.shape == other.shape:
            for i in range(len(s_flat)):
                out._storage[i] = s_flat[i] + o_flat[i]
        else:
            for i in range(out.numel):
                s_val = s_flat[i % len(s_flat)]
                o_val = o_flat[i % len(o_flat)]
                out._storage[i] = s_val + o_val

        if req_grad:
            out._prev = [self, other]
            def _backward(grad: 'Tensor'):
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = Tensor(self.shape, requires_grad=False)
                    if self.shape == out_shape:
                        for i in range(self.numel):
                            self.grad._data[i] += grad._data[i]
                    else:
                        for i in range(grad.numel):
                            self.grad._data[i % self.numel] += grad._data[i]

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = Tensor(other.shape, requires_grad=False)
                    if other.shape == out_shape:
                        for i in range(other.numel):
                            other.grad._data[i] += grad._data[i]
                    else:
                        for i in range(grad.numel):
                            other.grad._data[i % other.numel] += grad._data[i]
            out._grad_fn = _backward

        return out

    def sub(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            return self.add(-float(other))
        neg_other = other.mul(-1.0)
        return self.add(neg_other)

    def mul(self, other: Union['Tensor', float, int]) -> 'Tensor':
        """Elementwise multiplication with broadcasting."""
        if not isinstance(other, Tensor):
            val = float(other)
            out = Tensor(self.shape, requires_grad=self.requires_grad)
            for i, idx in enumerate(self._iter_indices()):
                out._storage[i] = self._storage[idx] * val
            if self.requires_grad:
                out._prev = [self]
                def _backward(grad: 'Tensor'):
                    if self.grad is None:
                        self.grad = Tensor(self.shape, requires_grad=False)
                    for i in range(self.numel):
                        self.grad._data[i] += grad._data[i] * val
                out._grad_fn = _backward
            return out

        out_shape = _broadcast_shapes(self.shape, other.shape)
        req_grad = self.requires_grad or other.requires_grad
        out = Tensor(out_shape, requires_grad=req_grad)

        s_flat = self.contiguous()._data
        o_flat = other.contiguous()._data

        if self.shape == other.shape:
            for i in range(len(s_flat)):
                out._storage[i] = s_flat[i] * o_flat[i]
        else:
            for i in range(out.numel):
                out._storage[i] = s_flat[i % len(s_flat)] * o_flat[i % len(o_flat)]

        if req_grad:
            out._prev = [self, other]
            def _backward(grad: 'Tensor'):
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = Tensor(self.shape, requires_grad=False)
                    for i in range(grad.numel):
                        self.grad._data[i % self.numel] += grad._data[i] * o_flat[i % len(o_flat)]
                if other.requires_grad:
                    if other.grad is None:
                        other.grad = Tensor(other.shape, requires_grad=False)
                    for i in range(grad.numel):
                        other.grad._data[i % other.numel] += grad._data[i] * s_flat[i % len(s_flat)]
            out._grad_fn = _backward

        return out

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Batched and 2D Matrix Multiplication with full autograd.
        Supports (..., M, K) @ (..., K, N) -> (..., M, N).
        """
        assert self.ndim >= 2 and other.ndim >= 2, f"Matrix multiplication requires at least 2D tensors, got {self.shape} and {other.shape}"
        m, k1 = self.shape[-2], self.shape[-1]
        k2, n = other.shape[-2], other.shape[-1]
        assert k1 == k2, f"Shape mismatch in matmul: ({m}, {k1}) vs ({k2}, {n})"

        req_grad = self.requires_grad or other.requires_grad

        if _NUMPY_AVAILABLE:
            s_arr = np.asarray(self.contiguous()._data, dtype=np.float32).reshape(self.shape)
            o_arr = np.asarray(other.contiguous()._data, dtype=np.float32).reshape(other.shape)
            res_arr = np.matmul(s_arr, o_arr)
            out = Tensor(res_arr.shape, data=res_arr.flatten().tolist(), requires_grad=req_grad)
            if req_grad:
                out._prev = [self, other]
                def _backward(grad: 'Tensor'):
                    g_arr = np.asarray(grad.contiguous()._data, dtype=np.float32).reshape(grad.shape)
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = Tensor(self.shape, requires_grad=False)
                        o_T = np.swapaxes(o_arr, -1, -2)
                        d_self = np.matmul(g_arr, o_T)
                        if d_self.shape != self.shape:
                            axes = tuple(range(d_self.ndim - len(self.shape)))
                            for ax in range(len(self.shape)):
                                if self.shape[ax] == 1 and d_self.shape[ax + len(axes)] > 1:
                                    axes += (ax + len(axes),)
                            d_self = np.sum(d_self, axis=axes, keepdims=True).reshape(self.shape)
                        cur_g = np.asarray(self.grad._data, dtype=np.float32).reshape(self.shape) + d_self
                        self.grad._data = cur_g.flatten().tolist()
                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = Tensor(other.shape, requires_grad=False)
                        s_T = np.swapaxes(s_arr, -1, -2)
                        d_other = np.matmul(s_T, g_arr)
                        if d_other.shape != other.shape:
                            axes = tuple(range(d_other.ndim - len(other.shape)))
                            for ax in range(len(other.shape)):
                                if other.shape[ax] == 1 and d_other.shape[ax + len(axes)] > 1:
                                    axes += (ax + len(axes),)
                            d_other = np.sum(d_other, axis=axes, keepdims=True).reshape(other.shape)
                        cur_g = np.asarray(other.grad._data, dtype=np.float32).reshape(other.shape) + d_other
                        other.grad._data = cur_g.flatten().tolist()
                out._grad_fn = _backward
            return out

        b_self = self.shape[:-2]
        b_other = other.shape[:-2]
        batch_shape = _broadcast_shapes(b_self, b_other) if (b_self or b_other) else ()
        out_shape = batch_shape + (m, n)

        batch_count = 1
        for b in batch_shape:
            batch_count *= b

        out = Tensor(out_shape, requires_grad=req_grad)

        s_mat = self.contiguous()._data
        o_mat = other.contiguous()._data

        stride_s = m * k1
        stride_o = k1 * n
        stride_out = m * n

        for b in range(batch_count):
            s_offset = (b % (len(s_mat) // stride_s)) * stride_s if stride_s > 0 else 0
            o_offset = (b % (len(o_mat) // stride_o)) * stride_o if stride_o > 0 else 0
            out_offset = b * stride_out

            for i in range(m):
                for j in range(n):
                    acc = 0.0
                    for p in range(k1):
                        acc += s_mat[s_offset + i * k1 + p] * o_mat[o_offset + p * n + j]
                    out._storage[out_offset + i * n + j] = acc

        if req_grad:
            out._prev = [self, other]
            def _backward(grad: 'Tensor'):
                g_mat = grad.contiguous()._data

                if self.requires_grad:
                    if self.grad is None:
                        self.grad = Tensor(self.shape, requires_grad=False)
                    for b in range(batch_count):
                        g_off = b * stride_out
                        o_off = (b % (len(o_mat) // stride_o)) * stride_o
                        s_off = (b % (len(s_mat) // stride_s)) * stride_s

                        for i in range(m):
                            for p in range(k1):
                                acc = 0.0
                                for j in range(n):
                                    acc += g_mat[g_off + i * n + j] * o_mat[o_off + p * n + j]
                                self.grad._data[s_off + i * k1 + p] += acc

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = Tensor(other.shape, requires_grad=False)
                    for b in range(batch_count):
                        g_off = b * stride_out
                        s_off = (b % (len(s_mat) // stride_s)) * stride_s
                        o_off = (b % (len(o_mat) // stride_o)) * stride_o

                        for p in range(k1):
                            for j in range(n):
                                acc = 0.0
                                for i in range(m):
                                    acc += s_mat[s_off + i * k1 + p] * g_mat[g_off + i * n + j]
                                other.grad._data[o_off + p * n + j] += acc

            out._grad_fn = _backward

        return out

    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Reduces tensor by summation."""
        if dim is None:
            total = sum(self.contiguous()._data)
            out = Tensor((1,), data=[total], requires_grad=self.requires_grad)
            if self.requires_grad:
                out._prev = [self]
                def _backward(grad: 'Tensor'):
                    if self.grad is None:
                        self.grad = Tensor(self.shape, requires_grad=False)
                    g_val = grad._data[0]
                    for i in range(self.numel):
                        self.grad._data[i] += g_val
                out._grad_fn = _backward
            return out

        dim = dim if dim >= 0 else self.ndim + dim
        new_shape = list(self.shape)
        if keepdim:
            new_shape[dim] = 1
        else:
            new_shape.pop(dim)

        out = Tensor(tuple(new_shape), requires_grad=self.requires_grad)
        # Sum over reduced dimension
        s_data = self.contiguous()._data
        stride_reduce = self.strides[dim]
        dim_size = self.shape[dim]

        for i in range(out.numel):
            # Accumulate across reduction axis
            out._data[i] = sum(s_data[i * dim_size + k] for k in range(dim_size))

        if self.requires_grad:
            out._prev = [self]
            def _backward(grad: 'Tensor'):
                if self.grad is None:
                    self.grad = Tensor(self.shape, requires_grad=False)
                g_flat = grad.contiguous()._data
                for i in range(out.numel):
                    g_val = g_flat[i]
                    for k in range(dim_size):
                        self.grad._data[i * dim_size + k] += g_val
            out._grad_fn = _backward

        return out

    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> 'Tensor':
        """Computes mean over given dimension or all elements."""
        s = self.sum(dim=dim, keepdim=keepdim)
        divisor = float(self.shape[dim] if dim is not None else self.numel)
        return s.mul(1.0 / max(1.0, divisor))

    def backward(self, gradient: Optional['Tensor'] = None):
        """
        Reverse-Mode Automatic Differentiation Graph Traversal.
        Back-propagates upstream gradients through the dynamically recorded DAG.
        """
        if self.grad is None:
            if gradient is not None:
                self.grad = gradient
            else:
                self.grad = Tensor(self.shape, requires_grad=False)
                self.grad.fill_(1.0)

        # Topological sort
        topo: List['Tensor'] = []
        visited = set()

        def build_topo(v: 'Tensor'):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Traverse in reverse topological order
        for node in reversed(topo):
            if node._grad_fn is not None and node.grad is not None:
                node._grad_fn(node.grad)

    # Magic operator overloads
    def __len__(self) -> int: return self.shape[0] if self.ndim > 0 else 0
    def __add__(self, other): return self.add(other)
    def __radd__(self, other): return self.add(other)
    def __sub__(self, other): return self.sub(other)
    def __mul__(self, other): return self.mul(other)
    def __rmul__(self, other): return self.mul(other)
    def __matmul__(self, other): return self.matmul(other)
    def __neg__(self): return self.mul(-1.0)

    def __repr__(self) -> str:
        data_preview = self.contiguous()._data
        if len(data_preview) > 8:
            preview = f"[{data_preview[0]:.4f}, {data_preview[1]:.4f}, ..., {data_preview[-1]:.4f}]"
        else:
            preview = f"[{', '.join(f'{x:.4f}' for x in data_preview)}]"
        return f"velograd.Tensor(shape={self.shape}, data={preview}, requires_grad={self.requires_grad})"

# Factory functions
def zeros(*shape: Any, requires_grad: bool = False) -> Tensor:
    t = Tensor(*shape, requires_grad=requires_grad)
    t.zero_()
    return t

def ones(*shape: Any, requires_grad: bool = False) -> Tensor:
    t = Tensor(*shape, requires_grad=requires_grad)
    t.fill_(1.0)
    return t

def randn(*shape: Any, requires_grad: bool = False) -> Tensor:
    t = Tensor(*shape, requires_grad=requires_grad)
    if _NUMPY_AVAILABLE:
        t._data = np.random.standard_normal(t.numel).astype(np.float32).tolist()
    else:
        t._data = [random.gauss(0.0, 1.0) for _ in range(t.numel)]
    return t

def tensor(data: Any, requires_grad: bool = False) -> Tensor:
    def _get_shape_and_flat(lst):
        if not isinstance(lst, list):
            return (), [float(lst)]
        sub_shape, flat = _get_shape_and_flat(lst[0])
        all_flat = []
        for item in lst:
            _, item_flat = _get_shape_and_flat(item)
            all_flat.extend(item_flat)
        return (len(lst),) + sub_shape, all_flat

    shape, flat = _get_shape_and_flat(data)
    t = Tensor(shape, data=flat, requires_grad=requires_grad)
    return t
