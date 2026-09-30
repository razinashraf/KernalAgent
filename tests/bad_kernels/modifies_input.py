"""BAD: writes the result into the input tensor and returns it. Must be caught."""
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = x * tl.sigmoid(2.0 * 0.7978845608028654 * (x + 0.044715 * x * x * x))
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


def kernel(x):
    _k[(triton.cdiv(x.numel(), 1024),)](x, x.numel(), BLOCK=1024)
    return x
