"""BAD: computes ReLU instead of GELU. Must be reported as 'incorrect'."""
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def kernel(x):
    out = torch.empty_like(x)
    _k[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out
