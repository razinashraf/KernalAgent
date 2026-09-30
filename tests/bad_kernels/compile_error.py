"""BAD: Triton code that doesn't compile (tl.arange needs a power-of-2 length).
Must be reported as 'compile_error' with Triton's message."""
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, out_ptr, n):
    offs = tl.arange(0, 1000)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def kernel(x):
    out = torch.empty_like(x)
    _k[(1,)](x, out, x.numel())
    return out
