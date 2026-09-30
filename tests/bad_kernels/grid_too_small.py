"""BAD: sizes the grid with floor division (n // BLOCK) instead of ceiling division
(triton.cdiv), so the last partial block is never computed when n is not a multiple of
BLOCK. Passes the power-of-2 shapes, must fail on the edge shapes -> 'incorrect'."""
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = x * tl.sigmoid(2.0 * 0.7978845608028654 * (x + 0.044715 * x * x * x))
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel(x):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (max(1, n // 1024),)   # BUG: should be triton.cdiv(n, 1024)
    _k[grid](x, out, n, BLOCK=1024)
    return out
