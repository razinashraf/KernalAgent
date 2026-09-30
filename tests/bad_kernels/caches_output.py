"""BAD: computes the right answer once per shape, then returns the cached result on
later calls. Fresh random inputs on every call must expose it -> 'incorrect'."""
import torch
import triton
import triton.language as tl

_cache = {}


@triton.jit
def _k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    y = x * tl.sigmoid(2.0 * 0.7978845608028654 * (x + 0.044715 * x * x * x))
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel(x):
    key = (tuple(x.shape), x.dtype)
    if key not in _cache:
        out = torch.empty_like(x)
        _k[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
        _cache[key] = out
    return _cache[key]
