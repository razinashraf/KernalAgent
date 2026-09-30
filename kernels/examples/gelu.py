"""Hand-written Triton kernel for problems/gelu.py — a known-good example.

HOW A TRITON KERNEL IS ORGANISED
The GPU runs many copies of `_gelu_kernel` at once. Each copy is a "program" (Triton's
word; in CUDA terms, roughly one thread block). Program number `pid` handles the
elements [pid * BLOCK, pid * BLOCK + BLOCK). Inside a program, Triton operates on whole
blocks of values at once (like NumPy arrays), and it decides how to spread them over the
program's threads.

We treat the (M, N) input as a flat array of M*N numbers: GELU doesn't care about rows.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    # The last program usually sticks out past the end of the array (n_elements is rarely
    # an exact multiple of BLOCK). The mask turns those out-of-range loads/stores off.
    # Forgetting it is the #1 Triton bug: it reads garbage or writes over other memory.
    mask = offsets < n_elements

    # COALESCING: neighbouring lanes load neighbouring addresses, so the hardware can
    # combine a warp's 32 loads into a few wide memory transactions. That is what makes
    # this simple kernel run near peak bandwidth.
    x = tl.load(x_ptr + offsets, mask=mask)

    # Do the math in float32 even for fp16 inputs: fp16 has only ~3 significant digits.
    # This costs nothing noticeable, because the kernel spends its time waiting for
    # memory, not doing arithmetic ("memory-bound").
    x = x.to(tl.float32)

    # tanh-GELU:  0.5 * x * (1 + tanh(u))  with  u = sqrt(2/pi) * (x + 0.044715 x^3)
    # Identity:   0.5 * (1 + tanh(u)) == sigmoid(2u),  so  gelu(x) = x * sigmoid(2u).
    # tl.sigmoid is built in and numerically well-behaved for large |u|.
    SQRT_2_OVER_PI = 0.7978845608028654
    u = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    y = x * tl.sigmoid(2.0 * u)

    # Cast back to the output's element type (fp16 or fp32) when storing.
    tl.store(out_ptr + offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def kernel(x):
    out = torch.empty_like(x)
    n = x.numel()
    # BLOCK = 1024 elements per program. Bigger blocks mean fewer programs (less
    # scheduling overhead) and more bytes in flight per program; beyond a few thousand,
    # each thread holds too many values in registers. 1024 with 4 warps (128 threads)
    # gives each thread 8 elements: 16 bytes for fp16, one wide vector load.
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)   # enough programs to cover all n elements
    _gelu_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4)
    return out
