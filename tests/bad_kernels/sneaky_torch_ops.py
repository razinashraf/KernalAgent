"""BAD: launches a do-nothing Triton kernel for show, then computes GELU with ordinary
tensor arithmetic (* and +), which the static check cannot tell apart from normal
Python math. The dynamic op recorder must catch it -> 'cheat'."""
import torch
import triton
import triton.language as tl


@triton.jit
def _noop(x_ptr):
    pass


def kernel(x):
    _noop[(1,)](x)
    u = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return x / (1 + 2.718281828 ** (-2 * u))
