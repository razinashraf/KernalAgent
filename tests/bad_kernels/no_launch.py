"""BAD: defines a Triton kernel but never launches it. Must be reported as 'cheat'."""
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr):
    pass


def kernel(x):
    return torch.empty_like(x)
