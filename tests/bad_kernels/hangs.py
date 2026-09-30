"""BAD: never returns. The sandbox must time out and kill it."""
import torch
import triton
import triton.language as tl


@triton.jit
def _k(x_ptr):
    pass


def kernel(x):
    while True:
        pass
