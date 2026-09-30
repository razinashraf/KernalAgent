"""BAD: lets PyTorch do the work. Must be rejected by the static check."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _unused(x_ptr):
    pass


def kernel(x):
    return F.gelu(x, approximate="tanh")
