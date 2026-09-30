import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(REPO_ROOT, "kernels", "examples")
BAD = os.path.join(REPO_ROOT, "tests", "bad_kernels")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Tests that need a real NVIDIA GPU are marked with @requires_gpu and skipped elsewhere.
requires_gpu = pytest.mark.skipif(not _has_cuda(), reason="needs an NVIDIA GPU with CUDA")
