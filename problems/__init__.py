"""The problem set: one file per operation we want a fast kernel for.

Every problem file defines the same things (see gelu.py for a fully commented example):

    NAME          short id, same as the file name
    CATEGORY      "elementwise" | "reduction" | "fused" | "matmul" | "attention" | "memory"
    DESCRIPTION   plain-English spec, shown to the LLM agent
    DTYPES        dtypes the kernel must support; DTYPES[0] is the one we benchmark
    TOLERANCES    {dtype: {"atol": ..., "rtol": ...}} for comparing against the reference
    SHAPES        {"small": {...}, "medium": {...}, "large": {...}}  (benchmarked)
    EDGE_SHAPES   awkward shapes (non-power-of-2, size 1, ...)       (correctness only)
    random_shape(rng)                 -> a fresh shape the kernel has never seen
    make_inputs(shape, dtype, seed)   -> list of CUDA tensors
    reference(*inputs)                -> the PyTorch "ground truth" output
    flops(shape)   (optional)         -> floating point operations, for compute-bound ops

A shape is a dict of named dimensions, e.g. {"M": 4096, "N": 1024}.
The candidate kernel always takes the same arguments as `reference` and returns a new
contiguous tensor with the same shape and dtype as the reference output.
"""

import importlib
import pkgutil


def list_problems():
    """Names of all problems, i.e. every module in this folder not starting with '_'."""
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if not m.name.startswith("_"))


def load_problem(name):
    return importlib.import_module(f"problems.{name}")
