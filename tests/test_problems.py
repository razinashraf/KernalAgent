"""The problem definitions themselves must be sane."""
import random

import pytest
import torch

from conftest import requires_gpu
from harness.correctness import compare, expected_output
from problems import list_problems, load_problem

REQUIRED = ["NAME", "CATEGORY", "DESCRIPTION", "DTYPES", "TOLERANCES", "SHAPES",
            "EDGE_SHAPES", "random_shape", "make_inputs", "reference"]
PROBLEMS = list_problems()


def is_pow2(v):
    return v > 0 and (v & (v - 1)) == 0


def test_twelve_problems():
    assert len(PROBLEMS) == 12, PROBLEMS


@pytest.mark.parametrize("name", PROBLEMS)
def test_problem_structure(name):
    p = load_problem(name)
    for attr in REQUIRED:
        assert hasattr(p, attr), f"{name} is missing {attr}"
    assert p.NAME == name
    assert set(p.SHAPES) == {"small", "medium", "large"}
    for dtype in p.DTYPES:
        assert dtype in p.TOLERANCES
    # At least one edge shape must have a dimension that is not a power of 2.
    assert any(not is_pow2(v) for s in p.EDGE_SHAPES for v in s.values())
    # random_shape returns the same dimension names as the named shapes.
    assert set(p.random_shape(random.Random(0))) == set(p.SHAPES["small"])


@requires_gpu
@pytest.mark.parametrize("name", PROBLEMS)
def test_inputs_deterministic_and_seed_dependent(name):
    p = load_problem(name)
    shape, dtype = p.SHAPES["small"], p.DTYPES[0]
    a = p.make_inputs(shape, dtype, 1)
    b = p.make_inputs(shape, dtype, 1)
    c = p.make_inputs(shape, dtype, 2)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert any(not torch.equal(x, y) for x, y in zip(a, c))
    assert all(t.is_cuda and t.is_contiguous() for t in a)


@requires_gpu
@pytest.mark.parametrize("name", PROBLEMS)
def test_tolerances_accept_pytorch_itself(name):
    """PyTorch's own result in each dtype must pass our tolerance vs the fp32 reference.
    If it doesn't, the tolerance is too strict and no kernel could reasonably pass."""
    p = load_problem(name)
    shapes = list(p.SHAPES.values()) + list(p.EDGE_SHAPES)
    for dtype in p.DTYPES:
        for shape in shapes:
            inputs = p.make_inputs(shape, dtype, 0)
            with torch.no_grad():
                got = p.reference(*inputs)
            result = compare(got, expected_output(p, inputs), **p.TOLERANCES[dtype])
            assert result["ok"], (name, shape, dtype, result)
            del inputs, got
        torch.cuda.empty_cache()
