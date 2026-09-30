"""Static checks: these run without a GPU."""
import os

from conftest import BAD, EXAMPLES
from harness.anticheat import static_check


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


GOOD_TEMPLATE = """
import math
import torch
import triton
import triton.language as tl

@triton.jit
def _k(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n).to(tl.float32)
    tl.store(out_ptr + offs, tl.exp(x).to(out_ptr.dtype.element_ty), mask=offs < n)

def kernel(x):
    out = torch.empty_like(x)
    scale = 1.0 / math.sqrt(x.shape[-1])
    _k[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    return out
"""


def with_host_line(line):
    """GOOD_TEMPLATE with one extra line inserted into the host-side kernel() function."""
    return GOOD_TEMPLATE.replace("    return out", f"    {line}\n    return out")


def test_examples_pass_static_check():
    for name in os.listdir(EXAMPLES):
        if name.endswith(".py"):
            assert static_check(read(os.path.join(EXAMPLES, name))) == [], name


def test_template_passes():
    # Also shows that tl.exp / .to(...) inside the @triton.jit function are not flagged,
    # and math.sqrt on the host is fine.
    assert static_check(GOOD_TEMPLATE) == []


def test_rejects_torch_functional():
    issues = static_check(read(os.path.join(BAD, "calls_torch.py")))
    assert any("torch.nn" in i for i in issues), issues


def test_rejects_torch_compute_function():
    assert static_check(with_host_line("out = torch.exp(x)"))


def test_rejects_tensor_compute_method():
    assert static_check(with_host_line("out = x.softmax(-1)"))


def test_rejects_matmul_operator():
    assert static_check(with_host_line("out = x @ x"))


def test_rejects_transpose_attribute():
    assert static_check(with_host_line("out = x.T.contiguous()"))


def test_rejects_dtype_cast_on_host():
    assert static_check(with_host_line("out = x.to(torch.float32)"))


def test_rejects_bad_imports():
    assert static_check("import os\n" + GOOD_TEMPLATE)
    assert static_check("import numpy as np\n" + GOOD_TEMPLATE)
    assert static_check("from problems import gelu\n" + GOOD_TEMPLATE)
    assert static_check("from torch.nn import functional\n" + GOOD_TEMPLATE)


def test_rejects_exec_and_getattr():
    assert static_check(with_host_line("exec('pass')"))
    assert static_check(with_host_line("f = getattr(torch, 'exp')"))


def test_rejects_global():
    assert static_check(with_host_line("global CACHE"))


def test_requires_kernel_function_and_jit():
    renamed = GOOD_TEMPLATE.replace("def kernel(", "def run(")
    assert any("kernel" in i for i in static_check(renamed))
    no_jit = GOOD_TEMPLATE.replace("@triton.jit\n", "")
    assert any("triton.jit" in i for i in static_check(no_jit))


def test_syntax_error_reported():
    issues = static_check("def kernel(:\n")
    assert issues and "SyntaxError" in issues[0]
