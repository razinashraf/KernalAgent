"""Anti-cheating checks.

An LLM that is rewarded for "fast and correct" will sometimes find shortcuts that are
neither: calling PyTorch to do the real work, caching an output, and so on. We defend in
two layers:

1. STATIC checks (static_check): read the source code without running it, and reject
   obvious shortcuts with a clear message the agent can learn from. Cheap, but it can be
   fooled (e.g. `x * y` on tensors looks like ordinary arithmetic).

2. DYNAMIC checks (used by correctness.py while the kernel actually runs):
   - OpRecorder records every PyTorch operation executed during the kernel call. Only
     memory allocation and "view" operations (which just reinterpret existing memory) are
     allowed. Any real computation (add, exp, matmul, copy, ...) means PyTorch did the work.
   - count_triton_launches proves at least one Triton kernel actually ran.
   - correctness.py also feeds fresh random inputs on every call (so cached or hardcoded
     outputs are wrong), tests shapes the kernel has never seen, and checks that the output
     does not share memory with an input.
"""

import ast

from torch.utils._python_dispatch import TorchDispatchMode

# ---------------------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------------------

# `import x` / `from x import y` is only allowed for these modules (or submodules of triton).
ALLOWED_IMPORTS = {"torch", "math", "typing"}
ALLOWED_IMPORT_PREFIXES = ("triton",)

# Outside Triton kernels, `torch.<name>` may only use these names: tensor allocation,
# dtypes, and device helpers. Anything else (torch.exp, torch.nn.functional, ...) would be
# PyTorch doing the computation.
TORCH_ALLOWED = {
    "empty", "empty_like", "empty_strided", "zeros", "zeros_like", "full",
    "float16", "float32", "float64", "bfloat16", "half", "float",
    "int8", "int16", "int32", "int64", "uint8", "bool",
    "cuda", "device", "dtype", "Tensor", "Size", "finfo", "iinfo", "is_tensor",
    "contiguous_format", "no_grad", "inference_mode",
}

# Method names that do computation if called on a tensor in host (non-Triton) code.
# Inside @triton.jit functions these names are fine (e.g. tl.sum, x.to(tl.float32)).
TENSOR_COMPUTE_METHODS = {
    "sum", "mean", "prod", "softmax", "log_softmax", "exp", "exp2", "log", "log2", "sqrt",
    "rsqrt", "tanh", "sigmoid", "relu", "gelu", "silu", "erf", "abs", "neg", "pow", "square",
    "add", "add_", "sub", "sub_", "mul", "mul_", "div", "div_", "addmm", "addmv", "baddbmm",
    "matmul", "mm", "bmm", "mv", "dot", "einsum", "max", "min", "amax", "amin", "argmax",
    "argmin", "logsumexp", "norm", "var", "std", "var_mean", "cumsum", "cumprod", "clamp",
    "clamp_", "where", "masked_fill", "layer_norm", "rms_norm", "copy_", "clone", "to",
    "float", "half", "double", "bfloat16", "type", "t", "transpose", "permute", "flip",
    "roll", "cat", "stack", "index_select", "gather", "scatter", "numpy", "tolist", "item",
    "cpu", "scaled_dot_product_attention", "linear",
}
# Attribute *reads* (not calls) that compute a transpose.
TENSOR_COMPUTE_ATTRIBUTES = {"T", "mT", "H", "mH"}

# Modules whose functions are fine to call in host code (math.sqrt, triton.cdiv, tl.constexpr, ...).
SAFE_MODULE_NAMES = {"math", "triton", "tl"}

BANNED_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input", "memoryview",
}


def _is_triton_kernel(func_def):
    """True if a function is decorated with @triton.jit (possibly under @triton.autotune)."""
    return any("jit" in ast.unparse(d) for d in func_def.decorator_list)


def _root_name(node):
    """For `a.b.c`, return 'a'. For anything more complex (calls, subscripts) return None."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _torch_attr(node):
    """For `torch.nn.functional.gelu` return 'nn'; None if the expression isn't torch.<x>..."""
    chain = []
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == "torch" and chain:
        return chain[-1]
    return None


class _HostCodeChecker(ast.NodeVisitor):
    """Walks the AST, collecting problems. Skips the bodies of Triton kernels."""

    def __init__(self):
        self.problems = []

    def _flag(self, node, msg):
        self.problems.append(f"line {getattr(node, 'lineno', '?')}: {msg}")

    def visit_FunctionDef(self, node):
        if _is_triton_kernel(node):
            return  # device code: tl.exp, tl.dot etc. are exactly what we want there
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self._check_import(node, alias.name)

    def visit_ImportFrom(self, node):
        self._check_import(node, node.module or "")
        if node.module == "torch":
            for alias in node.names:
                if alias.name not in TORCH_ALLOWED:
                    self._flag(node, f"`from torch import {alias.name}` is not allowed")

    def _check_import(self, node, module):
        if module in ALLOWED_IMPORTS or module.startswith(ALLOWED_IMPORT_PREFIXES):
            return
        self._flag(node, f"import of `{module}` is not allowed (allowed: torch, triton, math, typing)")

    def visit_Global(self, node):
        self._flag(node, "`global` is not allowed (module-level state could cache results)")

    def visit_Nonlocal(self, node):
        self._flag(node, "`nonlocal` is not allowed")

    def visit_Name(self, node):
        if node.id in BANNED_NAMES:
            self._flag(node, f"`{node.id}` is not allowed")

    def visit_Attribute(self, node):
        attr = _torch_attr(node)
        if attr is not None and attr not in TORCH_ALLOWED:
            self._flag(node, f"`torch.{attr}` is not allowed outside the Triton kernel; "
                             "do the computation in Triton")
            return  # don't report torch.nn again for the inner part of torch.nn.functional
        if node.attr in TENSOR_COMPUTE_ATTRIBUTES and _root_name(node) not in SAFE_MODULE_NAMES:
            self._flag(node, f"`.{node.attr}` (transpose) is not allowed outside the Triton kernel")
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr in TENSOR_COMPUTE_METHODS
                and _root_name(f) not in SAFE_MODULE_NAMES and _torch_attr(f) is None):
            self._flag(node, f"calling `.{f.attr}()` outside the Triton kernel is not allowed; "
                             "do the computation in Triton")
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.MatMult):
            self._flag(node, "the `@` matmul operator is not allowed outside the Triton kernel")
        self.generic_visit(node)


def static_check(source):
    """Return a list of human-readable problems with the candidate source (empty = OK)."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError on line {e.lineno}: {e.msg}"]

    problems = []
    top_level_funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    if "kernel" not in top_level_funcs:
        problems.append("the file must define a top-level function `kernel(...)` "
                        "taking the same arguments as the reference")
    all_funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not any(_is_triton_kernel(f) for f in all_funcs):
        problems.append("the file must contain at least one @triton.jit kernel")

    checker = _HostCodeChecker()
    checker.visit(tree)
    return problems + checker.problems


# ---------------------------------------------------------------------------------------
# Dynamic checks
# ---------------------------------------------------------------------------------------

# PyTorch ops that are allowed to run inside the candidate's `kernel()` call. They either
# allocate memory, fill fresh memory with a constant, create a view (a different way of
# looking at the SAME memory, no data movement), or just query metadata.
ALLOWED_TORCH_OPS = {
    # allocation
    "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided",
    "zeros", "zeros_like", "new_zeros", "zero_", "fill_", "full", "full_like", "new_full",
    # views (no data is read or written)
    "view", "_unsafe_view", "_reshape_alias", "reshape", "as_strided", "alias", "detach",
    "lift_fresh", "unsqueeze", "squeeze", "expand", "t", "transpose", "permute", "select",
    "slice", "split", "unbind", "flatten", "unflatten",
    # metadata queries
    "is_contiguous", "sym_size", "sym_stride", "sym_numel", "sym_storage_offset",
    "stride", "size", "dim", "numel", "is_same_size", "record_stream",
}


class OpRecorder(TorchDispatchMode):
    """Context manager that records every PyTorch (aten) op executed inside it.

    TorchDispatchMode is a PyTorch hook: while it is active, every tensor operation goes
    through __torch_dispatch__ below before running. Triton kernel launches do NOT go
    through here (Triton talks to the GPU driver directly), so any compute op we see here
    was done by PyTorch, not by the candidate's Triton kernel.
    """

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))

    def forbidden_ops(self):
        return sorted({op for op in self.ops if op not in ALLOWED_TORCH_OPS})


_launch_count = 0


def install_launch_counter():
    """Wrap Triton's JITFunction.run so we can count real kernel launches.

    `my_kernel[grid](...)` ends up calling JITFunction.run(..., warmup=False).
    """
    from triton.runtime.jit import JITFunction

    if getattr(JITFunction.run, "_is_counting_wrapper", False):
        return
    original_run = JITFunction.run

    def counting_run(self, *args, **kwargs):
        global _launch_count
        if not kwargs.get("warmup", False):
            _launch_count += 1
        return original_run(self, *args, **kwargs)

    counting_run._is_counting_wrapper = True
    JITFunction.run = counting_run


def triton_launch_count():
    return _launch_count


def shares_memory(a, b):
    """True if tensors a and b are backed by the same memory buffer."""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
