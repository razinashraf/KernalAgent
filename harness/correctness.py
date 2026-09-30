"""Correctness: does a candidate kernel compute the same thing as the PyTorch reference?

This code runs INSIDE the sandboxed worker process (see sandbox.py), never in the main
program. A buggy GPU kernel can corrupt the CUDA context ("illegal memory access") and
every later GPU call in that process fails, so each candidate gets a fresh process.

What we test, for every dtype in problem.DTYPES:
  - the named shapes (small / medium / large),
  - the edge shapes (size 1, non-powers-of-2, very long rows),
  - one random shape chosen fresh for this run, so a kernel can't special-case shapes.
For each shape the kernel is called 3 times with 3 different random seeds:
  call 1 = warmup (Triton compiles / autotunes here; not checked for cheating),
  calls 2 and 3 = checked. New random values each time means a kernel that caches or
  hardcodes its output will be wrong on at least one of them.
"""

import random
import traceback

import torch

from harness import anticheat


def to_fp32(tensors):
    return [t.float() if t.is_floating_point() else t for t in tensors]


def expected_output(problem, inputs):
    """The reference computed in float32: our best estimate of the 'true' answer.

    We compare fp16 kernels against this rather than against PyTorch's own fp16 output,
    because PyTorch's fp16 output has rounding errors of its own.
    """
    with torch.no_grad():
        return problem.reference(*to_fp32(inputs)).float()


def compare(out, expected, atol, rtol):
    """Elementwise check |out - expected| <= atol + rtol * |expected| (same rule as
    torch.allclose), plus details about the worst element for the agent's feedback."""
    out32 = out.float()
    diff = (out32 - expected).abs()
    allowed = atol + rtol * expected.abs()
    bad = ~(diff <= allowed)        # written this way so NaN counts as bad
    n_bad = int(bad.sum().item())
    result = {"ok": n_bad == 0, "n_bad": n_bad, "n_total": expected.numel()}
    if expected.numel() == 0:
        return result
    diff_for_max = torch.nan_to_num(diff, nan=float("inf"))
    flat_idx = int(diff_for_max.argmax().item())
    idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx), expected.shape))
    result.update({
        "max_abs_diff": float(diff_for_max.max().item()),
        "worst_index": list(idx),
        "got": float(out32[idx].item()),
        "expected": float(expected[idx].item()),
        "has_nan": bool(torch.isnan(out32).any().item()),
    })
    return result


def make_cases(problem, rng):
    """List of (label, shape, dtype) to test."""
    cases = []
    for dtype in problem.DTYPES:
        for name, shape in problem.SHAPES.items():
            cases.append((name, shape, dtype))
        for i, shape in enumerate(problem.EDGE_SHAPES):
            cases.append((f"edge{i}", shape, dtype))
        cases.append(("random", problem.random_shape(rng), dtype))
    return cases


def checked_call(kernel_fn, inputs):
    """Run the kernel once under the dynamic anti-cheat monitors.

    Returns (output, error_message_or_None).
    """
    launches_before = anticheat.triton_launch_count()
    recorder = anticheat.OpRecorder()
    with recorder:
        out = kernel_fn(*inputs)
        # Kernel launches are asynchronous; synchronize so that errors inside the GPU
        # kernel (e.g. out-of-bounds access) surface here, not at some later line.
        torch.cuda.synchronize()
    forbidden = recorder.forbidden_ops()
    if forbidden:
        return out, ("CHEAT: PyTorch did computation inside kernel(): ops "
                     f"{forbidden}. Only allocation (torch.empty etc.) and views are allowed "
                     "outside Triton kernels.")
    if anticheat.triton_launch_count() == launches_before:
        return out, "CHEAT: kernel() did not launch any Triton kernel."
    return out, None


def check_output(out, inputs, snapshots, expected, dtype):
    """Structural checks. Returns an error string or None."""
    if not isinstance(out, torch.Tensor):
        return f"kernel() must return a torch.Tensor, got {type(out).__name__}"
    if tuple(out.shape) != tuple(expected.shape):
        return f"wrong output shape: got {tuple(out.shape)}, expected {tuple(expected.shape)}"
    if out.dtype != dtype:
        return f"wrong output dtype: got {out.dtype}, expected {dtype}"
    if not out.is_cuda:
        return "output must be on the GPU"
    if not out.is_contiguous():
        return "output must be contiguous"
    for i, (inp, snap) in enumerate(zip(inputs, snapshots)):
        if anticheat.shares_memory(out, inp):
            return f"CHEAT: output shares memory with input {i}; return a new tensor"
        if not torch.equal(inp, snap):
            return f"kernel() modified input {i} in place; inputs must not be changed"
    return None


def check_correctness(problem, kernel_fn, seed):
    """Run all correctness cases. Returns a dict with 'status' == 'ok' on success.

    Other statuses: 'incorrect', 'cheat', 'compile_error', 'runtime_error'.
    """
    rng = random.Random(seed)
    cases = make_cases(problem, rng)
    for n_passed, (label, shape, dtype) in enumerate(cases):
        case = {"label": label, "shape": shape, "dtype": str(dtype).replace("torch.", "")}
        try:
            failure = _run_case(problem, kernel_fn, case, shape, dtype, rng)
        except Exception as e:  # compile errors, launch errors, illegal memory access...
            status = "compile_error" if is_compile_error(e) else "runtime_error"
            return _fail(status, error_text(e), case, n_passed, len(cases))
        if failure:
            status, message, details = failure
            return _fail(status, message, case, n_passed, len(cases), details)
        torch.cuda.empty_cache()

    return {"status": "ok", "message": f"all {len(cases)} cases passed",
            "cases_passed": len(cases), "cases_total": len(cases)}


def _run_case(problem, kernel_fn, case, shape, dtype, rng):
    """Warmup call + 2 checked calls for one shape/dtype. Returns None if all good,
    else (status, message, details)."""
    tol = problem.TOLERANCES[dtype]

    warm = problem.make_inputs(shape, dtype, rng.randrange(2**31))
    kernel_fn(*warm)  # compile + autotune happen here
    torch.cuda.synchronize()
    del warm

    for _ in range(2):
        case["seed"] = rng.randrange(2**31)
        inputs = problem.make_inputs(shape, dtype, case["seed"])
        snapshots = [t.clone() for t in inputs]
        expected = expected_output(problem, inputs)

        out, error = checked_call(kernel_fn, inputs)
        error = error or check_output(out, inputs, snapshots, expected, dtype)
        if error:
            return ("cheat" if error.startswith("CHEAT") else "incorrect"), error, None

        cmp = compare(out, expected, **tol)
        if not cmp["ok"]:
            msg = (f"{cmp['n_bad']} of {cmp['n_total']} elements outside tolerance "
                   f"(atol={tol['atol']}, rtol={tol['rtol']}). Worst at index "
                   f"{cmp['worst_index']}: got {cmp['got']:.6g}, expected "
                   f"{cmp['expected']:.6g} (abs diff {cmp['max_abs_diff']:.3g})"
                   + ("; output contains NaN" if cmp["has_nan"] else ""))
            return "incorrect", msg, cmp
        del inputs, snapshots, expected, out
    return None


def is_compile_error(e):
    try:
        from triton.compiler.errors import CompilationError
    except ImportError:
        return False
    return isinstance(e, CompilationError)


def error_text(e, max_chars=4000):
    """Traceback as text, trimmed to the end (where the useful part is)."""
    text = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    return text if len(text) <= max_chars else "...\n" + text[-max_chars:]


def _fail(status, message, case, n_passed, n_total, details=None):
    return {
        "status": status,
        "message": f"[{case['label']} shape={case['shape']} dtype={case['dtype']}] {message}",
        "failed_case": case,
        "details": details,
        "cases_passed": n_passed,
        "cases_total": n_total,
    }
