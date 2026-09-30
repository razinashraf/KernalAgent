"""Evaluate one candidate kernel file against one problem. The single entry point the
agent (Phase 2+) and the tests use.

THE KERNEL FILE FORMAT the harness expects:

    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _my_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        ...                                   # all the computation happens here

    def kernel(x):                            # same arguments as problem.reference
        out = torch.empty_like(x)             # allocation is fine
        grid = (triton.cdiv(x.numel(), 1024),)
        _my_kernel[grid](x, out, x.numel(), BLOCK=1024)
        return out                            # new contiguous tensor, same dtype

Rules: only torch/triton/math/typing imports; outside @triton.jit functions, torch may
only allocate memory and make views. See anticheat.py for the full list.

Usage:
    python -m harness.evaluate --problem gelu --kernel kernels/examples/gelu.py
"""

import argparse
import json
import random

from harness import anticheat, sandbox

# Statuses: "ok" means correct (and benchmarked if requested). Everything else is a failure:
#   static_reject, load_error, compile_error, runtime_error, incorrect, cheat, timeout, crash


def evaluate(problem_name, kernel_path, benchmark=True, seed=None, timeout_s=sandbox.DEFAULT_TIMEOUT_S):
    with open(kernel_path, encoding="utf-8") as f:
        source = f.read()
    issues = anticheat.static_check(source)
    if issues:
        return {"status": "static_reject", "message": "\n".join(issues)}

    # Fresh randomness every evaluation, from the OS, so nothing about the test inputs
    # can be predicted. The seed is recorded in the result so failures can be reproduced.
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    return sandbox.run_candidate(problem_name, kernel_path, seed, benchmark, timeout_s)


def add_speedups(result, baselines):
    """Attach speedup = baseline_time / kernel_time for each benchmarked shape (> 1 = faster)."""
    for shape_name, perf in result.get("benchmark", {}).items():
        base = baselines.get(shape_name)
        if not base:
            continue
        perf["eager_ms"] = base["eager_ms"]
        perf["compile_ms"] = base["compile_ms"]
        perf["speedup_vs_eager"] = base["eager_ms"] / perf["ms"]
        perf["speedup_vs_compile"] = base["compile_ms"] / perf["ms"]
    return result


def format_result(result):
    lines = [f"status: {result['status']}"]
    if result.get("message"):
        lines.append(f"message: {result['message']}")
    for shape_name, p in result.get("benchmark", {}).items():
        line = f"  {shape_name:>6}: {p['ms'] * 1000:9.1f} us  {p['gbps']:7.1f} GB/s"
        if p.get("pct_peak_bandwidth") is not None:
            line += f" ({p['pct_peak_bandwidth']:4.1f}% of peak)"
        if p.get("tflops") is not None:
            line += f"  {p['tflops']:6.2f} TFLOPS"
        if "speedup_vs_eager" in p:
            line += (f"  | {p['speedup_vs_eager']:.2f}x vs eager,"
                     f" {p['speedup_vs_compile']:.2f}x vs torch.compile")
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--json", action="store_true", help="print the raw result as JSON")
    args = parser.parse_args()

    result = evaluate(args.problem, args.kernel, benchmark=not args.no_benchmark, seed=args.seed)
    if result["status"] == "ok":
        from harness.baselines import load_baselines
        add_speedups(result, load_baselines().get(args.problem, {}))
    print(json.dumps(result, indent=2) if args.json else format_result(result))


if __name__ == "__main__":
    main()
