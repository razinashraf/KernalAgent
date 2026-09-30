"""The program that runs inside the sandbox subprocess. Do not import this from the main
program; sandbox.py starts it as `python -m harness.worker ...`.

It loads one candidate kernel, checks correctness, optionally benchmarks it, and writes
the result as JSON to --out. If this process crashes or hangs, the main program notices
(no JSON file / timeout) and carries on.
"""

import argparse
import importlib.util
import json
import sys

import torch

from harness import anticheat
from harness.benchmark import benchmark_kernel
from harness.correctness import check_correctness, error_text, is_compile_error
from problems import load_problem


def load_candidate(path):
    spec = importlib.util.spec_from_file_location("candidate_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.kernel


def run(problem_name, kernel_path, seed, benchmark):
    problem = load_problem(problem_name)
    anticheat.install_launch_counter()
    try:
        kernel_fn = load_candidate(kernel_path)
    except Exception as e:
        return {"status": "load_error", "message": error_text(e)}

    # Autograd bookkeeping is not needed and would only add overhead.
    with torch.no_grad():
        result = check_correctness(problem, kernel_fn, seed)
        if result["status"] == "ok" and benchmark:
            try:
                bench = benchmark_kernel(problem, kernel_fn, seed)
            except Exception as e:
                status = "compile_error" if is_compile_error(e) else "runtime_error"
                return {"status": status, "message": "during benchmark: " + error_text(e)}
            if "error" in bench:
                return {"status": "incorrect", "message": bench["error"]}
            result["benchmark"] = bench
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    result = run(args.problem, args.kernel, args.seed, args.benchmark)
    result["seed"] = args.seed
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
