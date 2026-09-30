"""Time the PyTorch baselines every agent kernel is compared against:

  eager          problem.reference(*inputs) as written: PyTorch runs one pre-built kernel
                 per operation, writing intermediates to memory in between.
  torch.compile  the same function compiled by PyTorch's compiler (TorchInductor), which
                 fuses operations and itself generates Triton kernels. This is the serious
                 competitor: beating eager is often easy, beating torch.compile is not.

Baselines are trusted code, so they run in this process (no sandbox). Results are saved to
results/baselines.json together with the GPU name, and reused if the GPU matches.

Usage:
    python -m harness.baselines                     # all problems
    python -m harness.baselines --problems gelu softmax
"""

import argparse
import json
import os

import torch
import torch._dynamo
import triton

from harness.benchmark import gpu_name, time_gpu_ms
from problems import list_problems, load_problem

BASELINES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "results", "baselines.json")
BASELINE_SEED = 1234


def measure_problem(problem, shape_names=None):
    """{shape_name: {"eager_ms", "compile_ms", ...}} for one problem, in DTYPES[0]."""
    dtype = problem.DTYPES[0]
    torch._dynamo.reset()   # forget graphs compiled for the previous problem
    # dynamic=False: compile a specialised version for each exact shape, the fastest option.
    compiled = torch.compile(problem.reference, dynamic=False)
    results = {}
    for name, shape in problem.SHAPES.items():
        if shape_names and name not in shape_names:
            continue
        inputs = problem.make_inputs(shape, dtype, BASELINE_SEED)
        with torch.no_grad():
            eager_ms = time_gpu_ms(lambda: problem.reference(*inputs))
            compiled(*inputs)    # the first call compiles (slow); don't time it
            compile_ms = time_gpu_ms(lambda: compiled(*inputs))
        results[name] = {"shape": shape, "dtype": str(dtype).replace("torch.", ""),
                         "eager_ms": eager_ms, "compile_ms": compile_ms}
        del inputs
        torch.cuda.empty_cache()
    return results


def load_baselines(path=BASELINES_PATH):
    """{problem: {shape: {...}}} for the current GPU, or {} if none saved for this GPU."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if not torch.cuda.is_available() or data.get("gpu") != gpu_name():
        return {}
    return data["problems"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", nargs="*", default=None)
    args = parser.parse_args()

    names = args.problems or list_problems()
    existing = load_baselines()
    for name in names:
        print(f"measuring {name} ...", flush=True)
        existing[name] = measure_problem(load_problem(name))
        for shape_name, r in existing[name].items():
            print(f"  {shape_name:>6}: eager {r['eager_ms'] * 1000:9.1f} us   "
                  f"torch.compile {r['compile_ms'] * 1000:9.1f} us")

    os.makedirs(os.path.dirname(BASELINES_PATH), exist_ok=True)
    with open(BASELINES_PATH, "w") as f:
        json.dump({"gpu": gpu_name(), "torch": torch.__version__, "triton": triton.__version__,
                   "problems": existing}, f, indent=2)
    print(f"saved to {BASELINES_PATH}")


if __name__ == "__main__":
    main()
