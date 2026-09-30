"""Benchmarking: how fast is a kernel, and how close is it to the hardware limit?

WHY NOT JUST `start = time.time(); kernel(x); print(time.time() - start)`?

1. GPU launches are ASYNCHRONOUS. `kernel(x)` only puts work in a queue for the GPU and
   returns to Python immediately, often in ~10 microseconds, long before the GPU has
   finished. time.time() would measure "how long it took to queue", not "how long it ran".
   You must synchronize (wait for the GPU) before stopping the clock, and it's better to
   use CUDA events, which are timestamps recorded by the GPU itself.

2. The FIRST call is special. Triton compiles the kernel on first use (can take
   seconds), autotuning tries many configs, and the GPU may be in a low-power clock state.
   So we WARM UP (run it a few times, untimed) before measuring.

3. One run is noisy. Other processes, clock boost and memory refresh make individual
   runs vary. We run many times and take the MEDIAN, which ignores rare slow outliers.

4. CACHES lie. If the same input is read twice in a row, the second run may find it in the
   GPU's L2 cache (fast on-chip memory) and look faster than real use. do_bench clears
   the L2 cache before each timed run by writing a large junk buffer.

triton.testing.do_bench does all four for us.

We also turn "milliseconds" into something physically meaningful:
  achieved bandwidth = bytes moved / time, compared against the GPU's peak bandwidth.
If a memory-bound kernel reaches ~80-90% of peak, there is little left to gain.
"""

import torch
import triton.testing

# Published peak DRAM bandwidth in GB/s, looked up by substring of the GPU name.
PEAK_BANDWIDTH_GBPS = {
    "T4": 320,
    "L4": 300,
    "A10G": 600,
    "V100": 900,
    "A100-SXM4-80GB": 2039,
    "A100": 1555,
    "H100": 3350,
}


def gpu_name():
    return torch.cuda.get_device_name(0)


def peak_bandwidth_gbps(name=None):
    name = name or gpu_name()
    for key, value in PEAK_BANDWIDTH_GBPS.items():   # dict order: specific keys first
        if key in name:
            return value
    return None


def time_gpu_ms(fn, warmup_ms=25, rep_ms=100):
    """Median runtime of fn() in milliseconds.

    warmup_ms / rep_ms are time budgets, not counts: do_bench first estimates how long
    one call takes, then picks how many warmup and timed runs fit in these budgets.
    """
    return triton.testing.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")


def bytes_moved(inputs, output):
    """Minimum DRAM traffic: read every input once, write the output once.

    This is a LOWER bound on what any kernel must move, so bandwidth computed from it
    is the "useful" bandwidth. A kernel that reads the input twice will look worse, which
    is exactly the point.
    """
    return sum(t.numel() * t.element_size() for t in inputs) + output.numel() * output.element_size()


def performance_summary(ms, nbytes, flops=None):
    """Turn a time into bandwidth / throughput numbers."""
    seconds = ms / 1e3
    gbps = nbytes / seconds / 1e9
    peak = peak_bandwidth_gbps()
    summary = {
        "ms": ms,
        "bytes": nbytes,
        "gbps": gbps,
        "peak_gbps": peak,
        "pct_peak_bandwidth": 100 * gbps / peak if peak else None,
    }
    if flops:
        summary["flops"] = flops
        summary["tflops"] = flops / seconds / 1e12
    return summary


def benchmark_kernel(problem, kernel_fn, seed):
    """Time a (correct) kernel on every named shape, in the benchmark dtype DTYPES[0].

    Also re-checks the output on these exact inputs, so a kernel can't behave differently
    when it's being timed.
    """
    from harness.correctness import compare, expected_output

    dtype = problem.DTYPES[0]
    results = {}
    for name, shape in problem.SHAPES.items():
        inputs = problem.make_inputs(shape, dtype, seed)
        out = kernel_fn(*inputs)
        cmp = compare(out, expected_output(problem, inputs), **problem.TOLERANCES[dtype])
        if not cmp["ok"]:
            return {"error": f"output incorrect on benchmark inputs for shape '{name}'"}

        ms = time_gpu_ms(lambda: kernel_fn(*inputs))
        flops = problem.flops(shape) if hasattr(problem, "flops") else None
        results[name] = {"shape": shape, "dtype": str(dtype).replace("torch.", ""),
                         **performance_summary(ms, bytes_moved(inputs, out), flops)}
        del inputs, out
        torch.cuda.empty_cache()
    return results
