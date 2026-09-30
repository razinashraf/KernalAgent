"""Benchmark sanity checks."""
import os

import torch

from conftest import EXAMPLES, requires_gpu
from harness import benchmark
from harness.evaluate import evaluate


def test_peak_bandwidth_lookup():
    assert benchmark.peak_bandwidth_gbps("Tesla T4") == 320
    assert benchmark.peak_bandwidth_gbps("NVIDIA A100-SXM4-80GB") == 2039
    assert benchmark.peak_bandwidth_gbps("Unknown GPU") is None


@requires_gpu
def test_time_gpu_is_positive_and_scales():
    small = torch.randn(1 << 20, device="cuda")
    big = torch.randn(1 << 26, device="cuda")
    t_small = benchmark.time_gpu_ms(lambda: small * 2)
    t_big = benchmark.time_gpu_ms(lambda: big * 2)
    assert 0 < t_small < t_big   # 64x more data must take longer


@requires_gpu
def test_bytes_moved():
    x = torch.empty(10, 10, dtype=torch.float16, device="cuda")
    assert benchmark.bytes_moved([x, x], x) == 3 * 200


@requires_gpu
def test_example_gelu_benchmark_is_physically_plausible():
    result = evaluate("gelu", os.path.join(EXAMPLES, "gelu.py"), benchmark=True)
    assert result["status"] == "ok", result.get("message")
    large = result["benchmark"]["large"]
    assert large["ms"] > 0
    # Nothing can beat the hardware: achieved bandwidth must be below peak (+10% slack
    # for spec-sheet rounding). If this fails, the timing is broken, not the kernel.
    if large["peak_gbps"]:
        assert large["gbps"] < 1.1 * large["peak_gbps"]


@requires_gpu
def test_baselines_measure_one_problem():
    from harness.baselines import measure_problem
    from problems import load_problem
    r = measure_problem(load_problem("gelu"), shape_names=["small"])
    assert r["small"]["eager_ms"] > 0 and r["small"]["compile_ms"] > 0
