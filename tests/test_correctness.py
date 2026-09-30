"""End-to-end: known-good kernels pass, each kind of bad kernel is caught for the right
reason. These run the full evaluate() pipeline, including the sandbox subprocess."""
import os

import pytest

from conftest import BAD, EXAMPLES, requires_gpu
from harness.evaluate import evaluate

pytestmark = requires_gpu


@pytest.mark.parametrize("problem", ["gelu", "softmax", "rmsnorm"])
def test_example_kernels_are_correct(problem):
    result = evaluate(problem, os.path.join(EXAMPLES, f"{problem}.py"), benchmark=False)
    assert result["status"] == "ok", result.get("message")
    assert result["cases_passed"] == result["cases_total"] > 0


@pytest.mark.parametrize("bad_file, expected_statuses", [
    ("wrong_math.py", {"incorrect"}),
    ("grid_too_small.py", {"incorrect"}),
    ("calls_torch.py", {"static_reject"}),
    ("sneaky_torch_ops.py", {"cheat"}),
    ("caches_output.py", {"incorrect"}),
    ("no_launch.py", {"cheat"}),
    ("modifies_input.py", {"cheat", "incorrect"}),
    ("compile_error.py", {"compile_error"}),
])
def test_bad_kernels_are_caught(bad_file, expected_statuses):
    result = evaluate("gelu", os.path.join(BAD, bad_file), benchmark=False)
    assert result["status"] in expected_statuses, result
    assert result.get("message")


def test_grid_too_small_passes_pow2_but_fails_edge():
    result = evaluate("gelu", os.path.join(BAD, "grid_too_small.py"), benchmark=False)
    assert result["status"] == "incorrect"
    assert result["failed_case"]["label"].startswith(("edge", "random"))


def test_incorrect_message_has_numbers():
    result = evaluate("gelu", os.path.join(BAD, "wrong_math.py"), benchmark=False)
    assert "Worst at index" in result["message"]
    assert result["details"]["max_abs_diff"] > 0
