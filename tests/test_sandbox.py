"""The sandbox must survive hangs and hard crashes of the candidate."""
import os
import sys

from conftest import BAD, requires_gpu
from harness import sandbox
from harness.evaluate import evaluate


def test_subprocess_timeout_is_enforced():
    r = sandbox.run_subprocess([sys.executable, "-c", "while True: pass"], timeout_s=3)
    assert r["timed_out"] and r["seconds"] < 30


def test_subprocess_hard_crash_is_survived():
    r = sandbox.run_subprocess([sys.executable, "-c", "import os; os.abort()"], timeout_s=30)
    assert not r["timed_out"] and r["returncode"] != 0


@requires_gpu
def test_hanging_kernel_times_out():
    result = evaluate("gelu", os.path.join(BAD, "hangs.py"), benchmark=False, timeout_s=60)
    assert result["status"] == "timeout"


@requires_gpu
def test_worker_crash_reported(tmp_path):
    # A kernel file whose import kills the process outright. It would never pass the
    # static check, so we call the sandbox directly to test crash handling.
    path = tmp_path / "crash.py"
    path.write_text("import os\nos._exit(3)\n")
    result = sandbox.run_candidate("gelu", str(path), seed=0, benchmark=False, timeout_s=120)
    assert result["status"] == "crash"
    assert "exit code 3" in result["message"]
