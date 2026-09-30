"""Run each candidate kernel in its own subprocess, with a timeout.

Why? A candidate kernel is untrusted, machine-generated code. It can:
  - hang (an infinite loop in Python, or a GPU kernel that never finishes),
  - crash the whole process (segfault in the driver, os.abort, ...),
  - corrupt the CUDA context (illegal memory access), after which EVERY later CUDA call in
    that process fails, even for good kernels.
If it ran inside the agent's own process, any of these would kill or poison the agent.
In a subprocess, the worst case is "that subprocess died", which we report and move on.

Note: this protects the main program from accidents, it is NOT a security sandbox. The
candidate still runs as your user with file and network access; the static checks in
anticheat.py reject imports like os/subprocess, which covers the obvious cases.
"""

import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TIMEOUT_S = 300


def run_subprocess(cmd, timeout_s, cwd=REPO_ROOT):
    """Run a command, never raising. Returns returncode/stdout/stderr/timed_out/seconds."""
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr,
                "timed_out": False, "seconds": time.time() - start}
    except subprocess.TimeoutExpired as e:
        # subprocess.run kills the child before raising TimeoutExpired.
        return {"returncode": None, "stdout": _text(e.stdout), "stderr": _text(e.stderr),
                "timed_out": True, "seconds": time.time() - start}


def _text(data):
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data


def _tail(text, n_chars=3000):
    return text if len(text) <= n_chars else "..." + text[-n_chars:]


def run_candidate(problem_name, kernel_path, seed, benchmark=True, timeout_s=DEFAULT_TIMEOUT_S):
    """Evaluate one kernel file in a fresh worker process. Always returns a result dict."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "result.json")
        cmd = [sys.executable, "-m", "harness.worker",
               "--problem", problem_name, "--kernel", os.path.abspath(kernel_path),
               "--seed", str(seed), "--out", out_path]
        if benchmark:
            cmd.append("--benchmark")
        proc = run_subprocess(cmd, timeout_s)

        if proc["timed_out"]:
            return {"status": "timeout", "seed": seed, "seconds": proc["seconds"],
                    "message": f"kernel evaluation did not finish within {timeout_s}s "
                               "(infinite loop, or far too slow)"}
        if not os.path.exists(out_path):
            return {"status": "crash", "seed": seed, "seconds": proc["seconds"],
                    "message": f"worker process died (exit code {proc['returncode']}). "
                               f"stderr:\n{_tail(proc['stderr'])}"}
        with open(out_path) as f:
            result = json.load(f)
    result["seconds"] = proc["seconds"]
    return result
