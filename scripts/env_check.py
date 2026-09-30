"""Phase 0 environment check.

Run this on the GPU machine (e.g. Google Colab with a T4 runtime):

    python scripts/env_check.py

It prints everything we need for ENVIRONMENT.md and runs one tiny Triton
kernel to prove that the whole toolchain (driver -> CUDA -> PyTorch -> Triton)
actually works, not just that the packages import.
"""

import platform
import subprocess
import sys


def section(title):
    print(f"\n=== {title} ===")


def main():
    section("nvidia-smi")
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        print(out.stdout or out.stderr)
    except FileNotFoundError:
        print("nvidia-smi NOT FOUND -> no NVIDIA driver on this machine. Stop here.")
        sys.exit(1)

    section("Python / OS")
    print("python  :", sys.version.split()[0])
    print("platform:", platform.platform())

    section("PyTorch")
    import torch

    print("torch        :", torch.__version__)
    print("torch CUDA   :", torch.version.cuda)
    print("cuda avail   :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("PyTorch cannot see a GPU. Stop here.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    print("GPU name     :", props.name)
    print("GPU memory   : %.1f GiB" % (props.total_memory / 1024**3))
    # Compute capability ("sm_XY") decides which hardware features exist.
    # e.g. 7.5 = Turing (T4), 8.0 = Ampere (A100). Native bf16 needs >= 8.0.
    print("compute cap  : %d.%d" % (props.major, props.minor))
    print("SM count     :", props.multi_processor_count)
    print("bf16 support :", torch.cuda.is_bf16_supported())

    section("Triton")
    import triton
    import triton.language as tl

    print("triton:", triton.__version__)

    # Smallest possible real Triton kernel: out = x + y.
    # Each "program" (think: one CUDA thread block) handles BLOCK elements.
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n  # guard the last, partially-full block
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 100_003  # deliberately not a multiple of BLOCK, to exercise the mask
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    ok = torch.allclose(out, x + y)
    print("triton smoke test (vector add):", "PASS" if ok else "FAIL")

    section("torch.compile")
    try:
        f = torch.compile(lambda a: torch.nn.functional.gelu(a) * 2)
        r = f(x)
        print("torch.compile smoke test:", "PASS" if torch.allclose(r, torch.nn.functional.gelu(x) * 2, atol=1e-5) else "FAIL")
    except Exception as e:  # report, don't crash: this is a diagnostic script
        print("torch.compile FAILED:", repr(e)[:500])


if __name__ == "__main__":
    main()
