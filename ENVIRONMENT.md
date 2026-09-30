# Environment

Checked 2026-09-30 by running `scripts/env_check.py` on Google Colab (free tier).

## Where the code runs

| | Dev laptop (writing code) | Google Colab (running code) |
|---|---|---|
| GPU | Intel Iris Xe (integrated) — **cannot run Triton** | **NVIDIA Tesla T4** |
| OS | Windows 11 | Linux 6.6 (glibc 2.39) |
| Python | 3.13.5 | 3.13.15 |

All GPU work (tests, benchmarks, agent runs) happens on Colab. See DECISIONS.md.

## GPU: Tesla T4

| Property | Value |
|---|---|
| Architecture | Turing |
| Compute capability | 7.5 (`sm_75`) |
| Memory | 15 GB GDDR6 (14.6 GiB usable) |
| Streaming multiprocessors (SMs) | 40 |
| Peak memory bandwidth (spec sheet) | **320 GB/s** |
| Peak FP32 compute (spec sheet) | ~8.1 TFLOPS |
| Peak FP16 tensor-core compute (spec sheet) | ~65 TFLOPS |
| Power cap | 70 W |
| Driver | 580.82.07 (supports CUDA up to 13.0) |

The peak numbers are NVIDIA's published figures, not measured. We use 320 GB/s as the
"100%" line when reporting achieved memory bandwidth in Phase 3.

## Software

| Package | Version |
|---|---|
| PyTorch | 2.11.0+cu128 (built for CUDA 12.8) |
| Triton | 3.6.0 |

## Smoke tests

- Triton vector-add kernel (n = 100,003, a non-multiple of the block size): **PASS**
- `torch.compile` (GELU × 2): **PASS**

## Things to keep in mind

- **bf16:** `torch.cuda.is_bf16_supported()` returns True, but the T4 has **no native
  bf16 hardware** (that arrived with Ampere, `sm_80`). bf16 math is emulated, so it runs
  and is correct but slow. Speedups for bf16 problems on this GPU say little about newer
  GPUs; fp16 is the realistic low-precision type here.
- **Free Colab limits:** the runtime disconnects when idle and has a session cap, and a
  T4 isn't always available. Runs must be resumable from `runs/` logs.
