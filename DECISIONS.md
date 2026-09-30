# Decisions

Choices made where the spec was ambiguous. Simpler option picked each time.

## Phase 0

- **No local NVIDIA GPU.** The dev laptop only has Intel Iris Xe graphics, which can't
  run Triton. Code is written locally; everything is run and tested on a free
  **Google Colab T4 GPU**. Colab was chosen over Kaggle because Kaggle's P100 option is
  too old for current Triton, and Colab has a simpler notebook-to-GitHub workflow.
- **Code transfer:** local repo -> GitHub -> `git clone` inside Colab.
- **Dependencies:** Colab already ships PyTorch + Triton that are built to match each
  other and the driver. So `requirements.txt` does NOT pin torch/triton; reinstalling
  them risks breaking that match. No venv/uv on Colab (the runtime is throwaway anyway).
- **bf16 on T4 is emulated** (see ENVIRONMENT.md). Problems will mainly use fp32/fp16;
  bf16 cases, if any, are kept few and labelled as emulated.
- **Peak bandwidth for roofline/feedback:** the T4 spec-sheet value, 320 GB/s.

## Phase 1

- **Problem file format:** plain module-level constants and functions (no classes). A
  shape is a dict of named dims (`{"M": 4096, "N": 1024}`) so it reads clearly in logs.
- **12 problems:** gelu, silu_mul, rope (elementwise); softmax, layernorm, rmsnorm,
  logsumexp (reductions); bias_gelu_residual, add_layernorm (fused); transpose (pure
  memory movement, added to teach coalescing); matmul_bias_relu, attention (harder).
- **One output tensor per problem.** add_layernorm returns only the normalized output,
  not the residual sum, to keep the harness simple.
- **Output dtype = input dtype** for every problem; no bf16 (emulated on T4).
- **Ground truth = reference computed in float32**, even for fp16 cases. A test checks
  that PyTorch's own fp16 result passes each tolerance, so tolerances aren't impossible.
- **Benchmark dtype = DTYPES[0] = fp16** for all problems. fp32 is correctness-only.
- **Attention reference uses F.scaled_dot_product_attention** (a fused kernel), so the
  eager baseline is strong and honest rather than a slow naive version.
- **torch.compile baseline uses default mode with dynamic=False.** max-autotune would be
  a stronger baseline but compiles far slower on a free Colab session.
- **Anti-cheat is two layers:** static AST checks give quick, readable rejections; the
  real enforcement is dynamic. A TorchDispatchMode records every PyTorch op run during the
  kernel call, and only allocation/view/metadata ops are allowed. We also count Triton
  launches (must be >= 1), check that outputs don't alias inputs and inputs aren't
  modified, and use fresh OS-random seeds plus one never-seen random shape per run.
- **The warmup call is not monitored.** Triton autotuning runs on the first call and
  may legitimately use torch ops (e.g. zeroing a buffer), so only calls 2 and 3 are checked.
- **The sandbox is a subprocess with a timeout, not a security boundary.** It protects
  the agent from crashes/hangs/CUDA-context corruption. Static import checks block
  os/subprocess etc., which is enough for LLM-written code on a throwaway Colab VM.
- **Bandwidth = minimum bytes (read each input once + write output once) / time.**
