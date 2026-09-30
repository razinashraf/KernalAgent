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
