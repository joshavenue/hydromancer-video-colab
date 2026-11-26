# Hydromancer Video Colab (MiniMax-H3 Ref2VA)

Private workspace for running MiniMax-H3 `Ref2VA` Jojo clips on Google Colab Pro (NVIDIA L4 — 24 GB VRAM).

## Quick Launch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshavenue/hydromancer-video-colab/blob/main/Hydromancer_Jojo_H3_Benchmark_Colab_L4.ipynb)

1. Open the link above in Chrome on your MacBook (authorize private GitHub repo access if Colab prompts once).
2. Confirm **Runtime → Change runtime type → L4 GPU**.
3. Run **Cells 1–8** (`SHOTS_TO_RUN = "j01"` is pre-selected for a 4s sample; switch to `"j01,j02,j04,j05"` for the full 4-shot Reservoir side-by-side).

## Files

- `Hydromancer_Jojo_H3_Benchmark_Colab_L4.ipynb` — End-to-end L4 benchmark & side-by-side comparison notebook.
- `retime_colab.py` — Standalone Colab/Linux `faster-whisper` port of `tools/retime.py`.
