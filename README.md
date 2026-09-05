# TRIBE v2 Local Runner — Windows

Local desktop runner for **TRIBE v2** brain-response prediction from video.

## Current status

The repository now contains the first executable-oriented MVP architecture:

- Windows setup EXE built by GitHub Actions;
- per-user/self-contained Python 3.11 installation;
- automatic dependency installation;
- automatic model download on first inference;
- simple desktop UI: choose video -> choose output -> analyze;
- quantized V-JEPA2/TRIBE-compatible **vision-only** path;
- raw cortical prediction export;
- timeline CSV;
- fsaverage5 cortical activation images;
- local HTML report.

**Important:** the code path is implemented, but it still needs a real Windows end-to-end validation run on target hardware before calling the MVP production-ready.

## Why this path

Meta's official TRIBE v2 combines:

- Llama 3.2 3B text features;
- Wav2Vec-BERT audio features;
- V-JEPA2 ViT-G video features;
- an 8-layer multimodal cortical encoder that predicts fsaverage5 activity.

Running every branch is unnecessarily heavy for the first desktop MVP. The current version starts with video only and preserves the original cortical head.

The selected long-term optimization path is **native Rust**, using the public `eugenehp/tribev2-rs` work as the foundation for the cortical encoder. See [ARCHITECTURE.md](ARCHITECTURE.md).

## First-run flow

1. Download `TRIBEv2LocalRunner-Setup.exe` from the GitHub Actions artifact.
2. Run it.
3. The setup installs a private Python runtime and local dependencies.
4. Start the app.
5. Select MP4/MOV/MKV/AVI/WEBM.
6. Click **Analyze video locally**.
7. On first use the model files are downloaded and cached.
8. The result folder receives:
   - `brain_predictions.npz`
   - `timeline.csv`
   - `brain_left_lateral.png`
   - `brain_right_lateral.png`
   - `activation_timeline.png`
   - `run_metadata.json`
   - `report.html`

No inference VPS is required.

## Build the Windows setup

GitHub:

```
Actions -> Build Windows bootstrap -> Run workflow
```

Artifact name:

```
TRIBEv2LocalRunner-Windows
```

## Local development

Windows / Python 3.11:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -U pip setuptools wheel
.venv\Scripts\pip install -r requirements-runtime.txt
.venv\Scripts\pip install --no-deps https://github.com/facebookresearch/tribev2/archive/refs/heads/main.zip
.venv\Scripts\python runtime\app.py
```

## Target hardware

Initial test target rather than a guaranteed minimum:

- Windows 10/11 x64;
- 16 GB system RAM or more;
- modern 6-core+ CPU;
- SSD;
- NVIDIA GPU is optional, but a recent CUDA-capable GPU with useful VRAM should materially improve the heavy visual encoder path.

CPU-only is intentionally supported; it may simply take substantially longer than video duration on slower machines.

## Accuracy rule

The optimization goal is **not** to generate something that merely looks like a neural heatmap. Any quantization, lower-level port, smaller encoder or distillation must be compared against official TRIBE v2 outputs.

A visually convincing but feature-incompatible model is a failed optimization.

## Semantics

TRIBE v2 outputs are predicted **fMRI-like cortical responses for an average subject**. They are not measurements of the actual brain activity of a viewer.

## License warning

Meta's TRIBE v2 release / pretrained weights are **CC BY-NC 4.0**. The runner does not bypass that restriction. If this becomes part of a commercial creative-analysis operation, the relevant permission/license must be resolved before deployment.
