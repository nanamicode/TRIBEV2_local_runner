# TRIBE v2 Local Runner — Windows

Local desktop runner for **TRIBE v2** brain-response prediction from video.

## Current status

The repository now contains the first executable-oriented MVP architecture:

- Windows setup EXE built by GitHub Actions;
- per-user/self-contained Python 3.11 installation;
- automatic dependency installation;
- automatic model download on first inference;
- desktop UI with explicit pipeline stages, current objective, dual progress bars and live terminal activity;
- quantized V-JEPA2/TRIBE-compatible **vision-only** path;
- raw cortical prediction export;
- robust within-clip normalization and Destrieux/fsaverage5 region aggregation;
- compact ChatGPT-ready neural summary + analysis prompt;
- whole-creative cortical maps;
- key-moment cortical maps;
- interactive 3D left/right cortical surfaces;
- local HTML brain report.

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
   - `brain_predictions.npz` — full raw cortical matrix;
   - `timeline.csv` — original prediction timeline;
   - `chatgpt_brain_summary.json` — compact AI-readable neural package;
   - `CHATGPT_ANALYSIS_PROMPT.txt` — ready-to-use interpretation instructions;
   - `creative_signature.json` — one-creative feature signature for future calibration;
   - `normalized_timeline.csv` — robust within-clip temporal normalization;
   - `roi_summary.csv` — Destrieux cortical-region aggregation;
   - `brain_left_lateral.png` / `brain_right_lateral.png`;
   - `brain_3d_left.html` / `brain_3d_right.html` when the interactive renderer is available;
   - `brain_peak_*.png` — strongest predicted cortical moments;
   - `activation_timeline.png`;
   - `run_metadata.json`;
   - `report.html` — visual brain report.

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


## AI-readable normalization

The runner does **not** send the full 20,484-vertex matrix to an LLM by default. After local inference it derives a compact, deterministic package containing:

- robust temporal z-scores inside the clip;
- mean absolute cortical response;
- left/right descriptive balance;
- spatial concentration;
- peak moments with timestamps;
- Destrieux region rankings on fsaverage5;
- a compact creative signature for later cross-creative calibration.

The compact file preserves links to the raw output while avoiding a huge token payload.

### Marketing-metric boundary

TRIBE output alone is not a validated CTR/CVR/CPA/ROAS predictor. The normalized package can support creative hypotheses and relative neural-pattern analysis. Numerical advertising metrics should only be predicted after collecting campaign outcomes and fitting/validating a calibration layer against the generated creative signatures.

A practical future calibration table is:

```
creative_signature -> hook rate / hold rate / CTR / CVR / CPA / ROAS
```

This lets the expensive cortical inference remain local while a much smaller downstream model learns which neural signatures actually correlate with business outcomes.


## Crash-safe resume / persistent analysis memory

Long CPU-only V-JEPA2 runs can take hours. The runner now keeps persistent state instead of treating every launch as a fresh analysis.

For each video/model combination it creates a stable run folder keyed by a content-aware fingerprint. Inside it:

- `run_state.json` records the last completed stage and attempt number;
- `.resume/prediction_parts/` stores cortical-inference batches as they finish;
- `brain_predictions.npz` + `timeline.csv` form the complete raw checkpoint;
- the V-JEPA2/neuralset EXCA cache remains persistent under `%LOCALAPPDATA%\T2F` on Windows.

Resume behavior:

1. If V-JEPA2 features already exist in the EXCA cache, they are reused automatically.
2. If some cortical-inference batches were already saved, they are loaded instead of recomputed.
3. If the complete raw cortical matrix exists, V-JEPA2 and TRIBE inference are skipped entirely and the runner resumes at normalization/report generation.
4. Existing normalized/report outputs are reused too.

### Windows DataLoader safe mode

The official TRIBE configuration can request around 20 DataLoader worker processes. On Windows this means process spawning and can exceed the practical capacity of a normal desktop. The runner therefore forces `data.num_workers = 0` for Windows/CPU inference and caps CPU batch size at 4. This prioritizes stability and resumability over a small amount of downstream loader parallelism; V-JEPA2 remains the dominant compute cost.
