# TRIBE v2 Local Runner (Windows MVP)

Local Windows runner for **TRIBE v2** brain-response prediction from video.

## Goal

Turn the public TRIBE v2 research release into a local desktop workflow:

1. run a Windows installer/launcher;
2. install a private Python runtime and dependencies under `%LOCALAPPDATA%\TRIBEv2LocalRunner`;
3. download the model files on first use;
4. choose a video;
5. run local inference on CPU or NVIDIA CUDA;
6. save:
   - raw per-timepoint cortical predictions (`.npz`),
   - a CSV timeline,
   - a cortical activation PNG,
   - a timeline PNG,
   - a small HTML report.

The first implementation deliberately defaults to a **vision-only TRIBE v2 path**. This avoids the gated Llama 3.2 text encoder and keeps the first MVP much more realistic on a normal desktop PC. The TRIBE checkpoint was trained with modality dropout and its feature configuration supports missing modalities, so this is a useful local path while keeping the original TRIBE v2 cortical head.

## Runtime profile used by the MVP

The default model package is:

- `Jessylg27/tribev2-lite-qv`
- original `facebook/tribev2` brain checkpoint;
- ViT-G-compatible V-JEPA2 video branch;
- TorchAO `Int8WeightOnlyConfig`;
- ~1.75 GB model package;
- 1 Hz video event frequency / batch size 2.

This profile keeps the video feature dimension expected by the untouched TRIBE v2 checkpoint.

## Important licensing note

Meta's TRIBE v2 release and checkpoint are published under **CC BY-NC 4.0**. That means this repository is suitable for research/prototyping unless you have separate permission for commercial use. The runner does not remove or bypass that restriction.

## Build

The Windows bootstrap executable is built with GitHub Actions:

```
Actions -> Build Windows bootstrap -> Run workflow
```

Artifact: `TRIBEv2LocalRunner-Windows`.

The bootstrap executable itself is intentionally small. On first run it downloads Python and the runtime dependencies, then launches the desktop app.

## Development run

Windows, Python 3.11+:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\pip install -r requirements-runtime.txt
.venv\Scripts\python runtime\app.py
```

## Architecture

```
bootstrap/installer.py
  -> installs per-user Python 3.11 if needed
  -> creates %LOCALAPPDATA%/TRIBEv2LocalRunner/env
  -> installs runtime requirements
  -> copies runtime files
  -> creates Desktop shortcut
  -> launches runtime/app.py

runtime/app.py
  -> native Tkinter UI
  -> choose video/output folder
  -> hardware profile + progress
  -> invokes runtime/engine.py

runtime/engine.py
  -> downloads quantized TRIBE package from Hugging Face
  -> loads packaged quantized V-JEPA2 branch
  -> creates visual-only video events
  -> executes TRIBE v2 cortical prediction
  -> writes npz/csv metadata

runtime/visualize.py
  -> creates aggregate fsaverage5 cortical maps
  -> creates activation-over-time chart
  -> generates a local HTML report
```

## What this is / is not

TRIBE v2 predicts **fMRI-like cortical responses for an average subject** from stimuli. This runner therefore outputs a *model prediction* of cortical activity, not a measurement of the actual viewer's brain and not an EEG-style attention score.

## Next engineering steps

- validate the quantized Windows path against official FP32 TRIBE outputs on fixed clips;
- add optional full multimodal mode (audio + transcript + Llama) behind a Hugging Face token/license step;
- add DirectML/ONNX experiments if they preserve feature compatibility;
- add resumable feature caching and chunk-level checkpointing for long videos;
- benchmark CPU vs CUDA on several common desktop classes.
