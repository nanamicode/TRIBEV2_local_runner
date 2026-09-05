from __future__ import annotations

import csv
import importlib.util
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


MODEL_ID = "Jessylg27/tribev2-lite-qv"


@dataclass
class RunResult:
    output_dir: Path
    predictions_path: Path
    timeline_csv: Path
    metadata_path: Path
    n_timesteps: int
    n_vertices: int
    seconds_elapsed: float


def _emit(cb: Callable[[str, float | None], None] | None, message: str, progress=None):
    if cb:
        cb(message, progress)


def get_hardware_summary() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or "unknown",
        "logical_cpus": os.cpu_count(),
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "torch": torch.__version__,
        "device": "cuda" if cuda else "cpu",
    }


def _video_duration(path: Path) -> float:
    # MoviePy is already a TRIBE dependency and is more format-tolerant than
    # trying to parse containers ourselves.
    from moviepy import VideoFileClip

    with VideoFileClip(str(path)) as clip:
        duration = float(clip.duration or 0.0)
    if duration <= 0:
        raise RuntimeError("Could not determine video duration.")
    return duration


def _patch_quantized_repo_for_windows_py311(model_dir: Path) -> None:
    """Repair pathlib class tags serialized by newer Python builds.

    Some exported TRIBE v2 config.yaml files contain PyYAML object tags such as
    `pathlib._local.WindowsPath`, which exist in newer Python versions but not
    in Python 3.11. Rewriting those tags to the public pathlib classes preserves
    the represented path while making the config loadable on our Windows 3.11
    runtime.
    """
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        return
    text = config_path.read_text(encoding="utf-8")
    patched = (
        text.replace("pathlib._local.WindowsPath", "pathlib.WindowsPath")
            .replace("pathlib._local.PosixPath", "pathlib.PosixPath")
            .replace("pathlib._local.Path", "pathlib.Path")
    )
    if patched != text:
        config_path.write_text(patched, encoding="utf-8")


def _load_quantized_tribe(model_dir: Path, cache_dir: Path, device: str):
    _patch_quantized_repo_for_windows_py311(model_dir)
    loader_path = model_dir / "load_quantized_tribev2.py"
    if not loader_path.exists():
        raise FileNotFoundError(f"Quantized loader not found: {loader_path}")

    spec = importlib.util.spec_from_file_location("tribev2_quantized_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the quantized TRIBE v2 loader.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_quantized_tribev2(
        model_dir,
        device=device,
        cache_folder=str(cache_dir),
    )


def _make_visual_event(video: Path, duration: float) -> pd.DataFrame:
    # This intentionally bypasses TribeModel.get_events_dataframe(), because
    # that method also extracts/transcribes audio. The MVP's default mode is
    # visual-only, which keeps first-run memory and model downloads practical.
    return pd.DataFrame(
        [
            {
                "type": "Video",
                "filepath": str(video.resolve()),
                "start": 0.0,
                "duration": duration,
                "timeline": "default",
                "subject": "default",
            }
        ]
    )


def run_video(
    video_path: str | Path,
    output_root: str | Path,
    progress_cb: Callable[[str, float | None], None] | None = None,
    device: str = "auto",
) -> RunResult:
    from huggingface_hub import snapshot_download
    import torch

    started = time.time()
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(video)

    root = Path(output_root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = root / f"{video.stem}-tribev2-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / ".tribev2-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _emit(progress_cb, "Checking hardware…", 0.02)
    hardware = get_hardware_summary()

    _emit(progress_cb, "Downloading/checking the quantized TRIBE v2 package…", 0.08)
    model_dir = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=str(cache_dir / "huggingface"),
        )
    )

    _emit(progress_cb, f"Loading TRIBE v2 on {device.upper()}…", 0.20)
    model = _load_quantized_tribe(model_dir, cache_dir / "features", device)

    _emit(progress_cb, "Reading video metadata…", 0.32)
    duration = _video_duration(video)
    events = _make_visual_event(video, duration)

    _emit(progress_cb, "Running cortical prediction…", 0.38)
    predictions, segments = model.predict(events=events, verbose=False)
    predictions = np.asarray(predictions, dtype=np.float32)

    if predictions.ndim != 2 or predictions.shape[0] == 0:
        raise RuntimeError(f"Unexpected TRIBE v2 output shape: {predictions.shape}")

    n_timesteps, n_vertices = predictions.shape
    _emit(progress_cb, "Saving raw predictions…", 0.86)

    pred_path = run_dir / "brain_predictions.npz"
    np.savez_compressed(
        pred_path,
        predictions=predictions,
        video_path=str(video.resolve()),
        duration_seconds=np.float32(duration),
    )

    # TRIBE v2 outputs one cortical vector per retained TR segment. We keep a
    # generic sequential timeline here and include segment timing when exposed
    # by the returned segment objects.
    timeline_path = run_dir / "timeline.csv"
    with timeline_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "start_seconds",
                "duration_seconds",
                "mean",
                "mean_abs",
                "std",
                "max",
                "min",
            ]
        )
        for i, row in enumerate(predictions):
            seg = segments[i] if i < len(segments) else None
            start = getattr(seg, "start", i)
            seg_duration = getattr(seg, "duration", 1.0)
            writer.writerow(
                [
                    i,
                    float(start),
                    float(seg_duration),
                    float(row.mean()),
                    float(np.abs(row).mean()),
                    float(row.std()),
                    float(row.max()),
                    float(row.min()),
                ]
            )

    metadata = {
        "runner": "TRIBE v2 Local Runner",
        "mode": "vision-only",
        "model_id": MODEL_ID,
        "video": str(video.resolve()),
        "video_duration_seconds": duration,
        "prediction_shape": [int(n_timesteps), int(n_vertices)],
        "hardware": hardware,
        "device_used": device,
        "elapsed_seconds": time.time() - started,
        "note": (
            "These are model-predicted fMRI-like cortical responses for an "
            "average subject, not measurements of a specific person's brain."
        ),
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _emit(progress_cb, "Creating cortical maps and local report…", 0.90)
    from visualize import create_report_assets

    create_report_assets(
        predictions=predictions,
        timeline_csv=timeline_path,
        output_dir=run_dir,
        metadata=metadata,
    )

    elapsed = time.time() - started
    metadata["elapsed_seconds"] = elapsed
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _emit(progress_cb, "Done.", 1.0)

    return RunResult(
        output_dir=run_dir,
        predictions_path=pred_path,
        timeline_csv=timeline_path,
        metadata_path=metadata_path,
        n_timesteps=n_timesteps,
        n_vertices=n_vertices,
        seconds_elapsed=elapsed,
    )
