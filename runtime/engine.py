from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


MODEL_ID = "Jessylg27/tribev2-lite-qv"
RUNNER_MODE = "vision-only"
RESUME_SCHEMA = 2


def _patch_exca_windows_uid_folders() -> None:
    """Hash EXCA cache folder UIDs on Windows.

    EXCA normally embeds verbose extractor configuration in cache paths.
    Hashing only the on-disk folder name keeps cache identity deterministic
    while avoiding Windows MAX_PATH failures.
    """
    if os.name != "nt":
        return

    try:
        from exca.base import BaseInfra
    except Exception:
        return

    if getattr(BaseInfra, "_tribev2_short_uid_folders", False):
        return

    def _short_uid_folder(self, create: bool = False):
        if self.folder is None:
            return None
        logical_uid = self.uid()
        digest = hashlib.sha256(logical_uid.encode("utf-8")).hexdigest()[:32]
        folder = Path(self.folder) / f"u-{digest}"
        if create:
            folder.mkdir(parents=True, exist_ok=True)
        return folder

    BaseInfra.uid_folder = _short_uid_folder
    BaseInfra._tribev2_short_uid_folders = True


def _short_feature_cache_dir() -> Path:
    """Return a deliberately short persistent feature-cache path."""
    override = os.environ.get("TRIBEV2_FEATURE_CACHE")
    if override:
        path = Path(override)
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA") or str(Path.home())
        path = Path(local) / "T2F"
    else:
        path = Path.home() / ".cache" / "tribev2" / "features"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class RunResult:
    output_dir: Path
    predictions_path: Path
    timeline_csv: Path
    metadata_path: Path
    normalized_path: Path
    report_path: Path
    n_timesteps: int
    n_vertices: int
    seconds_elapsed: float


def _emit(
    cb: Callable | None,
    message: str,
    progress: float | None = None,
    stage: str | None = None,
):
    if not cb:
        return
    try:
        cb(message, progress, stage)
    except TypeError:
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
    from moviepy import VideoFileClip

    with VideoFileClip(str(path)) as clip:
        duration = float(clip.duration or 0.0)
    if duration <= 0:
        raise RuntimeError("Could not determine video duration.")
    return duration


def _video_fingerprint(path: Path) -> str:
    """Fast content-aware fingerprint used to identify resumable runs."""
    stat = path.stat()
    h = hashlib.sha256()
    h.update(str(RESUME_SCHEMA).encode())
    h.update(MODEL_ID.encode())
    h.update(RUNNER_MODE.encode())
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime_ns).encode())

    chunk = 1024 * 1024
    with path.open("rb") as f:
        h.update(f.read(chunk))
        if stat.st_size > chunk:
            f.seek(max(0, stat.st_size - chunk))
            h.update(f.read(chunk))
    return h.hexdigest()


def _safe_stem(path: Path) -> str:
    stem = re.sub(r'[<>:"/\\|?*]+', "_", path.stem).strip(" .")
    stem = re.sub(r"\s+", " ", stem)
    return (stem or "video")[:72]


def _stable_run_dir(root: Path, video: Path, fingerprint: str) -> Path:
    return root / f"{_safe_stem(video)}-tribev2-{fingerprint[:10]}"


def _atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_state(run_dir: Path) -> dict:
    path = run_dir / "run_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _checkpoint(run_dir: Path, state: dict, stage: str, **extra) -> dict:
    state = dict(state)
    state.update(extra)
    state["schema"] = RESUME_SCHEMA
    state["stage"] = stage
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    completed = list(state.get("completed_stages", []))
    if stage not in completed:
        completed.append(stage)
    state["completed_stages"] = completed
    _atomic_json(run_dir / "run_state.json", state)
    return state


def _patch_quantized_repo_for_windows_py311(model_dir: Path) -> None:
    """Repair pathlib class tags serialized by newer Python builds."""
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


def _install_quantized_loader_compat(module) -> None:
    """Patch small config-schema drifts in third-party quantized loaders."""
    original = getattr(module, "TribeModel", None)
    if original is None:
        return

    class CompatTribeModel(original):
        def __init__(self, **kwargs):
            data_cfg = kwargs.get("data")
            if isinstance(data_cfg, dict):
                for feature_name in ("image_feature", "video_feature"):
                    feature_cfg = data_cfg.get(feature_name)
                    if isinstance(feature_cfg, dict):
                        feature_cfg.pop("device", None)
            super().__init__(**kwargs)

    module.TribeModel = CompatTribeModel


def _load_quantized_tribe(model_dir: Path, cache_dir: Path, device: str):
    _patch_exca_windows_uid_folders()
    _patch_quantized_repo_for_windows_py311(model_dir)
    loader_path = model_dir / "load_quantized_tribev2.py"
    if not loader_path.exists():
        raise FileNotFoundError(f"Quantized loader not found: {loader_path}")

    spec = importlib.util.spec_from_file_location("tribev2_quantized_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import the quantized TRIBE v2 loader.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _install_quantized_loader_compat(module)
    return module.load_quantized_tribev2(
        model_dir,
        device=device,
        cache_folder=str(cache_dir),
    )


def _configure_safe_runtime(model, device: str) -> dict:
    """Tune inference for ordinary Windows desktops.

    The official config may request ~20 DataLoader workers. On Windows each
    worker is a spawned process, which can freeze or exhaust RAM on a 12-thread
    desktop. Local inference is safer with zero worker processes; V-JEPA2 is
    already the dominant cost and its persistent EXCA cache remains enabled.
    """
    import torch

    tuning = {}
    logical = max(1, os.cpu_count() or 1)

    try:
        torch.set_num_threads(min(logical, 12))
        tuning["torch_num_threads"] = min(logical, 12)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(min(4, logical))
        tuning["torch_interop_threads"] = min(4, logical)
    except Exception:
        pass

    data = getattr(model, "data", None)
    if data is not None:
        old_workers = getattr(data, "num_workers", None)
        if os.name == "nt" or device == "cpu":
            try:
                data.num_workers = 0
                tuning["dataloader_num_workers_before"] = old_workers
                tuning["dataloader_num_workers"] = 0
            except Exception:
                pass

        if device == "cpu":
            try:
                old_batch = int(getattr(data, "batch_size", 4) or 4)
                safe_batch = max(1, min(old_batch, 4))
                data.batch_size = safe_batch
                tuning["batch_size_before"] = old_batch
                tuning["batch_size"] = safe_batch
            except Exception:
                pass

    return tuning


def _make_visual_event(video: Path, duration: float) -> pd.DataFrame:
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


def _expanded_batch_segments(model, batch) -> tuple[list, np.ndarray]:
    segments = []
    for segment in batch.segments:
        for t in np.arange(0, segment.duration - 1e-2, model.data.TR):
            segments.append(segment.copy(offset=t, duration=model.data.TR))

    if model.remove_empty_segments:
        keep = np.array([len(s.ns_events) > 0 for s in segments], dtype=bool)
    else:
        keep = np.ones(len(segments), dtype=bool)
    kept_segments = [s for i, s in enumerate(segments) if keep[i]]
    return kept_segments, keep


def _predict_resumable(
    model,
    events: pd.DataFrame,
    run_dir: Path,
    state: dict,
    progress_cb: Callable | None,
) -> tuple[np.ndarray, list, dict]:
    """Run TRIBE inference with persistent per-batch prediction checkpoints.

    V-JEPA2/neuralset feature extraction is cached persistently by EXCA in T2F.
    Once get_loaders() completes, later attempts reuse those features. The
    cortical inference output is additionally checkpointed batch-by-batch here.
    """
    import torch
    from einops import rearrange

    if getattr(model, "_model", None) is None:
        raise RuntimeError("TRIBE model weights are not loaded.")

    _emit(
        progress_cb,
        "Preparing video features. Existing V-JEPA2 cache will be reused automatically…",
        0.30,
        "predict",
    )
    loader = model.data.get_loaders(events=events, split_to_build="all")["all"]
    state = _checkpoint(
        run_dir,
        state,
        "features_ready",
        feature_cache=str(_short_feature_cache_dir()),
        loader_batches=len(loader),
    )

    core = model._model
    parts_dir = run_dir / ".resume" / "prediction_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    preds: list[np.ndarray] = []
    all_segments: list = []
    total_batches = max(1, len(loader))
    completed_batches: list[int] = []

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            kept_segments, keep = _expanded_batch_segments(model, batch)
            part_path = parts_dir / f"batch_{batch_idx:05d}.npz"
            y_pred = None

            if part_path.exists():
                try:
                    cached = np.load(part_path, allow_pickle=False)["predictions"]
                    if cached.ndim == 2 and cached.shape[0] == len(kept_segments):
                        y_pred = np.asarray(cached, dtype=np.float32)
                        _emit(
                            progress_cb,
                            f"Recovered cortical batch {batch_idx + 1}/{total_batches} from checkpoint.",
                            0.48 + 0.27 * ((batch_idx + 1) / total_batches),
                            "predict",
                        )
                except Exception:
                    y_pred = None

            if y_pred is None:
                batch = batch.to(core.device)
                raw = core(batch).detach().cpu().numpy()
                y_pred = rearrange(raw, "b d t -> (b t) d")[keep]
                y_pred = np.asarray(y_pred, dtype=np.float32)

                tmp = part_path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, predictions=y_pred)
                tmp.replace(part_path)

            preds.append(y_pred)
            all_segments.extend(kept_segments)
            completed_batches.append(batch_idx)

            state = _checkpoint(
                run_dir,
                state,
                "cortical_inference",
                completed_prediction_batches=completed_batches,
                total_prediction_batches=total_batches,
            )
            _emit(
                progress_cb,
                f"Cortical inference batch {batch_idx + 1}/{total_batches} saved persistently.",
                0.48 + 0.27 * ((batch_idx + 1) / total_batches),
                "predict",
            )

    if not preds:
        raise RuntimeError("TRIBE produced no cortical prediction batches.")

    predictions = np.concatenate(preds, axis=0)
    if len(all_segments) != predictions.shape[0]:
        raise ValueError(
            f"Number of samples: {predictions.shape[0]} != {len(all_segments)}"
        )

    state = _checkpoint(
        run_dir,
        state,
        "cortical_inference_complete",
        n_timesteps=int(predictions.shape[0]),
        n_vertices=int(predictions.shape[1]),
    )
    return predictions, all_segments, state


def _write_timeline(path: Path, predictions: np.ndarray, segments: list) -> None:
    tmp = path.with_suffix(".tmp.csv")
    with tmp.open("w", newline="", encoding="utf-8") as f:
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
    tmp.replace(path)


def _load_raw_checkpoint(pred_path: Path, timeline_path: Path):
    if not pred_path.exists() or not timeline_path.exists():
        return None
    try:
        predictions = np.asarray(
            np.load(pred_path, allow_pickle=False)["predictions"],
            dtype=np.float32,
        )
        if predictions.ndim != 2 or predictions.shape[0] == 0:
            return None
        timeline = pd.read_csv(timeline_path)
        if len(timeline) != predictions.shape[0]:
            return None
        return predictions
    except Exception:
        return None


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
    root.mkdir(parents=True, exist_ok=True)

    fingerprint = _video_fingerprint(video)
    run_dir = _stable_run_dir(root, video, fingerprint)
    run_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = root / ".tribev2-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_dir = _short_feature_cache_dir()

    state = _read_state(run_dir)
    attempts = int(state.get("attempts", 0)) + 1
    state.update(
        {
            "fingerprint": fingerprint,
            "video": str(video.resolve()),
            "model_id": MODEL_ID,
            "mode": RUNNER_MODE,
            "attempts": attempts,
        }
    )
    state = _checkpoint(run_dir, state, "started")

    if attempts > 1:
        _emit(
            progress_cb,
            f"Resume session found — continuing attempt {attempts} without discarding saved work.",
            0.01,
            "prepare",
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _emit(
        progress_cb,
        "Checking local hardware and selecting the inference device…",
        0.02,
        "prepare",
    )
    hardware = get_hardware_summary()
    duration = _video_duration(video)

    pred_path = run_dir / "brain_predictions.npz"
    timeline_path = run_dir / "timeline.csv"
    metadata_path = run_dir / "run_metadata.json"

    predictions = _load_raw_checkpoint(pred_path, timeline_path)
    segments = []
    tuning: dict = {}

    if predictions is not None:
        _emit(
            progress_cb,
            "Recovered complete raw cortical predictions — skipping V-JEPA2 and TRIBE inference.",
            0.78,
            "save_raw",
        )
        state = _checkpoint(
            run_dir,
            state,
            "raw_predictions_recovered",
            n_timesteps=int(predictions.shape[0]),
            n_vertices=int(predictions.shape[1]),
        )
    else:
        _emit(
            progress_cb,
            "Checking the quantized TRIBE v2 package and local model cache…",
            0.06,
            "model_download",
        )
        model_dir = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                cache_dir=str(cache_dir / "huggingface"),
            )
        )
        state = _checkpoint(run_dir, state, "model_package_ready")

        _emit(
            progress_cb,
            f"Loading the TRIBE v2 cortical model on {device.upper()}…",
            0.14,
            "model_load",
        )
        model = _load_quantized_tribe(model_dir, feature_cache_dir, device)
        tuning = _configure_safe_runtime(model, device)
        state = _checkpoint(run_dir, state, "model_loaded", runtime_tuning=tuning)

        if tuning.get("dataloader_num_workers") == 0:
            _emit(
                progress_cb,
                "Windows safe mode enabled: DataLoader workers reduced to 0 to prevent process-spawn freezes.",
                0.20,
                "model_load",
            )

        _emit(
            progress_cb,
            "Reading video duration and preparing the stimulus timeline…",
            0.24,
            "video_prepare",
        )
        events = _make_visual_event(video, duration)
        state = _checkpoint(run_dir, state, "video_ready", duration_seconds=duration)

        predictions, segments, state = _predict_resumable(
            model=model,
            events=events,
            run_dir=run_dir,
            state=state,
            progress_cb=progress_cb,
        )
        predictions = np.asarray(predictions, dtype=np.float32)

        if predictions.ndim != 2 or predictions.shape[0] == 0:
            raise RuntimeError(f"Unexpected TRIBE v2 output shape: {predictions.shape}")

        _emit(
            progress_cb,
            "Saving the full raw cortical prediction matrix…",
            0.78,
            "save_raw",
        )
        _write_timeline(timeline_path, predictions, segments)

        tmp_pred = pred_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_pred,
            predictions=predictions,
            video_path=str(video.resolve()),
            duration_seconds=np.float32(duration),
            fingerprint=fingerprint,
        )
        tmp_pred.replace(pred_path)
        state = _checkpoint(run_dir, state, "raw_predictions_saved")

    n_timesteps, n_vertices = predictions.shape

    metadata = {
        "runner": "TRIBE v2 Local Runner",
        "mode": RUNNER_MODE,
        "model_id": MODEL_ID,
        "video": str(video.resolve()),
        "video_fingerprint": fingerprint,
        "video_duration_seconds": duration,
        "prediction_shape": [int(n_timesteps), int(n_vertices)],
        "hardware": hardware,
        "device_used": device,
        "runtime_tuning": tuning or state.get("runtime_tuning", {}),
        "resume": {
            "schema": RESUME_SCHEMA,
            "attempt": attempts,
            "state_file": "run_state.json",
            "feature_cache": str(feature_cache_dir),
            "prediction_parts": ".resume/prediction_parts",
        },
        "elapsed_seconds": time.time() - started,
        "note": (
            "These are model-predicted fMRI-like cortical responses for an "
            "average subject, not measurements of a specific person's brain."
        ),
    }
    _atomic_json(metadata_path, metadata)

    # Post-processing is intentionally regenerated on every launch. It is cheap
    # compared with V-JEPA2/TRIBE inference and ensures improvements to peak
    # detection, normalization or rendering are applied to already-cached runs.
    _emit(
        progress_cb,
        "Rebuilding the normalized AI package from cached cortical predictions…",
        0.84,
        "normalize",
    )
    from normalize import build_normalized_package

    normalized_path = build_normalized_package(
        predictions=predictions,
        timeline_csv=timeline_path,
        output_dir=run_dir,
        metadata=metadata,
    )
    state = _checkpoint(run_dir, state, "normalized")

    _emit(
        progress_cb,
        "Rebuilding cortical surfaces, key moments and the interactive report…",
        0.91,
        "visualize",
    )
    from visualize import create_report_assets

    report_path = create_report_assets(
        predictions=predictions,
        timeline_csv=timeline_path,
        output_dir=run_dir,
        metadata=metadata,
        normalized_path=normalized_path,
    )
    state = _checkpoint(run_dir, state, "visualized")

    elapsed = time.time() - started
    metadata["elapsed_seconds"] = elapsed
    metadata["normalized_output"] = str(normalized_path.name)
    metadata["report"] = str(report_path.name)
    _atomic_json(metadata_path, metadata)
    state = _checkpoint(run_dir, state, "done", elapsed_seconds=elapsed)

    _emit(
        progress_cb,
        "Analysis complete — raw brain data, normalized AI package and report are ready.",
        1.0,
        "done",
    )

    return RunResult(
        output_dir=run_dir,
        predictions_path=pred_path,
        timeline_csv=timeline_path,
        metadata_path=metadata_path,
        normalized_path=normalized_path,
        report_path=report_path,
        n_timesteps=n_timesteps,
        n_vertices=n_vertices,
        seconds_elapsed=elapsed,
    )
