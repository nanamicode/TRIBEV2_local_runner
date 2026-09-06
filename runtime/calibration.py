from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CALIBRATION_SCHEMA = "1.0"
MIN_TRAIN_SAMPLES = 12
RECOMMENDED_SAMPLES = 30

TARGET_FIELDS = {
    "hook_rate_3s_pct": "Hook / 3s view rate (%)",
    "hold_rate_15s_pct": "Hold rate 15s (%)",
    "video_25_pct": "Video viewed 25% (%)",
    "video_50_pct": "Video viewed 50% (%)",
    "video_75_pct": "Video viewed 75% (%)",
    "video_95_pct": "Video viewed 95% (%)",
    "avg_watch_time_seconds": "Average watch time (s)",
    "ctr_pct": "CTR (%)",
    "cvr_pct": "CVR (%)",
    "cpa": "CPA",
    "roas": "ROAS",
}

CONTEXT_FIELDS = {
    "platform": "Platform",
    "placement": "Placement",
    "campaign_id": "Campaign ID",
    "ad_id": "Ad ID",
    "spend": "Spend",
    "impressions": "Impressions",
}


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("%", "").replace("R$", "").strip()
    if "," in text and "." in text:
        # Brazilian formatting 1.234,56.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    text = re.sub(r"[^0-9eE+\-.]", "", text)
    try:
        number = float(text)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def _slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")[:72]


def _interp(values: np.ndarray, n: int = 16) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(n, dtype=float)
    if values.size == 1:
        return np.repeat(values[0], n)
    old_x = np.linspace(0.0, 1.0, values.size)
    new_x = np.linspace(0.0, 1.0, n)
    return np.interp(new_x, old_x, values)


def extract_calibration_features(run_dir: str | Path) -> dict[str, float]:
    """Build a fixed, deterministic feature vector from one completed TRIBE run."""
    run_dir = Path(run_dir)
    summary = _read_json(run_dir / "chatgpt_brain_summary.json")
    signature = _read_json(run_dir / "creative_signature.json")
    metadata = _read_json(run_dir / "run_metadata.json")

    features: dict[str, float] = {}

    for key, value in signature.items():
        number = _safe_number(value)
        if number is not None:
            features[f"signature__{_slug(key)}"] = number

    source = summary.get("source", {})
    duration = _safe_number(source.get("video_duration_seconds"))
    if duration is None:
        duration = _safe_number(metadata.get("video_duration_seconds"))
    if duration is not None:
        features["video__duration_seconds"] = duration

    windows = summary.get("time_windows") or []
    if windows:
        z = np.array(
            [_safe_number(x.get("global_response_z_within_clip")) or 0.0 for x in windows],
            dtype=float,
        )
        abs_response = np.array(
            [_safe_number(x.get("global_mean_abs_response")) or 0.0 for x in windows],
            dtype=float,
        )
        concentration = np.array(
            [_safe_number(x.get("top10pct_vertex_response_share")) or 0.0 for x in windows],
            dtype=float,
        )
        left = np.array(
            [_safe_number(x.get("left_mean_abs_response")) or 0.0 for x in windows],
            dtype=float,
        )
        right = np.array(
            [_safe_number(x.get("right_mean_abs_response")) or 0.0 for x in windows],
            dtype=float,
        )

        for name, series in (
            ("z", z),
            ("abs", abs_response),
            ("concentration", concentration),
        ):
            resampled = _interp(series, 16)
            for idx, value in enumerate(resampled):
                features[f"timeline__{name}_{idx:02d}"] = float(value)

        features["timeline__z_min"] = float(np.min(z))
        features["timeline__z_max"] = float(np.max(z))
        features["timeline__z_mean"] = float(np.mean(z))
        features["timeline__z_std"] = float(np.std(z))
        features["timeline__z_positive_area"] = float(np.mean(np.clip(z, 0, None)))
        features["timeline__z_negative_area"] = float(np.mean(np.clip(-z, 0, None)))
        features["timeline__peak_position"] = float(np.argmax(z) / max(1, len(z) - 1))
        features["timeline__trough_position"] = float(np.argmin(z) / max(1, len(z) - 1))
        features["timeline__left_mean"] = float(np.mean(left))
        features["timeline__right_mean"] = float(np.mean(right))

    roi_path = run_dir / "roi_summary.csv"
    if roi_path.exists():
        try:
            roi = pd.read_csv(roi_path)
            for _, row in roi.iterrows():
                region = _slug(str(row.get("name", "region")))
                hemi = _slug(str(row.get("hemisphere", "unknown")))
                prefix = f"roi__{hemi}__{region}"
                for field in (
                    "mean_abs_response",
                    "temporal_variability",
                    "peak_within_region_z",
                ):
                    number = _safe_number(row.get(field))
                    if number is not None:
                        features[f"{prefix}__{field}"] = number
        except Exception:
            pass

    # Stable finite values only.
    return {
        key: float(value)
        for key, value in sorted(features.items())
        if math.isfinite(float(value))
    }


def _calibration_root(output_root: str | Path) -> Path:
    root = Path(output_root) / ".calibration"
    root.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(parents=True, exist_ok=True)
    return root


def prepare_calibration_bundle(run_dir: str | Path, output_root: str | Path) -> dict:
    """Create feature snapshot and, when models exist, local KPI hypotheses."""
    run_dir = Path(run_dir)
    features = extract_calibration_features(run_dir)
    metadata = _read_json(run_dir / "run_metadata.json")

    feature_payload = {
        "schema": "tribev2_calibration_features",
        "schema_version": CALIBRATION_SCHEMA,
        "video_fingerprint": metadata.get("video_fingerprint"),
        "video": metadata.get("video"),
        "feature_count": len(features),
        "features": features,
        "note": (
            "These features are derived from TRIBE v2 model output. Campaign KPIs "
            "must be learned from real labeled campaign outcomes."
        ),
    }
    (run_dir / "calibration_features.json").write_text(
        json.dumps(feature_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    template = {
        "schema": "tribev2_campaign_metrics",
        "schema_version": CALIBRATION_SCHEMA,
        "video_fingerprint": metadata.get("video_fingerprint"),
        "targets": {key: None for key in TARGET_FIELDS},
        "context": {key: None for key in CONTEXT_FIELDS},
        "instructions": (
            "Fill only metrics measured for this exact creative. Percent fields use "
            "percentage points (e.g. 2.4 means 2.4%). Never fill guessed values."
        ),
    }
    template_path = run_dir / "campaign_metrics_template.json"
    if not template_path.exists():
        template_path.write_text(
            json.dumps(template, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    prediction = predict_calibrated_metrics(run_dir, output_root)
    return {
        "features_path": str(run_dir / "calibration_features.json"),
        "template_path": str(template_path),
        "prediction_path": prediction.get("path"),
        "prediction_status": prediction.get("status"),
    }


def save_campaign_metrics(
    run_dir: str | Path,
    output_root: str | Path,
    targets: dict[str, Any],
    context: dict[str, Any] | None = None,
) -> dict:
    """Persist real campaign outcomes and upsert the calibration dataset."""
    run_dir = Path(run_dir)
    output_root = Path(output_root)
    context = context or {}
    metadata = _read_json(run_dir / "run_metadata.json")
    features = extract_calibration_features(run_dir)

    clean_targets = {
        key: _safe_number(targets.get(key))
        for key in TARGET_FIELDS
    }
    if not any(value is not None for value in clean_targets.values()):
        raise ValueError(
            "Enter at least one measured campaign outcome. Blank/guessed labels are not saved."
        )

    clean_context: dict[str, Any] = {}
    for key in CONTEXT_FIELDS:
        if key in ("spend", "impressions"):
            clean_context[key] = _safe_number(context.get(key))
        else:
            raw = context.get(key)
            clean_context[key] = str(raw).strip() if raw not in (None, "") else None

    payload = {
        "schema": "tribev2_campaign_metrics",
        "schema_version": CALIBRATION_SCHEMA,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "video_fingerprint": metadata.get("video_fingerprint"),
        "video": metadata.get("video"),
        "targets": clean_targets,
        "context": clean_context,
    }
    (run_dir / "campaign_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    row: dict[str, Any] = {
        "video_fingerprint": metadata.get("video_fingerprint"),
        "video": metadata.get("video"),
        "run_dir": str(run_dir),
        "saved_at": payload["saved_at"],
    }
    row.update({f"feature__{k}": v for k, v in features.items()})
    row.update({f"target__{k}": v for k, v in clean_targets.items()})
    row.update({f"context__{k}": v for k, v in clean_context.items()})

    cal_root = _calibration_root(output_root)
    dataset_path = cal_root / "calibration_dataset.csv"
    new_df = pd.DataFrame([row])
    if dataset_path.exists():
        try:
            old = pd.read_csv(dataset_path)
            fingerprint = row.get("video_fingerprint")
            if "video_fingerprint" in old.columns and fingerprint:
                old = old[old["video_fingerprint"].astype(str) != str(fingerprint)]
            combined = pd.concat([old, new_df], ignore_index=True, sort=False)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_csv(dataset_path, index=False)

    training = train_calibration_models(output_root)
    prediction = predict_calibrated_metrics(run_dir, output_root)
    return {
        "dataset_path": str(dataset_path),
        "records": int(len(combined)),
        "training": training,
        "prediction": prediction,
    }


def _quality_label(n: int) -> str:
    if n < MIN_TRAIN_SAMPLES:
        return "insufficient"
    if n < RECOMMENDED_SAMPLES:
        return "experimental"
    if n < 100:
        return "emerging"
    return "maturing"


def train_calibration_models(output_root: str | Path) -> dict:
    """Fit one small Ridge model per KPI using only labeled real outcomes."""
    cal_root = _calibration_root(output_root)
    dataset_path = cal_root / "calibration_dataset.csv"
    manifest_path = cal_root / "calibration_manifest.json"

    if not dataset_path.exists():
        manifest = {
            "schema_version": CALIBRATION_SCHEMA,
            "records": 0,
            "minimum_samples": MIN_TRAIN_SAMPLES,
            "models": {},
            "status": "no_dataset",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    data = pd.read_csv(dataset_path)
    feature_cols = [c for c in data.columns if c.startswith("feature__")]
    model_info: dict[str, dict] = {}

    try:
        from joblib import dump
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        manifest = {
            "schema_version": CALIBRATION_SCHEMA,
            "records": int(len(data)),
            "minimum_samples": MIN_TRAIN_SAMPLES,
            "models": {},
            "status": "sklearn_unavailable",
            "error": str(exc),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    if not feature_cols:
        manifest = {
            "schema_version": CALIBRATION_SCHEMA,
            "records": int(len(data)),
            "minimum_samples": MIN_TRAIN_SAMPLES,
            "models": {},
            "status": "no_features",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    for target in TARGET_FIELDS:
        target_col = f"target__{target}"
        if target_col not in data:
            continue

        valid = data[target_col].notna()
        subset = data.loc[valid]
        n = int(len(subset))
        info = {
            "samples": n,
            "quality": _quality_label(n),
            "label": TARGET_FIELDS[target],
            "trained": False,
        }
        if n < MIN_TRAIN_SAMPLES:
            model_info[target] = info
            continue

        X = subset[feature_cols].apply(pd.to_numeric, errors="coerce")
        y = pd.to_numeric(subset[target_col], errors="coerce").to_numpy(dtype=float)
        finite_y = np.isfinite(y)
        X = X.loc[finite_y]
        y = y[finite_y]
        n = int(len(y))
        info["samples"] = n
        info["quality"] = _quality_label(n)
        if n < MIN_TRAIN_SAMPLES:
            model_info[target] = info
            continue

        # Drop columns that have no usable data in this training subset.
        usable_cols = [c for c in X.columns if X[c].notna().any()]
        X = X[usable_cols]

        estimator = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "ridge",
                    RidgeCV(alphas=np.logspace(-3, 4, 40)),
                ),
            ]
        )

        folds = min(5, n)
        cv = KFold(n_splits=folds, shuffle=True, random_state=42)
        cv_pred = cross_val_predict(estimator, X, y, cv=cv)
        mae = float(mean_absolute_error(y, cv_pred))
        r2 = float(r2_score(y, cv_pred)) if n >= 3 else float("nan")
        residual = np.abs(y - cv_pred)
        q80 = float(np.quantile(residual, 0.80))
        q90 = float(np.quantile(residual, 0.90))

        estimator.fit(X, y)
        model_path = cal_root / "models" / f"{target}.joblib"
        dump(
            {
                "model": estimator,
                "feature_columns": usable_cols,
                "target": target,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            },
            model_path,
        )

        info.update(
            {
                "trained": True,
                "model_path": str(model_path),
                "cv_folds": folds,
                "cv_mae": mae,
                "cv_r2": r2 if math.isfinite(r2) else None,
                "cv_abs_error_q80": q80,
                "cv_abs_error_q90": q90,
            }
        )
        model_info[target] = info

    trained = sum(1 for x in model_info.values() if x.get("trained"))
    manifest = {
        "schema_version": CALIBRATION_SCHEMA,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "records": int(len(data)),
        "minimum_samples": MIN_TRAIN_SAMPLES,
        "recommended_samples": RECOMMENDED_SAMPLES,
        "trained_model_count": trained,
        "status": "ready" if trained else "collecting_labels",
        "models": model_info,
        "warning": (
            "Calibrated KPI outputs are empirical hypotheses learned from your labeled "
            "campaign history. They are not direct measurements produced by TRIBE v2."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def predict_calibrated_metrics(run_dir: str | Path, output_root: str | Path) -> dict:
    run_dir = Path(run_dir)
    cal_root = _calibration_root(output_root)
    manifest_path = cal_root / "calibration_manifest.json"
    output_path = run_dir / "calibrated_predictions.json"

    if not manifest_path.exists():
        return {"status": "not_trained", "path": None}

    manifest = _read_json(manifest_path)
    models = manifest.get("models") or {}
    features = extract_calibration_features(run_dir)
    predictions: dict[str, dict] = {}

    try:
        from joblib import load
    except Exception:
        return {"status": "joblib_unavailable", "path": None}

    for target, info in models.items():
        if not info.get("trained"):
            continue
        model_path = Path(info.get("model_path", ""))
        if not model_path.exists():
            continue
        try:
            artifact = load(model_path)
            columns = artifact["feature_columns"]
            row = {
                col: features.get(col.removeprefix("feature__"))
                for col in columns
            }
            X = pd.DataFrame([row], columns=columns)
            point = float(artifact["model"].predict(X)[0])
            q80 = _safe_number(info.get("cv_abs_error_q80"))
            q90 = _safe_number(info.get("cv_abs_error_q90"))
            predictions[target] = {
                "label": TARGET_FIELDS.get(target, target),
                "prediction": point,
                "samples": info.get("samples"),
                "quality": info.get("quality"),
                "cv_mae": info.get("cv_mae"),
                "interval_80": [point - q80, point + q80] if q80 is not None else None,
                "interval_90": [point - q90, point + q90] if q90 is not None else None,
            }
        except Exception as exc:
            predictions[target] = {"error": str(exc)}

    payload = {
        "schema": "tribev2_calibrated_predictions",
        "schema_version": CALIBRATION_SCHEMA,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_records": manifest.get("records", 0),
        "predictions": predictions,
        "status": "ready" if predictions else "not_enough_labeled_data",
        "interpretation": (
            "These are downstream empirical predictions calibrated against real campaign "
            "outcomes. They are not direct TRIBE v2 measurements and should be validated "
            "on held-out campaigns before operational use."
        ),
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"status": payload["status"], "path": str(output_path), "payload": payload}
