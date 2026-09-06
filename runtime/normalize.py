from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = "1.1"


def _safe_float(value: float | np.floating | int) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return value


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(values))
    if not math.isfinite(scale) or scale < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - median) / scale, -8.0, 8.0)


def _load_timeline(timeline_csv: Path, n_timesteps: int) -> pd.DataFrame:
    df = pd.read_csv(timeline_csv)
    if len(df) != n_timesteps:
        df = df.iloc[:n_timesteps].copy()
    if "start_seconds" not in df:
        df["start_seconds"] = np.arange(len(df), dtype=float)
    if "duration_seconds" not in df:
        df["duration_seconds"] = 1.0
    return df


def _decode_label(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _destrieux_regions(predictions: np.ndarray) -> tuple[list[dict], list[dict], str | None]:
    """Aggregate fsaverage5 vertices into Destrieux cortical regions.

    Returns (region_summaries, per_time_region_rows, warning).
    """
    try:
        from nilearn import datasets

        atlas = datasets.fetch_atlas_surf_destrieux()
        left_map = np.asarray(atlas["map_left"], dtype=int)
        right_map = np.asarray(atlas["map_right"], dtype=int)
        labels = [_decode_label(x) for x in atlas["labels"]]
    except Exception as exc:
        return [], [], f"Destrieux atlas unavailable: {exc}"

    n_timesteps, n_vertices = predictions.shape
    half = n_vertices // 2
    if half * 2 != n_vertices:
        return [], [], f"Expected paired hemispheres, got {n_vertices} vertices."
    if left_map.size != half or right_map.size != half:
        return [], [], (
            "Destrieux/fsaverage5 vertex mismatch: "
            f"atlas=({left_map.size},{right_map.size}) model_half={half}"
        )

    abs_pred = np.abs(predictions.astype(np.float64, copy=False))
    region_series: list[tuple[str, str, np.ndarray, int]] = []

    for hemi, mapping, values in (
        ("left", left_map, abs_pred[:, :half]),
        ("right", right_map, abs_pred[:, half:]),
    ):
        for region_id in np.unique(mapping):
            region_id = int(region_id)
            if region_id <= 0:
                continue
            mask = mapping == region_id
            if not np.any(mask):
                continue
            name = labels[region_id] if region_id < len(labels) else f"region_{region_id}"
            series = values[:, mask].mean(axis=1)
            region_series.append((name, hemi, series, int(mask.sum())))

    if not region_series:
        return [], [], "Destrieux atlas loaded but no cortical regions were resolved."

    region_means = np.array([x[2].mean() for x in region_series], dtype=np.float64)
    region_rank_z = _robust_z(region_means)

    summaries: list[dict] = []
    for idx, (name, hemi, series, n_vertices_region) in enumerate(region_series):
        peak_idx = int(np.argmax(series))
        temporal_z = _robust_z(series)
        summaries.append(
            {
                "name": name,
                "hemisphere": hemi,
                "n_vertices": n_vertices_region,
                "mean_abs_response": _safe_float(series.mean()),
                "region_rank_z": _safe_float(region_rank_z[idx]),
                "peak_abs_response": _safe_float(series[peak_idx]),
                "peak_timestep": peak_idx,
                "peak_within_region_z": _safe_float(temporal_z[peak_idx]),
                "temporal_variability": _safe_float(series.std()),
            }
        )

    per_time: list[dict] = []
    for t in range(n_timesteps):
        ranked = sorted(
            (
                {
                    "name": name,
                    "hemisphere": hemi,
                    "abs_response": _safe_float(series[t]),
                    "within_region_z": _safe_float(_robust_z(series)[t]),
                }
                for name, hemi, series, _ in region_series
            ),
            key=lambda x: x["abs_response"],
            reverse=True,
        )
        per_time.append({"timestep": t, "top_regions": ranked[:6]})

    summaries.sort(key=lambda x: x["mean_abs_response"], reverse=True)
    return summaries, per_time, None


def _peak_events(
    global_z: np.ndarray,
    global_abs: np.ndarray,
    starts: np.ndarray,
    durations: np.ndarray,
    per_time_regions: list[dict],
) -> list[dict]:
    try:
        from scipy.signal import find_peaks

        min_distance = max(1, int(round(len(global_z) / 12)))
        peak_indices, props = find_peaks(global_z, prominence=0.65, distance=min_distance)
        prominences = props.get("prominences", np.zeros_like(peak_indices, dtype=float))
        candidates = list(zip(peak_indices.tolist(), prominences.tolist()))
    except Exception:
        candidates = []

    # scipy.find_peaks intentionally ignores boundary samples. In short ads the
    # final frame/CTA can be the strongest event, so explicitly consider both
    # endpoints when they rise meaningfully above their only neighbor.
    if len(global_z) >= 2:
        endpoint_candidates = []
        left_prominence = float(global_z[0] - global_z[1])
        right_prominence = float(global_z[-1] - global_z[-2])
        if global_z[0] > global_z[1] and left_prominence >= 0.65:
            endpoint_candidates.append((0, left_prominence))
        if global_z[-1] > global_z[-2] and right_prominence >= 0.65:
            endpoint_candidates.append((len(global_z) - 1, right_prominence))

        existing = {idx for idx, _ in candidates}
        for item in endpoint_candidates:
            if item[0] not in existing:
                candidates.append(item)

    if not candidates and len(global_z):
        candidates = [(int(i), float(global_z[i])) for i in np.argsort(global_z)[-5:]]

    candidates.sort(key=lambda x: (global_z[x[0]], x[1]), reverse=True)
    events: list[dict] = []
    for idx, prominence in candidates[:8]:
        top_regions = (
            per_time_regions[idx]["top_regions"][:4]
            if idx < len(per_time_regions)
            else []
        )
        events.append(
            {
                "timestep": int(idx),
                "start_seconds": _safe_float(starts[idx]),
                "duration_seconds": _safe_float(durations[idx]),
                "global_response_z": _safe_float(global_z[idx]),
                "mean_abs_response": _safe_float(global_abs[idx]),
                "prominence": _safe_float(prominence),
                "top_regions": top_regions,
            }
        )
    events.sort(key=lambda x: x["start_seconds"])
    return events


def _trough_events(
    global_z: np.ndarray,
    global_abs: np.ndarray,
    starts: np.ndarray,
    durations: np.ndarray,
    per_time_regions: list[dict],
) -> list[dict]:
    """Identify locally weak response windows, including clip boundaries."""
    try:
        from scipy.signal import find_peaks

        min_distance = max(1, int(round(len(global_z) / 12)))
        indices, props = find_peaks(-global_z, prominence=0.65, distance=min_distance)
        prominences = props.get("prominences", np.zeros_like(indices, dtype=float))
        candidates = list(zip(indices.tolist(), prominences.tolist()))
    except Exception:
        candidates = []

    if len(global_z) >= 2:
        left_prominence = float(global_z[1] - global_z[0])
        right_prominence = float(global_z[-2] - global_z[-1])
        existing = {idx for idx, _ in candidates}
        if global_z[0] < global_z[1] and left_prominence >= 0.65 and 0 not in existing:
            candidates.append((0, left_prominence))
        last = len(global_z) - 1
        if global_z[-1] < global_z[-2] and right_prominence >= 0.65 and last not in existing:
            candidates.append((last, right_prominence))

    if not candidates and len(global_z):
        candidates = [(int(i), float(-global_z[i])) for i in np.argsort(global_z)[:5]]

    candidates.sort(key=lambda x: (global_z[x[0]], -x[1]))
    events: list[dict] = []
    for idx, prominence in candidates[:8]:
        top_regions = (
            per_time_regions[idx]["top_regions"][:4]
            if idx < len(per_time_regions)
            else []
        )
        events.append(
            {
                "timestep": int(idx),
                "start_seconds": _safe_float(starts[idx]),
                "duration_seconds": _safe_float(durations[idx]),
                "global_response_z": _safe_float(global_z[idx]),
                "mean_abs_response": _safe_float(global_abs[idx]),
                "prominence": _safe_float(prominence),
                "top_regions": top_regions,
            }
        )
    events.sort(key=lambda x: x["start_seconds"])
    return events


def _creative_signature(
    global_abs: np.ndarray,
    global_z: np.ndarray,
    starts: np.ndarray,
    durations: np.ndarray,
    left_abs: np.ndarray,
    right_abs: np.ndarray,
    concentration: np.ndarray,
    peak_events: list[dict],
    trough_events: list[dict],
) -> dict:
    total_duration = float((starts + durations).max()) if len(starts) else 0.0
    early_mask = starts < min(3.0, max(total_duration, 0.0))
    late_cut = max(0.0, total_duration - 3.0)
    late_mask = starts >= late_cut

    sustained = float(np.mean(global_z >= 1.0)) if len(global_z) else 0.0
    positive = float(np.mean(global_z > 0.0)) if len(global_z) else 0.0
    change_rate = float(np.mean(np.abs(np.diff(global_z)))) if len(global_z) > 1 else 0.0
    denom = left_abs + right_abs + 1e-12
    hemispheric_balance = float(np.mean((right_abs - left_abs) / denom)) if len(denom) else 0.0
    peak_density = (len(peak_events) / total_duration * 10.0) if total_duration > 0 else 0.0
    trough_density = (len(trough_events) / total_duration * 10.0) if total_duration > 0 else 0.0

    def masked_mean(mask: np.ndarray) -> float:
        if not len(global_z) or not np.any(mask):
            return 0.0
        return float(np.mean(global_z[mask]))

    return {
        "normalization_scope": "within_clip",
        "mean_abs_response": _safe_float(global_abs.mean() if len(global_abs) else 0.0),
        "peak_global_response_z": _safe_float(global_z.max() if len(global_z) else 0.0),
        "trough_global_response_z": _safe_float(global_z.min() if len(global_z) else 0.0),
        "sustained_high_response_fraction": _safe_float(sustained),
        "above_median_response_fraction": _safe_float(positive),
        "peak_density_per_10_seconds": _safe_float(peak_density),
        "trough_density_per_10_seconds": _safe_float(trough_density),
        "temporal_change_rate": _safe_float(change_rate),
        "spatial_concentration_top10pct_share": _safe_float(
            concentration.mean() if len(concentration) else 0.0
        ),
        "hemispheric_balance_right_minus_left": _safe_float(hemispheric_balance),
        "early_0_3s_response_z_mean": _safe_float(masked_mean(early_mask)),
        "late_last_3s_response_z_mean": _safe_float(masked_mean(late_mask)),
        "important_note": (
            "These are descriptive model-output features, not validated estimates "
            "of CTR, CVR, CPA, sales, attention, memory, emotion, or persuasion."
        ),
    }


def build_normalized_package(
    predictions: np.ndarray,
    timeline_csv: str | Path,
    output_dir: str | Path,
    metadata: dict,
) -> Path:
    """Create a compact, ChatGPT-readable representation of TRIBE v2 output."""
    output_dir = Path(output_dir)
    timeline_csv = Path(timeline_csv)
    predictions = np.asarray(predictions, dtype=np.float32)
    n_timesteps, n_vertices = predictions.shape

    timeline = _load_timeline(timeline_csv, n_timesteps)
    starts = timeline["start_seconds"].to_numpy(dtype=float)
    durations = timeline["duration_seconds"].to_numpy(dtype=float)

    abs_pred = np.abs(predictions.astype(np.float64, copy=False))
    global_abs = abs_pred.mean(axis=1)
    global_z = _robust_z(global_abs)
    spatial_std = predictions.astype(np.float64, copy=False).std(axis=1)

    half = n_vertices // 2
    left_abs = abs_pred[:, :half].mean(axis=1)
    right_abs = abs_pred[:, half:].mean(axis=1)

    # Fraction of total absolute response carried by the top 10% of vertices.
    k = max(1, int(round(n_vertices * 0.10)))
    partitioned = np.partition(abs_pred, n_vertices - k, axis=1)[:, -k:]
    concentration = partitioned.sum(axis=1) / (abs_pred.sum(axis=1) + 1e-12)

    regions, per_time_regions, atlas_warning = _destrieux_regions(predictions)
    peak_events = _peak_events(
        global_z,
        global_abs,
        starts,
        durations,
        per_time_regions,
    )
    trough_events = _trough_events(
        global_z,
        global_abs,
        starts,
        durations,
        per_time_regions,
    )
    signature = _creative_signature(
        global_abs,
        global_z,
        starts,
        durations,
        left_abs,
        right_abs,
        concentration,
        peak_events,
        trough_events,
    )

    windows: list[dict] = []
    for idx in range(n_timesteps):
        windows.append(
            {
                "timestep": idx,
                "start_seconds": _safe_float(starts[idx]),
                "duration_seconds": _safe_float(durations[idx]),
                "global_mean_abs_response": _safe_float(global_abs[idx]),
                "global_response_z_within_clip": _safe_float(global_z[idx]),
                "spatial_std": _safe_float(spatial_std[idx]),
                "left_mean_abs_response": _safe_float(left_abs[idx]),
                "right_mean_abs_response": _safe_float(right_abs[idx]),
                "top10pct_vertex_response_share": _safe_float(concentration[idx]),
                "top_regions": (
                    per_time_regions[idx]["top_regions"]
                    if idx < len(per_time_regions)
                    else []
                ),
            }
        )

    payload = {
        "schema": "tribev2_chatgpt_normalized",
        "schema_version": SCHEMA_VERSION,
        "source": {
            "model": metadata.get("model_id"),
            "mode": metadata.get("mode"),
            "prediction_shape": [int(n_timesteps), int(n_vertices)],
            "video_duration_seconds": metadata.get("video_duration_seconds"),
            "device_used": metadata.get("device_used"),
        },
        "meaning": {
            "what_it_is": (
                "A compact summary of TRIBE v2 model-predicted fMRI-like cortical "
                "responses for an average subject viewing this video."
            ),
            "what_it_is_not": (
                "It is not a measurement of a real viewer and does not directly "
                "measure attention, emotion, memory, persuasion, CTR, CVR, CPA or sales."
            ),
            "normalization": (
                "Temporal z values are robust within-clip z-scores using median/MAD. "
                "Region rank z-scores compare mean absolute response across cortical "
                "regions inside this clip. Raw mean-absolute values are retained."
            ),
        },
        "creative_signature": signature,
        "peak_events": peak_events,
        "trough_events": trough_events,
        "top_regions_overall": regions[:30],
        "time_windows": windows,
        "atlas": {
            "name": "Destrieux 2010 surface atlas / fsaverage5" if regions else None,
            "warning": atlas_warning,
        },
        "recommended_interpretation": {
            "safe_uses": [
                "Identify relative peaks and drops over the creative timeline.",
                "Describe whether predicted response is sustained or highly transient.",
                "Compare left/right balance and spatial concentration descriptively.",
                "Identify cortical regions with relatively high model-predicted response.",
                "Generate creative hypotheses to test in A/B experiments.",
            ],
            "requires_calibration_data": [
                "CTR or thumb-stop prediction",
                "hold-rate prediction",
                "conversion-rate prediction",
                "CPA/ROAS prediction",
                "memory or persuasion claims",
            ],
        },
    }

    normalized_path = output_dir / "chatgpt_brain_summary.json"
    normalized_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    signature_path = output_dir / "creative_signature.json"
    signature_path.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    roi_csv = output_dir / "roi_summary.csv"
    if regions:
        pd.DataFrame(regions).to_csv(roi_csv, index=False)
    else:
        pd.DataFrame(
            columns=[
                "name",
                "hemisphere",
                "n_vertices",
                "mean_abs_response",
                "region_rank_z",
                "peak_abs_response",
                "peak_timestep",
                "peak_within_region_z",
                "temporal_variability",
            ]
        ).to_csv(roi_csv, index=False)

    windows_csv = output_dir / "normalized_timeline.csv"
    pd.DataFrame(
        [
            {
                "timestep": w["timestep"],
                "start_seconds": w["start_seconds"],
                "duration_seconds": w["duration_seconds"],
                "global_mean_abs_response": w["global_mean_abs_response"],
                "global_response_z_within_clip": w["global_response_z_within_clip"],
                "spatial_std": w["spatial_std"],
                "left_mean_abs_response": w["left_mean_abs_response"],
                "right_mean_abs_response": w["right_mean_abs_response"],
                "top10pct_vertex_response_share": w["top10pct_vertex_response_share"],
            }
            for w in windows
        ]
    ).to_csv(windows_csv, index=False)

    prompt = """Use o arquivo chatgpt_brain_summary.json como a fonte neural desta analise.

Objetivo:
Transformar a previsao cortical do TRIBE v2 em insights praticos para quem cria anuncios.

Regras importantes:
1. Trate os dados como RESPOSTAS CORTICAIS PREVISTAS PELO MODELO para um sujeito medio, nao como medicao de um cerebro real.
2. Nao invente CTR, CVR, CPA, ROAS, vendas, memoria, emocao ou atencao como fatos observados.
3. Se estimar alguma metrica de marketing, chame explicitamente de hipotese/proxy nao calibrado e explique que uma previsao numerica confiavel exige dados historicos de campanhas ligados a estas assinaturas neurais.
4. Use principalmente creative_signature, peak_events, top_regions_overall e time_windows.
5. Diferencie claramente: dado observado no modelo -> interpretacao neurofuncional plausivel -> hipotese criativa a testar.

Entregue:
- Resumo executivo em linguagem simples.
- Timeline do criativo: onde a resposta sobe, cai e muda de padrao.
- 3 a 7 momentos mais importantes, com timestamps.
- Regioes corticais dominantes e interpretacao cautelosa do que isso pode sugerir.
- Pontos fortes provaveis do criativo.
- Pontos de melhoria e trechos que merecem A/B test.
- Hipoteses de hook, retencao e intensidade de resposta, sem apresenta-las como metricas reais.
- Quais dados reais de campanha deveriam ser anexados para calibrar previsoes futuras.
"""
    (output_dir / "CHATGPT_ANALYSIS_PROMPT.txt").write_text(prompt, encoding="utf-8")

    return normalized_path
