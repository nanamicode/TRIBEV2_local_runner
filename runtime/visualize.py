from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

# All rendering happens from the worker thread of the desktop app. Force a
# non-interactive Matplotlib backend so Windows/Tk does not try to create GUI
# figures outside Tkinter's main thread. This removes the "Starting a
# Matplotlib GUI outside of the main thread" failure mode while still allowing
# figures to be saved normally to PNG.
import matplotlib
matplotlib.use("Agg", force=True)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write through pathlib instead of letting plotting backends open Windows paths."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _save_figure_png(figure, path: Path, dpi: int | None = None) -> Path:
    """Render a Matplotlib/Nilearn figure in memory, then atomically write the PNG.

    This avoids backend-specific Windows path handling. It is especially useful
    for cached run directories whose names come from creative filenames.
    """
    from io import BytesIO

    buffer = BytesIO()
    kwargs = {"format": "png"}
    if dpi is not None:
        kwargs["dpi"] = dpi
    figure.savefig(buffer, **kwargs)
    _write_bytes_atomic(path, buffer.getvalue())
    return path


def _existing_brain_maps(output_dir: Path) -> list[Path]:
    candidates = [
        output_dir / "brain_left_lateral.png",
        output_dir / "brain_right_lateral.png",
    ]
    return candidates if all(p.exists() and p.stat().st_size > 0 for p in candidates) else []


def _save_timeline(
    timeline_csv: Path,
    output_dir: Path,
    normalized: dict | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    df = pd.read_csv(timeline_csv)
    out = output_dir / "activation_timeline.png"

    if normalized and normalized.get("time_windows"):
        windows = normalized["time_windows"]
        x = [float(w["start_seconds"]) for w in windows]
        y = [float(w["global_response_z_within_clip"]) for w in windows]
        ylabel = "Robust response z-score (within clip)"
        title = "TRIBE v2 — relative cortical response over time"
    else:
        x = df["start_seconds"]
        y = df["mean_abs"]
        ylabel = "Mean absolute predicted response"
        title = "TRIBE v2 — predicted cortical activity over time"

    fig = plt.figure(figsize=(12, 4.8), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(x, y, linewidth=2)
    ax.axhline(0, linewidth=1, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("Video time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)

    if normalized:
        for event in normalized.get("peak_events", [])[:8]:
            t = float(event["start_seconds"])
            z = float(event["global_response_z"])
            ax.scatter([t], [z], s=28)
            ax.annotate(
                f"{t:.1f}s",
                (t, z),
                xytext=(4, 7),
                textcoords="offset points",
                fontsize=8,
            )

    fig.tight_layout()
    _save_figure_png(fig, out)
    plt.close(fig)
    return out


def _load_surface_assets():
    from nilearn import datasets

    return datasets.fetch_surf_fsaverage(mesh="fsaverage5")


def _save_brain_maps(predictions: np.ndarray, output_dir: Path) -> list[Path]:
    from nilearn import plotting
    import matplotlib.pyplot as plt

    aggregate = np.mean(np.abs(predictions), axis=0)
    n = aggregate.size
    half = n // 2
    if half * 2 != n:
        raise RuntimeError(f"Expected paired left/right cortical vertices, got {n}")

    fs = _load_surface_assets()
    outputs = []
    specs = [
        ("left", fs.infl_left, fs.sulc_left, aggregate[:half]),
        ("right", fs.infl_right, fs.sulc_right, aggregate[half:]),
    ]
    vmax = float(np.percentile(aggregate, 99.0)) if aggregate.size else None
    for hemi, mesh, sulc, values in specs:
        out = output_dir / f"brain_{hemi}_lateral.png"
        display = plotting.plot_surf_stat_map(
            mesh,
            values,
            hemi=hemi,
            view="lateral",
            bg_map=sulc,
            colorbar=True,
            title=f"TRIBE v2 — {hemi} hemisphere / whole creative",
            cmap="inferno",
            vmax=vmax,
            symmetric_cbar=False,
        )
        _save_figure_png(display.figure, out, dpi=170)
        plt.close(display.figure)
        outputs.append(out)
    return outputs


def _save_interactive_brains(predictions: np.ndarray, output_dir: Path) -> list[Path]:
    from nilearn import plotting

    aggregate = np.mean(np.abs(predictions), axis=0)
    n = aggregate.size
    half = n // 2
    if half * 2 != n:
        return []

    fs = _load_surface_assets()
    outputs: list[Path] = []
    for hemi, mesh, sulc, values in (
        ("left", fs.infl_left, fs.sulc_left, aggregate[:half]),
        ("right", fs.infl_right, fs.sulc_right, aggregate[half:]),
    ):
        out = output_dir / f"brain_3d_{hemi}.html"
        view = plotting.view_surf(
            mesh,
            values,
            bg_map=sulc,
            cmap="inferno",
            symmetric_cmap=False,
            title=f"TRIBE v2 — interactive {hemi} hemisphere",
        )
        view.save_as_html(str(out))
        outputs.append(out)
    return outputs


def _save_peak_maps(
    predictions: np.ndarray,
    normalized: dict | None,
    output_dir: Path,
    max_events: int = 4,
) -> list[dict]:
    if not normalized:
        return []

    events = list(normalized.get("peak_events", []))
    if not events:
        return []

    # Choose the strongest events, then present them chronologically.
    strongest = sorted(
        events,
        key=lambda x: float(x.get("global_response_z", 0.0)),
        reverse=True,
    )[:max_events]
    strongest.sort(key=lambda x: float(x.get("start_seconds", 0.0)))

    try:
        from nilearn import plotting
        import matplotlib.pyplot as plt

        fs = _load_surface_assets()
    except Exception:
        return []

    half = predictions.shape[1] // 2
    results: list[dict] = []
    for rank, event in enumerate(strongest, start=1):
        idx = int(event.get("timestep", 0))
        if idx < 0 or idx >= len(predictions):
            continue
        values = np.abs(predictions[idx])
        vmax = float(np.percentile(values, 99.0))
        item = {
            "rank": rank,
            "timestep": idx,
            "start_seconds": float(event.get("start_seconds", idx)),
            "global_response_z": float(event.get("global_response_z", 0.0)),
            "images": [],
            "top_regions": event.get("top_regions", []),
        }
        for hemi, mesh, sulc, hemi_values in (
            ("left", fs.infl_left, fs.sulc_left, values[:half]),
            ("right", fs.infl_right, fs.sulc_right, values[half:]),
        ):
            out = output_dir / f"brain_peak_{rank:02d}_{hemi}.png"
            display = plotting.plot_surf_stat_map(
                mesh,
                hemi_values,
                hemi=hemi,
                view="lateral",
                bg_map=sulc,
                colorbar=True,
                title=f"{item['start_seconds']:.1f}s — {hemi}",
                cmap="inferno",
                vmax=vmax,
                symmetric_cbar=False,
            )
            _save_figure_png(display.figure, out, dpi=150)
            plt.close(display.figure)
            item["images"].append(out.name)
        results.append(item)
    return results


def _save_trough_maps(
    predictions: np.ndarray,
    normalized: dict | None,
    output_dir: Path,
    max_events: int = 4,
) -> list[dict]:
    if not normalized:
        return []

    events = list(normalized.get("trough_events", []))
    if not events:
        return []

    weakest = sorted(
        events,
        key=lambda x: float(x.get("global_response_z", 0.0)),
    )[:max_events]
    weakest.sort(key=lambda x: float(x.get("start_seconds", 0.0)))

    try:
        from nilearn import plotting
        import matplotlib.pyplot as plt
        fs = _load_surface_assets()
    except Exception:
        return []

    half = predictions.shape[1] // 2
    results: list[dict] = []
    for rank, event in enumerate(weakest, start=1):
        idx = int(event.get("timestep", 0))
        if idx < 0 or idx >= len(predictions):
            continue
        values = np.abs(predictions[idx])
        vmax = float(np.percentile(values, 99.0))
        item = {
            "rank": rank,
            "timestep": idx,
            "start_seconds": float(event.get("start_seconds", idx)),
            "global_response_z": float(event.get("global_response_z", 0.0)),
            "images": [],
            "top_regions": event.get("top_regions", []),
        }
        for hemi, mesh, sulc, hemi_values in (
            ("left", fs.infl_left, fs.sulc_left, values[:half]),
            ("right", fs.infl_right, fs.sulc_right, values[half:]),
        ):
            out = output_dir / f"brain_trough_{rank:02d}_{hemi}.png"
            display = plotting.plot_surf_stat_map(
                mesh,
                hemi_values,
                hemi=hemi,
                view="lateral",
                bg_map=sulc,
                colorbar=True,
                title=f"Weak window {item['start_seconds']:.1f}s — {hemi}",
                cmap="inferno",
                vmax=vmax,
                symmetric_cbar=False,
            )
            display.savefig(out, dpi=150)
            plt.close(display.figure)
            item["images"].append(out.name)
        results.append(item)
    return results


def _save_moment_frames(
    metadata: dict,
    normalized: dict | None,
    output_dir: Path,
    max_per_kind: int = 4,
) -> dict[tuple[str, int], str]:
    if not normalized:
        return {}

    video_path = Path(str(metadata.get("video") or ""))
    if not video_path.is_file():
        return {}

    events: list[tuple[str, dict]] = []
    peaks = sorted(
        normalized.get("peak_events", []),
        key=lambda x: float(x.get("global_response_z", 0.0)),
        reverse=True,
    )[:max_per_kind]
    troughs = sorted(
        normalized.get("trough_events", []),
        key=lambda x: float(x.get("global_response_z", 0.0)),
    )[:max_per_kind]
    events.extend(("peak", x) for x in peaks)
    events.extend(("trough", x) for x in troughs)

    if not events:
        return {}

    try:
        from moviepy import VideoFileClip
        from PIL import Image
    except Exception:
        return {}

    outputs: dict[tuple[str, int], str] = {}
    try:
        with VideoFileClip(str(video_path)) as clip:
            duration = float(clip.duration or 0.0)
            for kind, event in events:
                timestep = int(event.get("timestep", 0))
                start = float(event.get("start_seconds", 0.0))
                span = float(event.get("duration_seconds", 1.0))
                t = start + min(0.5, max(0.0, span / 2.0))
                if duration > 0:
                    t = min(max(0.0, t), max(0.0, duration - 1e-3))
                frame = np.asarray(clip.get_frame(t))
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                out = output_dir / f"creative_{kind}_{timestep:03d}.jpg"
                Image.fromarray(frame).save(out, quality=88, optimize=True)
                outputs[(kind, timestep)] = out.name
    except Exception:
        return outputs

    return outputs


def _fallback_brain_vector(predictions: np.ndarray, output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    aggregate = np.mean(np.abs(predictions), axis=0)
    out = output_dir / "cortical_vertices_fallback.png"
    fig = plt.figure(figsize=(12, 4), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(aggregate)
    ax.set_title("TRIBE v2 — aggregate cortical vertex response")
    ax.set_xlabel("fsaverage5 vertex")
    ax.set_ylabel("Mean absolute response")
    fig.tight_layout()
    _save_figure_png(fig, out)
    plt.close(fig)
    return [out]


def _calibration_card(output_dir: Path) -> str:
    prediction_path = output_dir / "calibrated_predictions.json"
    metrics_path = output_dir / "campaign_metrics.json"
    features_path = output_dir / "calibration_features.json"

    if prediction_path.exists():
        try:
            payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}

    predictions = payload.get("predictions") or {}
    rows = []
    for key, item in predictions.items():
        if not isinstance(item, dict) or "prediction" not in item:
            continue
        label = html.escape(str(item.get("label") or key))
        value = float(item.get("prediction"))
        quality = html.escape(str(item.get("quality") or "unknown"))
        samples = item.get("samples")
        interval = item.get("interval_80")
        interval_text = ""
        if isinstance(interval, list) and len(interval) == 2:
            interval_text = f" • 80% CV-residual band {float(interval[0]):.2f}–{float(interval[1]):.2f}"
        rows.append(
            f'<div class="metric"><span>{label}</span><strong>{value:.2f}</strong>'
            f'<small>{quality} • n={samples}{html.escape(interval_text)}</small></div>'
        )

    if rows:
        body = (
            '<div class="callout"><p><strong>Empirical calibration is active.</strong> '
            'These KPI values come from a downstream model fitted to real campaign outcomes, '
            'not directly from TRIBE v2.</p></div>'
            f'<div class="metrics">{"".join(rows)}</div>'
        )
    else:
        body = (
            '<div class="callout"><p><strong>Calibration dataset is still collecting labels.</strong> '
            'Add measured campaign outcomes for completed creatives. The runner will only train '
            'local KPI models after the minimum sample count is reached.</p></div>'
        )

    links = []
    if features_path.exists():
        links.append('<a href="calibration_features.json">calibration_features.json</a>')
    if metrics_path.exists():
        links.append('<a href="campaign_metrics.json">campaign_metrics.json</a>')
    if prediction_path.exists():
        links.append('<a href="calibrated_predictions.json">calibrated_predictions.json</a>')
    links.append('<a href="campaign_metrics_template.json">campaign_metrics_template.json</a>')

    return f"""
    <div class="card">
      <h2>Campaign calibration layer</h2>
      {body}
      <div class="download-grid">{"".join(links)}</div>
      <p class="hint">Context such as spend, placement and IDs is stored for auditability; pre-launch neural models are trained only from TRIBE-derived features to avoid campaign-leakage shortcuts.</p>
    </div>
    """


def _signature_cards(normalized: dict | None) -> str:
    if not normalized:
        return ""
    signature = normalized.get("creative_signature", {})
    specs = [
        ("Peak neural response", signature.get("peak_global_response_z"), "z"),
        (
            "Sustained high response",
            100.0 * float(signature.get("sustained_high_response_fraction", 0.0)),
            "%",
        ),
        (
            "Peak density",
            signature.get("peak_density_per_10_seconds"),
            "/ 10s",
        ),
        (
            "Spatial concentration",
            100.0 * float(signature.get("spatial_concentration_top10pct_share", 0.0)),
            "%",
        ),
        (
            "Early response",
            signature.get("early_0_3s_response_z_mean"),
            "z",
        ),
        (
            "Late response",
            signature.get("late_last_3s_response_z_mean"),
            "z",
        ),
    ]
    cards = []
    for label, value, suffix in specs:
        try:
            rendered = f"{float(value):.2f}{suffix}"
        except Exception:
            rendered = "—"
        cards.append(
            f'<div class="metric"><span>{html.escape(label)}</span>'
            f'<strong>{html.escape(rendered)}</strong></div>'
        )
    return "".join(cards)


def _event_cards(
    items: list[dict],
    kind: str,
    frame_lookup: dict[tuple[str, int], str],
) -> str:
    cards = []
    label = "High-response moment" if kind == "peak" else "Low-response moment"
    for item in items:
        regions = ", ".join(
            html.escape(
                f"{r.get('name','?')} ({r.get('hemisphere','?')})"
            )
            for r in item.get("top_regions", [])[:3]
        )
        images = "".join(
            f'<img src="{html.escape(name)}" alt="cortical moment">'
            for name in item.get("images", [])
        )
        frame_name = frame_lookup.get((kind, int(item.get("timestep", 0))))
        creative_frame = (
            f'<img src="{html.escape(frame_name)}" alt="creative frame">'
            if frame_name
            else ""
        )
        cards.append(
            f"""
            <section class="peak-card">
              <div class="peak-head">
                <strong>{item['start_seconds']:.1f}s</strong>
                <span>{label} • relative response z = {item['global_response_z']:.2f}</span>
              </div>
              <div class="moment-layout">
                <div class="creative-frame">{creative_frame}</div>
                <div class="brain-pair">{images}</div>
              </div>
              <p>Top cortical regions in this model window: {regions or 'atlas unavailable'}.</p>
            </section>
            """
        )
    return "".join(cards)



def create_report_assets(
    predictions: np.ndarray,
    timeline_csv: Path,
    output_dir: Path,
    metadata: dict,
    normalized_path: Path | None = None,
) -> Path:
    normalized = None
    if normalized_path and Path(normalized_path).exists():
        try:
            normalized = json.loads(Path(normalized_path).read_text(encoding="utf-8"))
        except Exception:
            normalized = None

    timeline_png = _save_timeline(timeline_csv, output_dir, normalized)

    notes: list[str] = []
    interactive: list[Path] = []
    peak_maps: list[dict] = []
    trough_maps: list[dict] = []
    frame_lookup: dict[tuple[str, int], str] = {}

    # Static cortical maps are the core visualization and should remain usable
    # even if an optional renderer (for example Plotly) is unavailable.
    try:
        brain_maps = _save_brain_maps(predictions, output_dir)
    except Exception as exc:
        previous = _existing_brain_maps(output_dir)
        if previous:
            brain_maps = previous
            notes.append(
                "Static cortical re-render failed, so the last valid cached brain maps "
                f"were preserved. Renderer error: {exc}"
            )
        else:
            try:
                brain_maps = _fallback_brain_vector(predictions, output_dir)
                notes.append(f"Static cortical surface rendering failed: {exc}")
            except Exception as fallback_exc:
                brain_maps = []
                notes.append(
                    "Static cortical rendering and its diagnostic fallback both failed. "
                    f"Surface error: {exc}; fallback error: {fallback_exc}"
                )

    try:
        interactive = _save_interactive_brains(predictions, output_dir)
    except Exception as exc:
        notes.append(f"Interactive 3D cortex unavailable: {exc}")

    try:
        peak_maps = _save_peak_maps(predictions, normalized, output_dir)
    except Exception as exc:
        notes.append(f"High-response cortical maps unavailable: {exc}")

    try:
        trough_maps = _save_trough_maps(predictions, normalized, output_dir)
    except Exception as exc:
        notes.append(f"Low-response cortical maps unavailable: {exc}")

    try:
        frame_lookup = _save_moment_frames(metadata, normalized, output_dir)
    except Exception as exc:
        notes.append(f"Creative moment thumbnails unavailable: {exc}")

    brain_note = " ".join(notes)

    aggregate_imgs = "".join(
        f'<img src="{html.escape(p.name)}" alt="brain map">'
        for p in brain_maps
    )
    if not aggregate_imgs:
        aggregate_imgs = (
            '<div class="render-placeholder">Static brain images were unavailable for '
            'this post-processing pass. Neural predictions and normalized data remain intact.</div>'
        )

    iframe_html = ""
    if len(interactive) == 2:
        iframe_html = f"""
        <div class="viewer-grid">
          <iframe src="{html.escape(interactive[0].name)}" title="Interactive left hemisphere"></iframe>
          <iframe src="{html.escape(interactive[1].name)}" title="Interactive right hemisphere"></iframe>
        </div>
        <p class="hint">Drag the interactive surfaces to rotate the cortex and inspect the spatial prediction.</p>
        """

    normalized_links = ""
    if normalized_path:
        normalized_links = """
        <div class="download-grid">
          <a href="chatgpt_brain_summary.json">chatgpt_brain_summary.json</a>
          <a href="CHATGPT_ANALYSIS_PROMPT.txt">CHATGPT_ANALYSIS_PROMPT.txt</a>
          <a href="normalized_timeline.csv">normalized_timeline.csv</a>
          <a href="roi_summary.csv">roi_summary.csv</a>
          <a href="creative_signature.json">creative_signature.json</a>
          <a href="brain_predictions.npz">brain_predictions.npz</a>
        </div>
        """

    meta = html.escape(json.dumps(metadata, indent=2, ensure_ascii=False))
    note = html.escape(brain_note)
    report = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRIBE v2 Local Report</title>
<style>
:root {{
  --bg:#090b10; --panel:#12151d; --panel2:#171b26; --border:#272c3a;
  --text:#f5f7fb; --muted:#a6adbb; --accent:#8b5cf6; --accent2:#6d28d9;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family:Inter,Segoe UI,Arial,sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width:1280px; margin:auto; padding:40px 28px 80px; }}
.hero {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-end; margin-bottom:28px; }}
.eyebrow {{ color:#b9a3ff; font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:12px; }}
h1 {{ font-size:42px; margin:8px 0 8px; letter-spacing:-.03em; }}
h2 {{ margin:0 0 18px; font-size:22px; }}
p {{ color:var(--muted); line-height:1.6; }}
.badge {{ border:1px solid #4c3b77; background:#181228; color:#c4b5fd; padding:8px 12px; border-radius:999px; white-space:nowrap; }}
.card {{ background:linear-gradient(180deg,var(--panel2),var(--panel)); border:1px solid var(--border); border-radius:22px; padding:24px; margin:18px 0; box-shadow:0 18px 50px rgba(0,0,0,.18); }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:12px; }}
.metric {{ background:#0e1118; border:1px solid #242938; border-radius:16px; padding:16px; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; margin-bottom:8px; }}
.metric strong {{ font-size:24px; }}
.brain-pair {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
.brain-pair img {{ width:100%; border-radius:15px; background:#080a0e; }}
.viewer-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
.viewer-grid iframe {{ width:100%; height:480px; border:1px solid var(--border); border-radius:16px; background:white; }}
.timeline {{ width:100%; border-radius:16px; background:white; }}
.peak-card {{ background:#0e1118; border:1px solid #242938; border-radius:18px; padding:18px; margin:14px 0; }}
.peak-head {{ display:flex; justify-content:space-between; gap:16px; margin-bottom:14px; }}
.peak-head strong {{ font-size:24px; color:#c4b5fd; }}
.peak-head span {{ color:var(--muted); }}
.download-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }}
.download-grid a {{ color:#ddd6fe; text-decoration:none; background:#0e1118; border:1px solid #2d2940; padding:14px; border-radius:14px; }}
.download-grid a:hover {{ border-color:var(--accent); }}
.callout {{ border-left:3px solid var(--accent); padding:2px 0 2px 16px; }}
.hint {{ font-size:13px; }}
pre {{ white-space:pre-wrap; color:#cbd1dc; font-size:12px; overflow:auto; }}
@media(max-width:850px) {{ .brain-pair,.viewer-grid {{ grid-template-columns:1fr; }} .hero {{ align-items:flex-start; flex-direction:column; }} }}
</style>
</head>
<body><main>
<div class="hero">
  <div>
    <div class="eyebrow">Local neural prediction</div>
    <h1>TRIBE v2 Brain Report</h1>
    <p>Predicted fMRI-like cortical response for an average subject viewing this creative.</p>
  </div>
  <div class="badge">Vision-only • fsaverage5</div>
</div>

<div class="card">
  <h2>Creative neural signature</h2>
  <div class="metrics">{_signature_cards(normalized)}</div>
  <p class="hint">These indices summarize this model output. They are not calibrated advertising KPIs.</p>
</div>

<div class="card">
  <h2>Response timeline</h2>
  <img class="timeline" src="{html.escape(timeline_png.name)}" alt="activation timeline">
  <p>Peaks are relative to this clip using robust within-clip normalization.</p>
</div>

<div class="card">
  <h2>Whole-creative cortical map</h2>
  <div class="brain-pair">{aggregate_imgs}</div>
  {iframe_html}
</div>

<div class="card">
  <h2>High-response moments</h2>
  {_event_cards(peak_maps, "peak", frame_lookup) or '<p>No high-response maps were available for this run.</p>'}
</div>

<div class="card">
  <h2>Low-response moments</h2>
  {_event_cards(trough_maps, "trough", frame_lookup) or '<p>No low-response maps were available for this run.</p>'}
</div>

<div class="card">
  <h2>AI-ready normalized output</h2>
  <div class="callout">
    <p><strong>Use chatgpt_brain_summary.json with CHATGPT_ANALYSIS_PROMPT.txt.</strong> The compact file keeps the timeline, peak moments, cortical-region summaries and descriptive neural signature without forcing an LLM to ingest every raw vertex.</p>
  </div>
  {normalized_links}
</div>

{_calibration_card(output_dir)}

<div class="card">
  <h2>Scientific interpretation boundary</h2>
  <p>This visualization is generated from TRIBE v2 model predictions. It is not a scan or measurement of any actual viewer. Marketing metrics such as CTR, hold rate, conversion rate, CPA or ROAS require empirical calibration against campaign outcomes before numerical prediction can be considered meaningful.</p>
  <p>{note}</p>
</div>

<details class="card">
  <summary>Run metadata</summary>
  <pre>{meta}</pre>
</details>
</main></body></html>"""

    report_path = output_dir / "report.html"
    report_path.write_text(report, encoding="utf-8")
    return report_path
