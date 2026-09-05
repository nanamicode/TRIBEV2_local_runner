from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _save_timeline(timeline_csv: Path, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    df = pd.read_csv(timeline_csv)
    out = output_dir / "activation_timeline.png"
    fig = plt.figure(figsize=(12, 4.8), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(df["start_seconds"], df["mean_abs"])
    ax.set_title("TRIBE v2 — predicted cortical activity over time")
    ax.set_xlabel("Video time (s)")
    ax.set_ylabel("Mean absolute predicted response")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _save_brain_maps(predictions: np.ndarray, output_dir: Path) -> list[Path]:
    # Aggregate absolute response over time. This gives a simple "where the
    # model predicts the strongest cortical response" view for the whole clip.
    from nilearn import datasets, plotting
    import matplotlib.pyplot as plt

    aggregate = np.mean(np.abs(predictions), axis=0)
    n = aggregate.size
    half = n // 2
    if half * 2 != n:
        raise RuntimeError(f"Expected paired left/right cortical vertices, got {n}")

    fs = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    outputs = []
    specs = [
        ("left", fs.infl_left, fs.sulc_left, aggregate[:half]),
        ("right", fs.infl_right, fs.sulc_right, aggregate[half:]),
    ]
    for hemi, mesh, sulc, values in specs:
        out = output_dir / f"brain_{hemi}_lateral.png"
        display = plotting.plot_surf_stat_map(
            mesh,
            values,
            hemi=hemi,
            view="lateral",
            bg_map=sulc,
            colorbar=True,
            title=f"TRIBE v2 — {hemi} hemisphere",
            cmap="inferno",
        )
        display.savefig(out, dpi=160)
        plt.close(display.figure)
        outputs.append(out)
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
    fig.savefig(out)
    plt.close(fig)
    return [out]


def create_report_assets(
    predictions: np.ndarray,
    timeline_csv: Path,
    output_dir: Path,
    metadata: dict,
) -> None:
    timeline_png = _save_timeline(timeline_csv, output_dir)
    try:
        brain_maps = _save_brain_maps(predictions, output_dir)
        brain_note = ""
    except Exception as exc:
        brain_maps = _fallback_brain_vector(predictions, output_dir)
        brain_note = (
            "The 3D/surface renderer was unavailable, so the report used a "
            f"vertex-space fallback. Renderer error: {exc}"
        )

    imgs = "\n".join(
        f'<img src="{html.escape(p.name)}" style="max-width:100%;border-radius:16px;margin:10px 0">'
        for p in [timeline_png, *brain_maps]
    )
    meta = html.escape(json.dumps(metadata, indent=2))
    note = html.escape(brain_note)
    report = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TRIBE v2 Local Report</title>
<style>
body {{ margin:0; font-family:Segoe UI,Arial,sans-serif; background:#0b0b0f; color:#f5f5f7; }}
main {{ max-width:1100px; margin:auto; padding:36px; }}
.card {{ background:#15151c; border:1px solid #2a2a35; border-radius:22px; padding:24px; margin:18px 0; }}
h1 {{ font-size:34px; }}
small,p {{ color:#b6b6c4; line-height:1.55; }}
pre {{ white-space:pre-wrap; color:#d4d4dc; }}
</style>
</head>
<body><main>
<h1>TRIBE v2 Local Report</h1>
<p>Vision-only local inference. This is a predicted fMRI-like response for an average subject; it is not a direct measurement of a real viewer.</p>
<div class="card">{imgs}</div>
<div class="card"><p>{note}</p><pre>{meta}</pre></div>
</main></body></html>"""
    (output_dir / "report.html").write_text(report, encoding="utf-8")
