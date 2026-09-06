# Empirical Campaign Calibration

The TRIBE v2 Local Runner now has a second layer that learns from **real campaign outcomes** instead of pretending that cortical predictions are advertising KPIs.

## Principle

The neural pipeline remains:

```
creative -> V-JEPA2 -> TRIBE v2 -> fsaverage5 cortical prediction
```

The calibration layer is separate:

```
TRIBE-derived creative signature
          +
real campaign outcomes
          |
          v
local supervised calibrator
          |
          v
empirical KPI hypotheses for future creatives
```

TRIBE v2 does not directly output CTR, CVR, CPA, ROAS, hook rate, hold rate, memory, emotion or persuasion.

## What is saved per creative

Every completed/cached run now produces:

- `calibration_features.json`
- `campaign_metrics_template.json`
- `campaign_metrics.json` after the user enters real outcomes
- `calibrated_predictions.json` when trained calibrators are available

The shared local calibration workspace is:

```
<TRIBEv2 Results>/.calibration/
    calibration_dataset.csv
    calibration_manifest.json
    models/
        hook_rate_3s_pct.joblib
        ctr_pct.joblib
        ...
```

## Neural feature vector

The calibrator does not ingest the raw 20,484 vertices directly.

It builds a deterministic fixed-length vector from:

- scalar `creative_signature` values;
- video duration;
- 16-point normalized temporal response profile;
- temporal min/max/mean/std;
- positive and negative response area;
- relative peak and trough positions;
- left/right mean response;
- spatial concentration profile;
- Destrieux ROI mean response;
- ROI temporal variability;
- ROI within-region peak z-score.

This preserves the expensive neural analysis while making supervised calibration practical on a desktop.

## Real outcome fields

Prediction targets:

- Hook / 3-second view rate (%)
- Hold rate 15s (%)
- Video viewed 25% (%)
- Video viewed 50% (%)
- Video viewed 75% (%)
- Video viewed 95% (%)
- Average watch time (seconds)
- CTR (%)
- CVR (%)
- CPA
- ROAS

Audit/context fields:

- platform
- placement
- campaign ID
- ad ID
- spend
- impressions

Context fields are stored but are deliberately excluded from the pre-launch neural feature vector so the model cannot cheat by learning campaign-delivery shortcuts.

## Training policy

The runner currently uses a regularized Ridge regression pipeline per KPI:

```
median imputation -> standardization -> RidgeCV
```

The model is not trained until a KPI has at least **12 labeled creatives**.

Quality labels:

- <12: insufficient
- 12–29: experimental
- 30–99: emerging
- 100+: maturing

For every trained target the runner stores:

- cross-validated MAE;
- cross-validated R²;
- 80% absolute-residual band;
- 90% absolute-residual band;
- number of labeled creatives;
- calibration quality state.

The prediction JSON therefore exposes uncertainty instead of presenting an unsupported single number as truth.

## Why Ridge first

The first dataset will be small and have many correlated neural features. A strongly regularized linear model is easier to audit and substantially harder to overfit than beginning with a large neural network.

When the dataset grows, the calibration layer can benchmark:

- Elastic Net;
- gradient-boosted trees;
- random forests;
- small MLPs;
- Bayesian regression;
- multi-task models.

A more complex model should only replace Ridge when held-out validation demonstrates a real improvement.

## Cache-first development

A completed neural run is intentionally reusable.

When `brain_predictions.npz` and `timeline.csv` exist for the video's fingerprint, the app changes the main action to:

```
REPROCESSAR CACHE
```

That skips the expensive V-JEPA2/TRIBE inference and rebuilds the inexpensive stages:

- normalization;
- peak/trough detection;
- cortical visualizations;
- creative frame thumbnails;
- calibration features;
- calibrated KPI predictions;
- HTML report.

This allows UI, reporting, normalization and calibration development to continue on the same creative without waiting hours for another V-JEPA2 pass.
