# Architecture

## Objective

Run TRIBE v2 locally on ordinary Windows desktop hardware and accept arbitrary video files without a hosted inference service.

The product target is deliberately split into two generations.

## Generation A — bring-up / validation path

The repository currently contains a Windows desktop MVP that:

- downloads a private Python 3.11 runtime on first installation;
- installs PyTorch + TorchAO locally;
- installs Meta's official TRIBE v2 source;
- downloads the quantized `Jessylg27/tribev2-lite-qv` package on first inference;
- runs **vision-only** inference;
- keeps the original TRIBE v2 cortical prediction head;
- writes raw cortical predictions, timeline CSV, surface images and a local HTML report.

Why vision-only first:

- the official text branch uses Llama 3.2 3B;
- audio + transcription adds another heavyweight stack;
- the TRIBE training/configuration supports missing modalities;
- video is the modality most directly useful for the first creative-analysis MVP.

This generation exists to get a reproducible Windows result and a reference output before replacing more of the Python stack.

## Generation B — selected low-level target

The primary optimization target is now a native Rust backend based on the public
`eugenehp/tribev2-rs` implementation.

Reasons:

- the TRIBE cortical encoder is already ported to Rust;
- pretrained weights can be loaded as safetensors;
- the project provides CPU, Burn/wgpu and RLX backend work;
- it includes surface/ROI/NIfTI/stimulus visualization code;
- it reports numerical parity tests against the official Python implementation;
- source is Apache-2.0 (the pretrained TRIBE weights remain CC BY-NC 4.0).

### Planned native pipeline

```
Windows UI
  |
  +-- hardware probe
  |
  +-- media decode / ffmpeg
  |
  +-- V-JEPA2 visual feature extractor
  |     |
  |     +-- quantized / reduced-memory path
  |
  +-- native TRIBE v2 brain encoder
  |     |
  |     +-- CPU: optimized BLAS/RLX path
  |     +-- NVIDIA: CUDA path
  |     +-- other GPU: wgpu/DirectX-compatible experiments
  |
  +-- fsaverage5 outputs
        |
        +-- raw vertices
        +-- brain maps
        +-- timeline
        +-- HTML report
```

## Main bottleneck

The official visual extractor is V-JEPA2 ViT-G. Its public FP32 package is much larger than the final TRIBE cortical head. Simply porting the final head to Rust does not make the whole video pipeline lightweight.

Therefore optimization order is:

1. preserve TRIBE feature compatibility;
2. validate INT8/low-precision V-JEPA2 outputs against the official extractor;
3. cache visual features per video chunk;
4. stream/chunk long videos instead of holding the whole clip in RAM;
5. move the cortical head to native Rust;
6. only then add full audio/text mode.

## Compatibility rule

Do **not** substitute a smaller V-JEPA2 model merely because it is faster unless an adapter/distillation stage is trained and validated. The pretrained TRIBE head expects specific feature dimensions/distributions. A smaller encoder with incompatible features can produce a result that looks like a brain map while no longer being a faithful TRIBE v2 prediction.

## Output semantics

TRIBE v2 predicts fMRI-like responses for an average subject on the fsaverage5 cortical surface (~20k vertices). It does not read the user's brain and it is not a literal EEG/attention measurement.

For creative analysis, downstream tooling may derive interpretable summaries, but those should be described as **model-predicted cortical response**, not measured human neural activity.

## Licensing

Meta's released TRIBE v2 model/code assets are CC BY-NC 4.0. The low-level Rust implementation's source may be Apache-2.0, but that does not change the license on Meta's pretrained weights.

Because a production creative-analysis tool is likely a commercial use, obtain the required model/weights permission before commercial deployment.
