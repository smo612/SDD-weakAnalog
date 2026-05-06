# Main Results Bundle Handoff

Last updated: 2026-05-06

Bundle path:

```text
WORKSTATION_RUNNABLE_MAIN_RESULTS_BUNDLE_2026-05-06/
```

## Purpose

This bundle is the compact runnable folder for the current weak1700 report-ready story.

Key difference versus the old 2026-04-29 bundle:

- this bundle does **not** copy any result folders
- reproducibility is verified by rerunning selected official results inside the bundle

## Current Official External Results

Outside this bundle, the current official report-facing results are:

- `results_wint_0506/` : weak1700 Kodak mainline
- `results_wsq_0506_cat6/` : current primary cat candidate
- `results_wsq_0506_cat2/` : cat backup
- `results_wsq_0506_cat7/` : second backup

## What Is Included

- runtime code
- semantic / SIC modules
- local assets (`data/`, `checkpoints/`, `ddpm_models/`)
- official current probe/build scripts

## What Is Intentionally Excluded

- all `results_*` folders
- temporary bridge folders
- exploratory clutter not needed for current reruns

## Minimal Verification

Inside this bundle, the preferred first verification is:

```powershell
python scripts\probe_wsq_0506_cat6.py
python scripts\build_wsq_0506_cat6_bundle.py
```

This verifies that the bundle can regenerate one of the current official report-facing cat candidates from scratch.
