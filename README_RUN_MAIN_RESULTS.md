# Workstation Runnable Main Results Bundle

Created: 2026-05-06

This is the compact runnable bundle for the current weak1700 report-ready mainline.

Unlike the older bundle, this one is intentionally created **without any result folders**.
It is meant to verify reproducibility by rerunning selected official results inside the bundle.

## Current Official Story

Primary current report assets live outside this bundle in the parent workspace:

- `results_wint_0506/`
- `results_wsq_0506_cat6/`
- `results_wsq_0506_cat2/`
- `results_wsq_0506_cat7/`

Inside this bundle, those result folders are **not copied**. They should be regenerated locally.

## Included Runtime Assets

Included locally but ignored by Git:

- `data/`
- `checkpoints/`
- `ddpm_models/`

Included code:

- `run_sdd_final.py`
- `sweep.py`
- `run_diffusion.py`
- `run_analog_semantic.py`
- `run_analog_semantic_toy.py`
- `src/`
- `layer/`
- `SIC/`
- `utils/`

Included reproducibility scripts:

- `scripts/probe_wint_0506.py`
- `scripts/build_wint_0506_bundle.py`
- `scripts/build_wint_0506_audit.py`
- `scripts/build_wint_fair_0506.py`
- `scripts/select_cat_locals_0506.py`
- `scripts/probe_cpack_0506.py`
- `scripts/probe_wsq_0506_cat6.py`
- `scripts/build_wsq_0506_cat6_bundle.py`
- `scripts/probe_wsq_0506_cat2.py`
- `scripts/build_wsq_0506_cat2_bundle.py`
- `scripts/probe_wsq_0506_cat7.py`
- `scripts/build_wsq_0506_cat7_bundle.py`

## Current Cat Inputs

Square-report remotes already included in `data/cat/`:

- `cat6_center_square.png`
- `cat2_center_square.png`
- `cat7_center_square.png`

These are the current report-facing cat remotes used with the standard `resize_128` protocol.

## Recommended Verification Run

First set UTF-8 on Windows/PowerShell:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

Then from the bundle root, rerun one full official cat candidate:

```powershell
C:\Users\jing5\anaconda3\envs\se\python.exe scripts\probe_wsq_0506_cat6.py
C:\Users\jing5\anaconda3\envs\se\python.exe scripts\build_wsq_0506_cat6_bundle.py
```

Expected regenerated folder:

```text
results_wsq_0506_cat6/
```

## Why This Bundle Exists

This bundle is for:

1. workstation-copyable reruns
2. Git-safe code/MD changes
3. verifying that the current report story is reproducible from a compact folder

It is not a historical archive of every exploratory result.
