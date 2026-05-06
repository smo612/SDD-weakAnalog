import json
from pathlib import Path

import numpy as np


EPS = 1e-12


def load_complex_npy(path):
    arr = np.load(path)

    if np.iscomplexobj(arr):
        return arr.reshape(-1).astype(np.complex64)

    if arr.ndim >= 1 and arr.shape[-1] == 2:
        real = arr[..., 0]
        imag = arr[..., 1]
        return (real + 1j * imag).reshape(-1).astype(np.complex64)

    raise ValueError(f"Unsupported array format in {path}, shape={arr.shape}, dtype={arr.dtype}")


def _safe_db(num, den, eps=EPS):
    return 10.0 * np.log10((float(num) + eps) / (float(den) + eps))


def _complex_corr(a, b):
    num = np.vdot(a, b)
    den = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real) + EPS
    return num / den


def read_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def compute_symbol_fidelity_metrics(y_clean, y_desired, y_noisy=None):
    if y_desired is None:
        return {}

    if y_noisy is None:
        L = min(len(y_clean), len(y_desired))
        y_noisy_use = None
    else:
        L = min(len(y_clean), len(y_desired), len(y_noisy))
        y_noisy_use = y_noisy[:L]

    if L <= 0:
        return {}

    y_clean = y_clean[:L]
    y_desired = y_desired[:L]

    ref_power = float(np.mean(np.abs(y_desired) ** 2)) + EPS
    mse_clean = float(np.mean(np.abs(y_clean - y_desired) ** 2))
    nmse_clean_db = _safe_db(mse_clean, ref_power)
    symbol_sinr_clean_db = _safe_db(ref_power, mse_clean)
    corr_clean = _complex_corr(y_clean, y_desired)

    metrics = {
        "reference_power": ref_power,
        "mse_clean": mse_clean,
        "nmse_clean_db": nmse_clean_db,
        "symbol_sinr_clean_db": symbol_sinr_clean_db,
        "corr_clean_abs": float(np.abs(corr_clean)),
        "corr_clean_angle_rad": float(np.angle(corr_clean)),
    }

    if y_noisy_use is not None:
        mse_noisy = float(np.mean(np.abs(y_noisy_use - y_desired) ** 2))
        nmse_noisy_db = _safe_db(mse_noisy, ref_power)
        symbol_sinr_noisy_db = _safe_db(ref_power, mse_noisy)
        corr_noisy = _complex_corr(y_noisy_use, y_desired)
        mse_gain_db = _safe_db(mse_noisy, mse_clean)
        relative_gain_pct = ((mse_noisy - mse_clean) / (mse_noisy + EPS)) * 100.0

        metrics.update({
            "mse_noisy": mse_noisy,
            "nmse_noisy_db": nmse_noisy_db,
            "symbol_sinr_noisy_db": symbol_sinr_noisy_db,
            "corr_noisy_abs": float(np.abs(corr_noisy)),
            "corr_noisy_angle_rad": float(np.angle(corr_noisy)),
            "mse_gain_db": mse_gain_db,
            "symbol_gain_db": symbol_sinr_clean_db - symbol_sinr_noisy_db,
            "relative_mse_improvement_pct": float(relative_gain_pct),
        })

    return metrics


def build_unified_digital_metrics(
    backend_name,
    y_clean,
    noisy_path="bridge/y_adc.npy",
    desired_path="bridge_tx_remote/x_tx.npy",
    analog_meta_path="bridge/meta.json",
    extra_metrics=None,
):
    analog_meta = read_json_if_exists(analog_meta_path)
    metrics = {
        "backend": backend_name,
        "metric_mode": "symbol_fidelity",
        "SINR_analog": analog_meta.get("SINR_analog"),
    }

    noisy = None
    desired = None

    noisy_path = Path(noisy_path)
    desired_path = Path(desired_path)

    if noisy_path.exists():
        try:
            noisy = load_complex_npy(noisy_path)
        except Exception:
            noisy = None

    if desired_path.exists():
        try:
            desired = load_complex_npy(desired_path)
        except Exception:
            desired = None

    symbol_metrics = compute_symbol_fidelity_metrics(y_clean, desired, noisy)
    metrics.update(symbol_metrics)

    if "symbol_sinr_clean_db" in symbol_metrics:
        metrics["SINR_after_digital"] = symbol_metrics["symbol_sinr_clean_db"]
    else:
        metrics["metric_mode"] = "backend_native"

    if extra_metrics:
        metrics.update(extra_metrics)

    if metrics.get("SINR_after_digital") is None:
        physics_sinr = metrics.get("SINR_after_digital_physics")
        if physics_sinr is not None:
            metrics["SINR_after_digital"] = physics_sinr

    return metrics
