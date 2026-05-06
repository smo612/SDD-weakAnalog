from __future__ import annotations

import json
import math
import re
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image
from pytorch_msssim import ms_ssim


REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO / "config.py"
OUT_ROOT = REPO / "results_probe_wint_0506"
sys.path.insert(0, str(REPO))

import sweep


SETTINGS = [
    {"label": "weak1700_seed2025", "asic_nsym": 1700, "seed": 2025},
    {"label": "weak1700_seed2026", "asic_nsym": 1700, "seed": 2026},
    {"label": "weak1700_seed2027", "asic_nsym": 1700, "seed": 2027},
]
TARGET_SINR_PRE = [-60, -55, -50, -45, -40, -35, -30]
RUN_TAGS = ["precomp_analog", "precomp_digital", "precomp_diffusion"]
V3C_TAG = "trackb_ref_q3_midmid_crop"
REPAIR_TARGET = -35


def target_rsi_scale(target_sinr_pre: float, intercept: float) -> int:
    return int(round(10 ** ((intercept - target_sinr_pre) / 10.0)))


def patch_config(asic_nsym: int, seed_value: int) -> str:
    original = CONFIG_PATH.read_text(encoding="utf-8")
    patched, count_nsym = re.subn(r"(?m)^ASIC_NSYM\s*=\s*\d+", f"ASIC_NSYM = {asic_nsym}", original, count=1)
    patched, count_seed = re.subn(r"(?m)^SEED\s*=\s*\d+", f"SEED = {seed_value}", patched, count=1)
    if count_nsym != 1 or count_seed != 1:
        raise RuntimeError("Failed to patch config.py for weak integer probe")
    CONFIG_PATH.write_text(patched, encoding="utf-8")
    return original


def restore_config(original_text: str) -> None:
    CONFIG_PATH.write_text(original_text, encoding="utf-8")


def load_rgb_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def safe_db(v: float) -> float:
    v = min(max(float(v), 0.0), 1.0 - 1e-8)
    return -10.0 * math.log10(1.0 - v)


def compute_true_ms_ssim(run_dir: Path) -> tuple[float, float]:
    original = run_dir / "img_original_remote.png"
    recon = run_dir / "img_recon_remote.png"
    x = load_rgb_tensor(original)
    y = load_rgb_tensor(recon)
    with torch.no_grad():
        val = ms_ssim(x, y, data_range=1.0, size_average=True, win_size=7).item()
    return float(val), float(safe_db(val))


def calibrate_intercept(setting_dir: Path) -> float:
    calib_scale = 107900
    calib_dir = setting_dir / "_calib_precomp_analog"
    row = sweep.run_one(
        rsi_scale=calib_scale,
        run_tag="precomp_analog",
        local_img=sweep.DEFAULT_LOCAL,
        remote_img=sweep.DEFAULT_REMOTE,
        out_dir=calib_dir,
    )
    return float(row["sinr_pre"]) + 10.0 * math.log10(calib_scale)


def run_and_record(
    *,
    out_dir: Path,
    rsi_scale: int,
    target_sinr_pre: int,
    run_tag: str,
    diffusion_t_start: int | None,
    protocol: str,
) -> dict:
    row = sweep.run_one(
        rsi_scale=rsi_scale,
        run_tag=run_tag,
        local_img=sweep.DEFAULT_LOCAL,
        remote_img=sweep.DEFAULT_REMOTE,
        out_dir=out_dir,
        diffusion_t_start=diffusion_t_start,
    )
    true_ms_ssim, true_ms_ssim_db = compute_true_ms_ssim(out_dir)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["true_ms_ssim"] = true_ms_ssim
    summary["true_ms_ssim_db"] = true_ms_ssim_db
    summary["ssim_proxy_db"] = float(safe_db(summary["ms_ssim"])) if summary.get("ms_ssim") is not None else float("nan")
    summary["protocol"] = protocol
    summary["target_sinr_pre"] = target_sinr_pre
    summary["derived_rsi_scale"] = rsi_scale
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "rsi_scale": rsi_scale,
        "target_sinr_pre": target_sinr_pre,
        "run_tag": run_tag,
        "sinr_pre": summary.get("sinr_pre"),
        "sinr_after_analog": summary.get("sinr_after_analog"),
        "sinr_after_digital": summary.get("sinr_after_digital"),
        "psnr": summary.get("psnr"),
        "ms_ssim": summary.get("ms_ssim"),
        "true_ms_ssim_db": true_ms_ssim_db,
        "time_sec": summary.get("time_sec"),
        "variant": "t2" if diffusion_t_start == 2 else "baseline",
    }


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    rows: list[dict] = []

    for setting in SETTINGS:
        label = setting["label"]
        seed = setting["seed"]
        asic_nsym = setting["asic_nsym"]
        setting_dir = OUT_ROOT / label
        setting_dir.mkdir(exist_ok=True)

        original_config = patch_config(asic_nsym, seed)
        try:
            intercept = calibrate_intercept(setting_dir)
            for target in TARGET_SINR_PRE:
                rsi_scale = target_rsi_scale(target, intercept)
                for run_tag in RUN_TAGS:
                    run_dir = setting_dir / f"sinr{target:+d}dB_{run_tag}"
                    rec = run_and_record(
                        out_dir=run_dir,
                        rsi_scale=rsi_scale,
                        target_sinr_pre=target,
                        run_tag=run_tag,
                        diffusion_t_start=None,
                        protocol="wint_0506_realistic_default",
                    )
                    rec["setting_label"] = label
                    rec["seed"] = seed
                    rec["asic_nsym"] = asic_nsym
                    rows.append(rec)

                    if target == REPAIR_TARGET and run_tag == "precomp_diffusion":
                        t2_dir = setting_dir / f"sinr{target:+d}dB_{run_tag}_t2"
                        rec_t2 = run_and_record(
                            out_dir=t2_dir,
                            rsi_scale=rsi_scale,
                            target_sinr_pre=target,
                            run_tag=run_tag,
                            diffusion_t_start=2,
                            protocol="wint_0506_realistic_default",
                        )
                        rec_t2["setting_label"] = label
                        rec_t2["seed"] = seed
                        rec_t2["asic_nsym"] = asic_nsym
                        rows.append(rec_t2)
        finally:
            restore_config(original_config)

    v3c_dir = OUT_ROOT / "v3c_ref"
    v3c_dir.mkdir(exist_ok=True)
    original_config = patch_config(1700, 2025)
    try:
        intercept_2025 = calibrate_intercept(v3c_dir)
    finally:
        restore_config(original_config)
    for target in TARGET_SINR_PRE:
        rsi_scale = target_rsi_scale(target, intercept_2025)
        run_dir = v3c_dir / f"sinr{target:+d}dB_{V3C_TAG}"
        rec = run_and_record(
            out_dir=run_dir,
            rsi_scale=rsi_scale,
            target_sinr_pre=target,
            run_tag=V3C_TAG,
            diffusion_t_start=None,
            protocol="wint_0506_realistic_default",
        )
        rec["setting_label"] = "v3c_ref"
        rec["seed"] = 0
        rec["asic_nsym"] = 0
        rows.append(rec)

    (OUT_ROOT / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
