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
MATCH_PATH = REPO / "results_cpack_match_0506" / "cat_local_matches.json"
OUT_ROOT = REPO / "results_cpack_smoke_0506"
sys.path.insert(0, str(REPO))

import sweep


TARGET_SINR_PRE = [-60, -55, -50, -45]
RUN_TAGS = ["precomp_analog", "precomp_digital", "precomp_diffusion"]
CALIB_SCALE = 107900


def target_rsi_scale(target_sinr_pre: float, intercept: float) -> int:
    return int(round(10 ** ((intercept - target_sinr_pre) / 10.0)))


def patch_config(asic_nsym: int, seed_value: int) -> str:
    original = CONFIG_PATH.read_text(encoding="utf-8")
    patched, count_nsym = re.subn(r"(?m)^ASIC_NSYM\s*=\s*\d+", f"ASIC_NSYM = {asic_nsym}", original, count=1)
    patched, count_seed = re.subn(r"(?m)^SEED\s*=\s*\d+", f"SEED = {seed_value}", patched, count=1)
    if count_nsym != 1 or count_seed != 1:
        raise RuntimeError("Failed to patch config.py for cpack smoke")
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
    x = load_rgb_tensor(run_dir / "img_original_remote.png")
    y = load_rgb_tensor(run_dir / "img_recon_remote.png")
    with torch.no_grad():
        val = ms_ssim(x, y, data_range=1.0, size_average=True, win_size=7).item()
    return float(val), float(safe_db(val))


def calibrate_intercept(local_img: str, remote_img: str, calib_dir: Path) -> float:
    row = sweep.run_one(
        rsi_scale=CALIB_SCALE,
        run_tag="precomp_analog",
        local_img=local_img,
        remote_img=remote_img,
        out_dir=calib_dir,
    )
    return float(row["sinr_pre"]) + 10.0 * math.log10(CALIB_SCALE)


def run_and_record(
    *,
    out_dir: Path,
    local_img: str,
    remote_img: str,
    rsi_scale: int,
    target_sinr_pre: int,
    run_tag: str,
) -> dict:
    row = sweep.run_one(
        rsi_scale=rsi_scale,
        run_tag=run_tag,
        local_img=local_img,
        remote_img=remote_img,
        out_dir=out_dir,
    )
    true_ms_ssim, true_ms_ssim_db = compute_true_ms_ssim(out_dir)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["true_ms_ssim"] = true_ms_ssim
    summary["true_ms_ssim_db"] = true_ms_ssim_db
    summary["protocol"] = "cpack_0506_weak1700_integer_smoke"
    summary["target_sinr_pre"] = target_sinr_pre
    summary["derived_rsi_scale"] = rsi_scale
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "local_img": local_img,
        "remote_img": remote_img,
        "target_sinr_pre": target_sinr_pre,
        "rsi_scale": rsi_scale,
        "run_tag": run_tag,
        "sinr_pre": summary.get("sinr_pre"),
        "sinr_after_analog": summary.get("sinr_after_analog"),
        "sinr_after_digital": summary.get("sinr_after_digital"),
        "psnr": summary.get("psnr"),
        "ms_ssim": summary.get("ms_ssim"),
        "true_ms_ssim_db": true_ms_ssim_db,
        "time_sec": summary.get("time_sec"),
    }


def load_matches() -> list[dict]:
    payload = json.loads(MATCH_PATH.read_text(encoding="utf-8"))
    return payload["matches"]


def write_summary(rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(Path(row["remote_img"]).name, []).append(row)

    md = [
        "# Cat Pack Weak1700 Integer Smoke",
        "",
        "- Protocol: `weak1700 + integer SINR + matched local Kodak per cat`",
        "- Methods: `Analog / MP / Diffusion`",
        "- Targets: `-60 / -55 / -50 / -45 dB`",
        "",
    ]
    summary_json = {}
    for cat_name in sorted(grouped):
        bucket = grouped[cat_name]
        local_name = Path(bucket[0]["local_img"]).name
        md.append(f"## {cat_name}")
        md.append("")
        md.append(f"- matched local: `{local_name}`")
        md.append("")
        md.append("| target_sinr_pre | actual sinr_pre | sinr_after_analog | analog | mp | diff | diff-mp | true msssim diff-mp |")
        md.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        cat_rows = []
        for target in TARGET_SINR_PRE:
            analog = next(r for r in bucket if r["target_sinr_pre"] == target and r["run_tag"] == "precomp_analog")
            mp = next(r for r in bucket if r["target_sinr_pre"] == target and r["run_tag"] == "precomp_digital")
            diff = next(r for r in bucket if r["target_sinr_pre"] == target and r["run_tag"] == "precomp_diffusion")
            md.append(
                f"| {target:+d} | {float(analog['sinr_pre']):.2f} | {float(analog['sinr_after_analog']):.2f} | "
                f"{float(analog['psnr']):.2f} | {float(mp['psnr']):.2f} | {float(diff['psnr']):.2f} | "
                f"{float(diff['psnr']) - float(mp['psnr']):+.2f} | {float(diff['true_ms_ssim_db']) - float(mp['true_ms_ssim_db']):+.2f} |"
            )
            cat_rows.append(
                {
                    "target_sinr_pre": target,
                    "actual_sinr_pre": float(analog["sinr_pre"]),
                    "sinr_after_analog": float(analog["sinr_after_analog"]),
                    "analog_psnr": float(analog["psnr"]),
                    "mp_psnr": float(mp["psnr"]),
                    "diff_psnr": float(diff["psnr"]),
                    "diff_minus_mp_psnr": float(diff["psnr"]) - float(mp["psnr"]),
                    "diff_minus_mp_true_msssim_db": float(diff["true_ms_ssim_db"]) - float(mp["true_ms_ssim_db"]),
                }
            )
        md.append("")
        summary_json[cat_name] = {"local": local_name, "rows": cat_rows}

    (OUT_ROOT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    rows: list[dict] = []
    matches = load_matches()

    original_config = patch_config(1700, 2025)
    try:
        for match in matches:
            remote_name = match["remote_name"]
            local_name = match["selected_local"]
            local_img = f"data/kodak/{local_name}"
            remote_img = f"data/cat/{remote_name}"
            safe_remote_stem = Path(remote_name).stem
            intercept = calibrate_intercept(
                local_img,
                remote_img,
                OUT_ROOT / f"{safe_remote_stem}_calib_precomp_analog",
            )
            for target in TARGET_SINR_PRE:
                rsi_scale = target_rsi_scale(target, intercept)
                for run_tag in RUN_TAGS:
                    run_dir = OUT_ROOT / f"{safe_remote_stem}_sinr{target:+d}dB_{run_tag}"
                    rows.append(
                        run_and_record(
                            out_dir=run_dir,
                            local_img=local_img,
                            remote_img=remote_img,
                            rsi_scale=rsi_scale,
                            target_sinr_pre=target,
                            run_tag=run_tag,
                        )
                    )
    finally:
        restore_config(original_config)

    (OUT_ROOT / "rows.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(rows)
    print(f"[OK] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
