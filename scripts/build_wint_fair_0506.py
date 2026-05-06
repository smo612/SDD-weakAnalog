from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent.parent
SRC_MAIN = REPO / "results_wint_0506" / "tables" / "rows.json"
SRC_CAT = REPO / "results_wint_0506_cat" / "tables" / "rows.json"
OUT_ROOT = REPO / "results_wint_fair_0506"
OUT_FIG = OUT_ROOT / "figures"
OUT_NOTES = OUT_ROOT / "notes"
TARGETS = [-60, -55, -50, -45, -40, -35, -30]


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)


def select_main_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row.get("setting_label") == "v3c_ref":
            out.append(row)
            continue
        target = int(row["target_sinr_pre"])
        if row["run_tag"] == "precomp_diffusion" and target == -35:
            if row["variant"] == "t2":
                out.append(row)
        elif row["variant"] == "baseline":
            out.append(row)
    return out


def mean_realistic_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in select_main_rows(rows):
        if row.get("setting_label") == "v3c_ref":
            continue
        key = (int(row["target_sinr_pre"]), row["run_tag"])
        grouped.setdefault(key, []).append(row)
    out = []
    for (target, run_tag), bucket in sorted(grouped.items()):
        out.append(
            {
                "target_sinr_pre": target,
                "run_tag": run_tag,
                "sinr_after_analog": float(np.mean([float(r["sinr_after_analog"]) for r in bucket])),
                "psnr": float(np.mean([float(r["psnr"]) for r in bucket])),
                "true_ms_ssim_db": float(np.mean([float(r["true_ms_ssim_db"]) for r in bucket])),
            }
        )
    return out


def cat_selected_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        target = int(row["target_sinr_pre"])
        if row["run_tag"] == "precomp_diffusion" and target == -35:
            if row["variant"] == "t2":
                out.append(row)
        elif row["variant"] == "baseline":
            out.append(row)
    return sorted(out, key=lambda r: (int(r["target_sinr_pre"]), r["run_tag"]))


def v3c_rows(rows: list[dict]) -> list[dict]:
    return sorted([r for r in rows if r["run_tag"] == "trackb_ref_q3_midmid_crop"], key=lambda r: int(r["target_sinr_pre"]))


def interp_extrap(xs: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    order = np.argsort(xp)
    xp = xp[order]
    yp = yp[order]
    ys = np.interp(xs, xp, yp)
    if len(xp) >= 2:
        left_mask = xs < xp[0]
        right_mask = xs > xp[-1]
        if np.any(left_mask):
            slope = (yp[1] - yp[0]) / (xp[1] - xp[0])
            ys[left_mask] = yp[0] + slope * (xs[left_mask] - xp[0])
        if np.any(right_mask):
            slope = (yp[-1] - yp[-2]) / (xp[-1] - xp[-2])
            ys[right_mask] = yp[-1] + slope * (xs[right_mask] - xp[-1])
    return ys


def build_case(real_rows: list[dict], v3_rows: list[dict], label: str) -> dict:
    x_common = np.array([next(float(r["sinr_after_analog"]) for r in real_rows if int(r["target_sinr_pre"]) == t and r["run_tag"] == "precomp_analog") for t in TARGETS], dtype=float)
    x_v3c = np.array([float(next(r["sinr_after_analog"] for r in v3_rows if int(r["target_sinr_pre"]) == t)) for t in TARGETS], dtype=float)

    out = {
        "label": label,
        "targets": TARGETS,
        "x_common": x_common.tolist(),
        "x_v3c_raw": x_v3c.tolist(),
        "series": {},
    }
    for run_tag in ["precomp_analog", "precomp_digital", "precomp_diffusion"]:
        out["series"][run_tag] = {
            "psnr": [float(next(r["psnr"] for r in real_rows if int(r["target_sinr_pre"]) == t and r["run_tag"] == run_tag)) for t in TARGETS],
            "true_ms_ssim_db": [float(next(r["true_ms_ssim_db"] for r in real_rows if int(r["target_sinr_pre"]) == t and r["run_tag"] == run_tag)) for t in TARGETS],
        }

    v3_psnr = np.array([float(next(r["psnr"] for r in v3_rows if int(r["target_sinr_pre"]) == t)) for t in TARGETS], dtype=float)
    v3_mss = np.array([float(next(r["true_ms_ssim_db"] for r in v3_rows if int(r["target_sinr_pre"]) == t)) for t in TARGETS], dtype=float)
    out["series"]["v3c_raw"] = {
        "psnr": v3_psnr.tolist(),
        "true_ms_ssim_db": v3_mss.tolist(),
    }
    out["series"]["v3c_matched"] = {
        "psnr": interp_extrap(x_common, x_v3c, v3_psnr).tolist(),
        "true_ms_ssim_db": interp_extrap(x_common, x_v3c, v3_mss).tolist(),
    }
    return out


def plot_case(case: dict, metric: str, out_name: str, title: str) -> None:
    x = np.array(case["x_common"], dtype=float)
    plt.figure(figsize=(8.8, 6.0))
    styles = {
        "precomp_analog": ("Analog Only", "#6f6f6f", "v", 2.0, 7),
        "precomp_digital": ("MP", "#d4720a", "s", 2.0, 7),
        "precomp_diffusion": ("Diffusion", "#1a6bb5", "*", 2.6, 11),
        "v3c_matched": ("V3c Surrogate (matched x)", "#c1121f", "P", 2.4, 9),
    }
    for key, (label, color, marker, lw, ms) in styles.items():
        y = np.array(case["series"][key][metric], dtype=float)
        ls = "--" if key == "v3c_matched" else "-"
        plt.plot(x, y, label=label, color=color, marker=marker, linewidth=lw, markersize=ms, linestyle=ls)

    plt.xlabel("Actual Post-Analog SINR (matched realistic x) (dB)", fontsize=11)
    plt.ylabel("True MS-SSIM-dB" if metric == "true_ms_ssim_db" else "Image PSNR (dB)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, ls="--", alpha=0.45)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_FIG / out_name, dpi=300)
    plt.close()


def write_case_summary(case: dict, stem: str) -> None:
    diff_psnr = np.array(case["series"]["precomp_diffusion"]["psnr"], dtype=float)
    mp_psnr = np.array(case["series"]["precomp_digital"]["psnr"], dtype=float)
    v3_psnr = np.array(case["series"]["v3c_matched"]["psnr"], dtype=float)
    diff_mss = np.array(case["series"]["precomp_diffusion"]["true_ms_ssim_db"], dtype=float)
    mp_mss = np.array(case["series"]["precomp_digital"]["true_ms_ssim_db"], dtype=float)
    v3_mss = np.array(case["series"]["v3c_matched"]["true_ms_ssim_db"], dtype=float)

    lines = [
        f"# {case['label']} Fair Overlay Summary",
        "",
        "| target_sinr_pre | realistic x (sinr_after_analog) | diff-mp (PSNR) | diff-v3c_matched (PSNR) | diff-mp (MS-SSIM) | diff-v3c_matched (MS-SSIM) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, target in enumerate(TARGETS):
        lines.append(
            f"| {target:+d} | {case['x_common'][idx]:.2f} | {diff_psnr[idx]-mp_psnr[idx]:+.2f} | {diff_psnr[idx]-v3_psnr[idx]:+.2f} | {diff_mss[idx]-mp_mss[idx]:+.2f} | {diff_mss[idx]-v3_mss[idx]:+.2f} |"
        )
    (OUT_NOTES / f"{stem}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    text = """# WINT Fair Overlay

This folder applies a matched-after-analog fairness audit to the existing
`results_wint_0506` and `results_wint_0506_cat` results.

No new experiments were run.

Method:
- keep the realistic three-line results on their actual post-analog SINR
- interpolate the V3c surrogate onto those same x locations
- compare Diff/MP/Analog/V3c under matched actual operating points
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    main_rows = load_rows(SRC_MAIN)
    cat_rows = load_rows(SRC_CAT)

    main_case = build_case(mean_realistic_rows(main_rows), v3c_rows(select_main_rows(main_rows)), "Mainline")
    cat_case = build_case([r for r in cat_selected_rows(cat_rows) if r["run_tag"] != "trackb_ref_q3_midmid_crop"], [r for r in cat_selected_rows(cat_rows) if r["run_tag"] == "trackb_ref_q3_midmid_crop"], "Cat")

    plot_case(main_case, "psnr", "main_fair_psnr.png", "WINT 0506 Mainline Fair Overlay (PSNR)")
    plot_case(main_case, "true_ms_ssim_db", "main_fair_msssim.png", "WINT 0506 Mainline Fair Overlay (True MS-SSIM)")
    plot_case(cat_case, "psnr", "cat_fair_psnr.png", "WINT 0506 Cat Fair Overlay (PSNR)")
    plot_case(cat_case, "true_ms_ssim_db", "cat_fair_msssim.png", "WINT 0506 Cat Fair Overlay (True MS-SSIM)")

    (OUT_NOTES / "main_case.json").write_text(json.dumps(main_case, indent=2), encoding="utf-8")
    (OUT_NOTES / "cat_case.json").write_text(json.dumps(cat_case, indent=2), encoding="utf-8")
    write_case_summary(main_case, "main")
    write_case_summary(cat_case, "cat")
    write_readme()
    print(f"[OK] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
