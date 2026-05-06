from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image


REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "results_probe_wint_0506"
OUT_ROOT = REPO / "results_wint_0506"
OUT_FIG = OUT_ROOT / "figures"
OUT_TAB = OUT_ROOT / "tables"
OUT_NOTES = OUT_ROOT / "notes"
BADGE_GREEN = "#00883a"
TARGETS = [-60, -55, -50, -45, -40, -35, -30]
SERIES = [
    ("trackb_ref_q3_midmid_crop", "SDD Fig4", "#c1121f"),
    ("precomp_analog", "Aux-TX Only", "#6f6f6f"),
    ("precomp_digital", "Aux-TX + MP", "#d4720a"),
    ("precomp_diffusion", "Aux-TX + Diffusion", "#1a6bb5"),
]


def read_rows() -> list[dict]:
    return json.loads((SOURCE / "rows.json").read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)


def select_mean_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        if row["setting_label"] == "v3c_ref":
            continue
        if row["variant"] == "t2" and int(row["target_sinr_pre"]) != -35:
            continue
        key = (int(row["target_sinr_pre"]), row["run_tag"])
        if row["run_tag"] == "precomp_diffusion" and int(row["target_sinr_pre"]) == -35:
            if row["variant"] != "t2":
                continue
        elif row["variant"] != "baseline":
            continue
        grouped.setdefault(key, []).append(row)
    out = []
    for (target, run_tag), bucket in sorted(grouped.items()):
        out.append(
            {
                "target_sinr_pre": target,
                "run_tag": run_tag,
                "sinr_pre": float(np.mean([float(r["sinr_pre"]) for r in bucket])),
                "psnr": float(np.mean([float(r["psnr"]) for r in bucket])),
                "psnr_min": float(np.min([float(r["psnr"]) for r in bucket])),
                "psnr_max": float(np.max([float(r["psnr"]) for r in bucket])),
                "true_ms_ssim_db": float(np.mean([float(r["true_ms_ssim_db"]) for r in bucket])),
                "true_ms_ssim_db_min": float(np.min([float(r["true_ms_ssim_db"]) for r in bucket])),
                "true_ms_ssim_db_max": float(np.max([float(r["true_ms_ssim_db"]) for r in bucket])),
            }
        )
    return out


def v3c_rows(rows: list[dict]) -> list[dict]:
    out = [r for r in rows if r["setting_label"] == "v3c_ref"]
    return sorted(out, key=lambda r: int(r["target_sinr_pre"]))


def plot_metric(mean_rows: list[dict], v3c: list[dict], metric: str, out_name: str, title: str) -> None:
    plt.figure(figsize=(8.8, 6.0))
    if metric == "true_ms_ssim_db":
        plt.ylabel("True MS-SSIM-dB", fontsize=11)
    else:
        plt.ylabel("Image PSNR (dB)", fontsize=11)
    style_map = {
        "trackb_ref_q3_midmid_crop": ("P", "--", 2.4, 9),
        "precomp_analog": ("v", "-", 2.0, 7),
        "precomp_digital": ("s", "-", 2.0, 7),
        "precomp_diffusion": ("*", "-", 2.6, 11),
    }
    for run_tag, label, color in SERIES:
        if run_tag == "trackb_ref_q3_midmid_crop":
            pts = v3c
            xs = [int(r["target_sinr_pre"]) for r in pts]
            ys = [float(r["true_ms_ssim_db"] if metric == "true_ms_ssim_db" else r["psnr"]) for r in pts]
        else:
            pts = [r for r in mean_rows if r["run_tag"] == run_tag]
            pts = sorted(pts, key=lambda r: int(r["target_sinr_pre"]))
            xs = [int(r["target_sinr_pre"]) for r in pts]
            if metric == "true_ms_ssim_db":
                ys = [r["true_ms_ssim_db"] for r in pts]
                ymin = [r["true_ms_ssim_db_min"] for r in pts]
                ymax = [r["true_ms_ssim_db_max"] for r in pts]
            else:
                ys = [r["psnr"] for r in pts]
                ymin = [r["psnr_min"] for r in pts]
                ymax = [r["psnr_max"] for r in pts]
            plt.fill_between(xs, ymin, ymax, color=color, alpha=0.10)
        marker, ls, lw, ms = style_map[run_tag]
        plt.plot(xs, ys, label=label, color=color, marker=marker, linestyle=ls, linewidth=lw, markersize=ms)
    plt.xlabel("Target Pre-Cancellation SINR (dB)", fontsize=11)
    plt.xticks(TARGETS)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, ls="--", alpha=0.45)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_FIG / out_name, dpi=300)
    plt.close()


def representative_row(rows: list[dict], target: int, run_tag: str) -> dict:
    matches = [r for r in rows if r["setting_label"] == "weak1700_seed2025" and int(r["target_sinr_pre"]) == target and r["run_tag"] == run_tag]
    if run_tag == "precomp_diffusion" and target == -35:
        matches = [r for r in matches if r["variant"] == "t2"]
    else:
        matches = [r for r in matches if r["variant"] == "baseline"]
    return matches[0]


def visual_grid(rows: list[dict], metric_key: str, metric_label: str, out_name: str) -> None:
    fig = plt.figure(figsize=(1.25 + len(TARGETS) * 1.55 + 1.55, 0.68 + 4 * 1.62), dpi=220, facecolor="white")
    gs = GridSpec(5, len(TARGETS) + 2, figure=fig, height_ratios=[0.48] + [1.55] * 4, width_ratios=[1.25] + [1.55] * len(TARGETS) + [1.55], hspace=0.055, wspace=0.04, left=0.012, right=0.995, top=0.91, bottom=0.025)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    for ci, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, ci + 1]); ax.set_facecolor("#2d2d2d"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(0.5, 0.5, f"{target:.1f} dB", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax = fig.add_subplot(gs[0, len(TARGETS) + 1]); ax.set_facecolor("#11134a"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.text(0.5, 0.5, "Original", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", transform=ax.transAxes)

    series = SERIES
    original = np.asarray(Image.open(SOURCE / "weak1700_seed2025" / "sinr-60dB_precomp_analog" / "img_original_remote.png").convert("RGB"))
    for ri, (run_tag, label, color) in enumerate(series):
        ax_label = fig.add_subplot(gs[ri + 1, 0]); ax_label.set_facecolor(color); ax_label.set_xticks([]); ax_label.set_yticks([])
        for spine in ax_label.spines.values(): spine.set_visible(False)
        ax_label.text(0.5, 0.5, label, ha="center", va="center", color="white", fontsize=8.0, fontweight="bold", transform=ax_label.transAxes, wrap=True)
        for ci, target in enumerate(TARGETS):
            ax_img = fig.add_subplot(gs[ri + 1, ci + 1]); ax_img.axis("off")
            if run_tag == "trackb_ref_q3_midmid_crop":
                row = next(r for r in rows if r["setting_label"] == "v3c_ref" and int(r["target_sinr_pre"]) == target and r["run_tag"] == run_tag)
                run_dir = SOURCE / "v3c_ref" / f"sinr{target:+d}dB_{run_tag}"
            else:
                row = representative_row(rows, target, run_tag)
                suffix = "_t2" if run_tag == "precomp_diffusion" and target == -35 else ""
                run_dir = SOURCE / "weak1700_seed2025" / f"sinr{target:+d}dB_{run_tag}{suffix}"
            img = np.asarray(Image.open(run_dir / "img_recon_remote.png").convert("RGB"))
            ax_img.imshow(img)
            ax_img.text(0.5, 0.03, f"{metric_label} {float(row[metric_key]):.2f} dB", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="white", transform=ax_img.transAxes, bbox=dict(boxstyle="round,pad=0.22", facecolor=BADGE_GREEN, alpha=0.88, edgecolor="none"))
            rect = mpatches.Rectangle((0, 0), 0.018, 1, transform=ax_img.transAxes, clip_on=True, facecolor=color, edgecolor="none", alpha=0.95)
            ax_img.add_patch(rect)
        ax_orig = fig.add_subplot(gs[ri + 1, len(TARGETS) + 1]); ax_orig.axis("off"); ax_orig.imshow(original)
    fig.suptitle("Reconstruction Quality Comparison", fontsize=12.5, fontweight="bold", y=0.985)
    fig.savefig(OUT_FIG / out_name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_notes(rows: list[dict], mean_rows: list[dict], v3c: list[dict]) -> None:
    lines = [
        "# WINT 0506 Summary",
        "",
        "| target_sinr_pre | analog | mp | diff | v3c | diff-mp | diff-v3c | true_msssim diff-mp |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        analog = next(r for r in mean_rows if r["target_sinr_pre"] == target and r["run_tag"] == "precomp_analog")
        mp = next(r for r in mean_rows if r["target_sinr_pre"] == target and r["run_tag"] == "precomp_digital")
        diff = next(r for r in mean_rows if r["target_sinr_pre"] == target and r["run_tag"] == "precomp_diffusion")
        v3 = next(r for r in v3c if int(r["target_sinr_pre"]) == target)
        lines.append(f"| {target:+d} | {analog['psnr']:.2f} | {mp['psnr']:.2f} | {diff['psnr']:.2f} | {float(v3['psnr']):.2f} | {diff['psnr']-mp['psnr']:+.2f} | {diff['psnr']-float(v3['psnr']):+.2f} | {diff['true_ms_ssim_db']-mp['true_ms_ssim_db']:+.2f} |")
    (OUT_NOTES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "raw_source": str(SOURCE),
        "target_sinr_pre": TARGETS,
        "repair_rule": "Use T_START=2 for weak1700 diffusion at -35 dB only; all other weak points remain baseline.",
        "weak_seeds": [2025, 2026, 2027],
    }
    (OUT_NOTES / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme() -> None:
    text = """# WINT 0506 Mainline Bundle

Weak1700 integer-SINR mainline bundle.

- x-axis targets: `-60, -55, -50, -45, -40, -35, -30 dB`
- weak realistic branch: 3-seed mean (`2025/2026/2027`)
- repair rule: Diffusion uses `T=2` only at `-35 dB`
- fourth line: `V3c` surrogate

Contents:
- 4-line PSNR / True MS-SSIM plots
- 4-row visual grids with uniform green metric badges
- summary tables and metadata
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = read_rows()
    ensure_dirs()
    shutil.copy2(SOURCE / "rows.json", OUT_TAB / "rows.json")
    mean_rows = select_mean_rows(rows)
    v3c = v3c_rows(rows)
    plot_metric(mean_rows, v3c, "psnr", "wint_0506_psnr.png", "PSNR Comparison")
    plot_metric(mean_rows, v3c, "true_ms_ssim_db", "wint_0506_msssim_db.png", "MS-SSIM Comparison")
    visual_grid(rows, "psnr", "PSNR", "wint_0506_visual_grid_psnr.png")
    visual_grid(rows, "true_ms_ssim_db", "MS-SSIM", "wint_0506_visual_grid_msssim.png")
    write_notes(rows, mean_rows, v3c)
    write_readme()
    print(f"[OK] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
