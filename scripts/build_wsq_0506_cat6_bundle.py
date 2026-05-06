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
SOURCE = REPO / "results_probe_wsq_0506_cat6"
OUT_ROOT = REPO / "results_wsq_0506_cat6"
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


def selected_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        target = int(row["target_sinr_pre"])
        if row["run_tag"] == "precomp_diffusion" and target == -35:
            if row["variant"] != "t2":
                continue
        elif row["variant"] != "baseline":
            continue
        out.append(row)
    return sorted(out, key=lambda r: (int(r["target_sinr_pre"]), r["run_tag"]))


def run_dir_for(row: dict) -> Path:
    target = int(row["target_sinr_pre"])
    suffix = "_t2" if row["run_tag"] == "precomp_diffusion" and target == -35 else ""
    return SOURCE / f"sinr{target:+d}dB_{row['run_tag']}{suffix}"


def plot_metric(rows: list[dict], metric: str, out_name: str, title: str) -> None:
    plt.figure(figsize=(8.8, 6.0))
    for run_tag, label, color in SERIES:
        pts = [r for r in rows if r["run_tag"] == run_tag]
        pts = sorted(pts, key=lambda r: int(r["target_sinr_pre"]))
        xs = [int(r["target_sinr_pre"]) for r in pts]
        ys = [float(r["true_ms_ssim_db"] if metric == "true_ms_ssim_db" else r["psnr"]) for r in pts]
        marker = {
            "precomp_analog": "v",
            "precomp_digital": "s",
            "precomp_diffusion": "*",
            "trackb_ref_q3_midmid_crop": "P",
        }[run_tag]
        lw = 2.0 if run_tag != "precomp_diffusion" else 2.6
        ms = 7 if run_tag != "precomp_diffusion" else 11
        plt.plot(xs, ys, label=label, color=color, marker=marker, linewidth=lw, markersize=ms)
    plt.xlabel("Target Pre-Cancellation SINR (dB)", fontsize=11)
    plt.xticks(TARGETS)
    plt.ylabel("True MS-SSIM-dB" if metric == "true_ms_ssim_db" else "Image PSNR (dB)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, ls="--", alpha=0.45)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_FIG / out_name, dpi=300)
    plt.close()


def visual_grid(rows: list[dict], metric_key: str, metric_label: str, out_name: str) -> None:
    analog_rows = [r for r in rows if r["run_tag"] == "precomp_analog"]
    analog_rows = sorted(analog_rows, key=lambda r: int(r["target_sinr_pre"]))
    original = np.asarray(Image.open(run_dir_for(analog_rows[0]) / "img_original_remote.png").convert("RGB"))
    fig = plt.figure(figsize=(1.25 + len(TARGETS) * 1.55 + 1.55, 0.68 + len(SERIES) * 1.62), dpi=220, facecolor="white")
    gs = GridSpec(len(SERIES) + 1, len(TARGETS) + 2, figure=fig, height_ratios=[0.48] + [1.55] * len(SERIES), width_ratios=[1.25] + [1.55] * len(TARGETS) + [1.55], hspace=0.055, wspace=0.04, left=0.012, right=0.995, top=0.91, bottom=0.025)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    for ci, target in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, ci + 1]); ax.set_facecolor("#2d2d2d"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(0.5, 0.5, f"{target:.1f} dB", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    ax = fig.add_subplot(gs[0, len(TARGETS) + 1]); ax.set_facecolor("#11134a"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.text(0.5, 0.5, "Original", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold", transform=ax.transAxes)
    for ri, (run_tag, label, color) in enumerate(SERIES):
        ax_label = fig.add_subplot(gs[ri + 1, 0]); ax_label.set_facecolor(color); ax_label.set_xticks([]); ax_label.set_yticks([])
        for spine in ax_label.spines.values(): spine.set_visible(False)
        ax_label.text(0.5, 0.5, label, ha="center", va="center", color="white", fontsize=8.0, fontweight="bold", transform=ax_label.transAxes, wrap=True)
        for ci, target in enumerate(TARGETS):
            ax_img = fig.add_subplot(gs[ri + 1, ci + 1]); ax_img.axis("off")
            row = next(r for r in rows if int(r["target_sinr_pre"]) == target and r["run_tag"] == run_tag)
            img = np.asarray(Image.open(run_dir_for(row) / "img_recon_remote.png").convert("RGB"))
            ax_img.imshow(img)
            ax_img.text(0.5, 0.03, f"{metric_label} {float(row[metric_key]):.2f} dB", ha="center", va="bottom", fontsize=6.5, fontweight="bold", color="white", transform=ax_img.transAxes, bbox=dict(boxstyle="round,pad=0.22", facecolor=BADGE_GREEN, alpha=0.88, edgecolor="none"))
            rect = mpatches.Rectangle((0, 0), 0.018, 1, transform=ax_img.transAxes, clip_on=True, facecolor=color, edgecolor="none", alpha=0.95)
            ax_img.add_patch(rect)
        ax_orig = fig.add_subplot(gs[ri + 1, len(TARGETS) + 1]); ax_orig.axis("off"); ax_orig.imshow(original)
    fig.suptitle("Reconstruction Quality Comparison", fontsize=12.5, fontweight="bold", y=0.985)
    fig.savefig(OUT_FIG / out_name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_notes(rows: list[dict]) -> None:
    lines = [
        "# WSQ 0506 Cat6 Summary",
        "",
        "| target_sinr_pre | actual_sinr_pre | analog | mp | diff | v3c | diff-mp | diff-v3c | true_msssim diff-mp |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        analog = next(r for r in rows if int(r["target_sinr_pre"]) == target and r["run_tag"] == "precomp_analog")
        mp = next(r for r in rows if int(r["target_sinr_pre"]) == target and r["run_tag"] == "precomp_digital")
        diff = next(r for r in rows if int(r["target_sinr_pre"]) == target and r["run_tag"] == "precomp_diffusion")
        v3c = next(r for r in rows if int(r["target_sinr_pre"]) == target and r["run_tag"] == "trackb_ref_q3_midmid_crop")
        lines.append(f"| {target:+d} | {float(analog['sinr_pre']):.2f} | {float(analog['psnr']):.2f} | {float(mp['psnr']):.2f} | {float(diff['psnr']):.2f} | {float(v3c['psnr']):.2f} | {float(diff['psnr'])-float(mp['psnr']):+.2f} | {float(diff['psnr'])-float(v3c['psnr']):+.2f} | {float(diff['true_ms_ssim_db'])-float(mp['true_ms_ssim_db']):+.2f} |")
    (OUT_NOTES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta = {
        "raw_source": str(SOURCE),
        "local_img": rows[0]["local_img"],
        "remote_img": rows[0]["remote_img"],
        "remote_presentation_fix": "cat6 was center-cropped offline to a square before the standard resize_128 protocol.",
        "repair_rule": "Use T_START=2 for weak1700 diffusion at -35 dB only; all other cat6 weak points remain baseline.",
    }
    (OUT_NOTES / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_readme() -> None:
    text = """# WSQ 0506 Cat6 Bundle

Weak1700 integer-SINR cat6 bundle using a square center-cropped remote image.

- local: `kodim12`
- remote: `cat6_center_square.png`
- base protocol: `resize_128`
- x-axis targets: `-60, -55, -50, -45, -40, -35, -30 dB`
- repair rule: Diffusion uses `T=2` only at `-35 dB`
- fourth line: `V3c` surrogate
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = selected_rows(read_rows())
    ensure_dirs()
    shutil.copy2(SOURCE / "rows.json", OUT_TAB / "rows.json")
    plot_metric(rows, "psnr", "wsq_0506_cat6_psnr.png", "PSNR Comparison")
    plot_metric(rows, "true_ms_ssim_db", "wsq_0506_cat6_msssim_db.png", "MS-SSIM Comparison")
    visual_grid(rows, "psnr", "PSNR", "wsq_0506_cat6_visual_grid_psnr.png")
    visual_grid(rows, "true_ms_ssim_db", "MS-SSIM", "wsq_0506_cat6_visual_grid_msssim.png")
    write_notes(rows)
    write_readme()
    print(f"[OK] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
