from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parent.parent
SRC_MAIN = REPO / "results_wint_0506" / "tables" / "rows.json"
SRC_CAT = REPO / "results_wint_0506_cat" / "tables" / "rows.json"
OUT_ROOT = REPO / "results_wint_audit_0506"
OUT_FIG = OUT_ROOT / "figures"
OUT_NOTES = OUT_ROOT / "notes"
TARGETS = [-60, -55, -50, -45, -40, -35, -30]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_NOTES.mkdir(parents=True, exist_ok=True)


def pick_main_rows(rows: list[dict]) -> list[dict]:
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


def mean_main_realistic(rows: list[dict]) -> list[dict]:
    selected = pick_main_rows(rows)
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in selected:
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
                "sinr_after_analog": sum(float(r["sinr_after_analog"]) for r in bucket) / len(bucket),
                "psnr": sum(float(r["psnr"]) for r in bucket) / len(bucket),
                "true_ms_ssim_db": sum(float(r["true_ms_ssim_db"]) for r in bucket) / len(bucket),
            }
        )
    return out


def main_v3c(rows: list[dict]) -> list[dict]:
    selected = pick_main_rows(rows)
    out = [r for r in selected if r.get("setting_label") == "v3c_ref"]
    return sorted(out, key=lambda r: int(r["target_sinr_pre"]))


def cat_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        target = int(row["target_sinr_pre"])
        if row["run_tag"] == "precomp_diffusion" and target == -35:
            if row["variant"] == "t2":
                out.append(row)
        elif row["variant"] == "baseline":
            out.append(row)
    return sorted(out, key=lambda r: (int(r["target_sinr_pre"]), r["run_tag"]))


def compute_offsets(realistic_rows: list[dict], v3c_rows: list[dict]) -> list[dict]:
    out = []
    for target in TARGETS:
        r_analog = next(r for r in realistic_rows if int(r["target_sinr_pre"]) == target and r["run_tag"] == "precomp_analog")
        v3c = next(r for r in v3c_rows if int(r["target_sinr_pre"]) == target)
        out.append(
            {
                "target_sinr_pre": target,
                "realistic_sinr_after_analog": float(r_analog["sinr_after_analog"]),
                "v3c_sinr_after_analog": float(v3c["sinr_after_analog"]),
                "offset_db": float(v3c["sinr_after_analog"]) - float(r_analog["sinr_after_analog"]),
            }
        )
    return out


def plot_reaxis(realistic_rows: list[dict], v3c_rows: list[dict], metric_key: str, title: str, out_name: str) -> None:
    plt.figure(figsize=(8.8, 6.0))
    styles = {
        "precomp_analog": ("Analog Only", "#6f6f6f", "v", 2.0, 7),
        "precomp_digital": ("MP", "#d4720a", "s", 2.0, 7),
        "precomp_diffusion": ("Diffusion", "#1a6bb5", "*", 2.6, 11),
    }
    for run_tag, (label, color, marker, lw, ms) in styles.items():
        pts = [r for r in realistic_rows if r["run_tag"] == run_tag]
        pts = sorted(pts, key=lambda r: float(r["sinr_after_analog"]))
        xs = [float(r["sinr_after_analog"]) for r in pts]
        ys = [float(r[metric_key]) for r in pts]
        plt.plot(xs, ys, label=label, color=color, marker=marker, linewidth=lw, markersize=ms)

    vpts = sorted(v3c_rows, key=lambda r: float(r["sinr_after_analog"]))
    vxs = [float(r["sinr_after_analog"]) for r in vpts]
    vys = [float(r[metric_key]) for r in vpts]
    plt.plot(vxs, vys, label="V3c Surrogate", color="#c1121f", marker="P", linestyle="--", linewidth=2.4, markersize=9)

    plt.xlabel("Actual Post-Analog SINR (dB)", fontsize=11)
    plt.ylabel("True MS-SSIM-dB" if metric_key == "true_ms_ssim_db" else "Image PSNR (dB)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(True, ls="--", alpha=0.45)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(OUT_FIG / out_name, dpi=300)
    plt.close()


def write_summary(main_offsets: list[dict], cat_offsets: list[dict], main_rows: list[dict], cat_rows_selected: list[dict], main_v3c_rows: list[dict], cat_v3c_rows: list[dict]) -> None:
    lines = [
        "# WINT 0506 Fairness Audit",
        "",
        "This audit re-reads the existing `rows.json` files and compares the V3c surrogate against the realistic lines using actual `sinr_after_analog` instead of nominal target bins.",
        "",
        "## Main Offsets",
        "",
        "| target_sinr_pre | realistic sinr_after_analog | v3c sinr_after_analog | v3c offset |",
        "|---:|---:|---:|---:|",
    ]
    for row in main_offsets:
        lines.append(f"| {row['target_sinr_pre']:+d} | {row['realistic_sinr_after_analog']:.2f} | {row['v3c_sinr_after_analog']:.2f} | {row['offset_db']:+.2f} |")
    lines += [
        "",
        "## Cat Offsets",
        "",
        "| target_sinr_pre | realistic sinr_after_analog | v3c sinr_after_analog | v3c offset |",
        "|---:|---:|---:|---:|",
    ]
    for row in cat_offsets:
        lines.append(f"| {row['target_sinr_pre']:+d} | {row['realistic_sinr_after_analog']:.2f} | {row['v3c_sinr_after_analog']:.2f} | {row['offset_db']:+.2f} |")

    def line_gap(rows_real, rows_v3, target):
        diff = next(r for r in rows_real if int(r["target_sinr_pre"]) == target and r["run_tag"] == "precomp_diffusion")
        mp = next(r for r in rows_real if int(r["target_sinr_pre"]) == target and r["run_tag"] == "precomp_digital")
        v3 = next(r for r in rows_v3 if int(r["target_sinr_pre"]) == target)
        return float(diff["psnr"]) - float(mp["psnr"]), float(diff["psnr"]) - float(v3["psnr"])

    lines += [
        "",
        "## Interpretation",
        "",
    ]
    m_dm45, m_dv45 = line_gap(main_rows, main_v3c_rows, -45)
    c_dm45, c_dv45 = line_gap(cat_rows_selected, cat_v3c_rows, -45)
    lines.append(f"- At target -45 dB, main realistic Diff-MP = {m_dm45:+.2f} dB while Diff-V3c = {m_dv45:+.2f} dB.")
    lines.append(f"- At target -45 dB, cat realistic Diff-MP = {c_dm45:+.2f} dB while Diff-V3c = {c_dv45:+.2f} dB.")
    lines.append("- The realistic three-line comparisons remain internally fair because they share the same actual post-analog operating point.")
    lines.append("- The V3c surrogate is not directly fair at the same nominal x-bin whenever its actual `sinr_after_analog` is shifted relative to the realistic lines.")
    (OUT_NOTES / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme() -> None:
    text = """# WINT 0506 Audit

Fairness audit for `results_wint_0506` and `results_wint_0506_cat`.

This folder does not contain new experiments.
It only re-reads the existing `rows.json` files and re-plots them on the actual
post-analog SINR axis to inspect whether the V3c surrogate is being compared
under the same operating point as the realistic lines.
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rows_main = load_json(SRC_MAIN)
    rows_cat = load_json(SRC_CAT)

    main_real = mean_main_realistic(rows_main)
    main_v = main_v3c(rows_main)
    cat_sel = cat_rows(rows_cat)
    cat_real = [r for r in cat_sel if r["run_tag"] != "trackb_ref_q3_midmid_crop"]
    cat_v = [r for r in cat_sel if r["run_tag"] == "trackb_ref_q3_midmid_crop"]

    main_offsets = compute_offsets(main_real, main_v)
    cat_offsets = compute_offsets(cat_real, cat_v)

    plot_reaxis(main_real, main_v, "psnr", "WINT 0506 Mainline Re-Axis Audit (PSNR)", "wint_0506_reaxis_psnr.png")
    plot_reaxis(main_real, main_v, "true_ms_ssim_db", "WINT 0506 Mainline Re-Axis Audit (True MS-SSIM)", "wint_0506_reaxis_msssim.png")
    plot_reaxis(cat_real, cat_v, "psnr", "WINT 0506 Cat Re-Axis Audit (PSNR)", "wint_0506_cat_reaxis_psnr.png")
    plot_reaxis(cat_real, cat_v, "true_ms_ssim_db", "WINT 0506 Cat Re-Axis Audit (True MS-SSIM)", "wint_0506_cat_reaxis_msssim.png")

    (OUT_NOTES / "main_offsets.json").write_text(json.dumps(main_offsets, indent=2), encoding="utf-8")
    (OUT_NOTES / "cat_offsets.json").write_text(json.dumps(cat_offsets, indent=2), encoding="utf-8")
    write_summary(main_offsets, cat_offsets, main_real, cat_sel, main_v, cat_v)
    write_readme()
    print(f"[OK] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
