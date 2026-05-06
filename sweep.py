#!/usr/bin/env python3
"""
sweep.py - SDD main comparison sweep

Current ai_comp mode keeps the four main report lines under a unified Aux-TX
analog condition:
  1. precomp_analog     - Aux-TX realistic + Analog Only
  2. precomp_digital    - Aux-TX realistic + MP
  3. precomp_hammer_v2  - Aux-TX realistic + HW-Aware Hammerstein
  4. precomp_diffusion  - Aux-TX realistic + Diffusion (Ours)

fig4_ref mode is a lighter three-line reference sweep for the new
SDD-paper-Fig.4-like task:
  1. precomp_analog     - Aux-TX realistic + Analog Only
  2. precomp_digital    - Aux-TX realistic + MP
  3. precomp_diffusion  - Aux-TX realistic + Diffusion (Ours)

fig4_sweep mode is the final three-line SSIM-style sweep:
  1. precomp_analog     - Aux-TX realistic + Analog Only
  2. precomp_diffusion  - Aux-TX realistic + Diffusion (Ours)
  3. trackb_ref_q2      - Track B reference (toy analog + MP + NTSCC q2)

fig4_sweep can also be pointed at a second Track B variant:
  3. trackb_ref_q3_prefix - Track B shape-oriented variant
                             (toy analog + MP prefix-fit + NTSCC q3 HR)
  3. trackb_ref_q3_midmid - Track B morphology-balanced variant
                             (toy analog + MP legacy-fit + NTSCC q3 HR, L=3, lambda=1, orders=1)
  3. trackb_ref_q3_midmid_crop - V3c crop-aligned variant
                                 (toy analog + MP legacy-fit + NTSCC q3 resize_128, L=3, lambda=1, orders=1)

ai_comp_v3c keeps the original four AI comparison lines, but runs them
under the V3c fair protocol:
  - resize_128
  - NTSCC quality_3
  - V3c anchor RSI grid
"""

import argparse
import json
import os
import time
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DEFAULT_LOCAL  = "data/kodak/kodim01.png"
DEFAULT_REMOTE = "data/kodak/kodim24.png"
RSI_SCALE_LIST = [10000, 50000, 100000, 250000, 1000000, 3150000, 10000000]
FIG4_RSI_SCALE_LIST = [3000, 5000, 10000, 50000, 100000, 250000, 1000000, 3150000, 10000000]
FIG4_SWEEP_RSI_SCALE_LIST = [5000, 10000, 50000, 100000, 250000, 500000, 1000000, 2000000, 3150000]
V3C_ANCHOR_RSI_SCALE_LIST = [3413000, 1079000, 341300, 107900, 34130, 10790, 3413]


def get_cmd_args(run_tag, fair_fit=False, prefix_calibration=False, hv2_prefix_only=False,
                 mp_prefix_calibration=False, prefix_samples=2048,
                 digital_internal_normalize=False, realistic_protocol_align=False,
                 realistic_q3_resize128_align=False):
    if fair_fit and (prefix_calibration or hv2_prefix_only):
        raise ValueError("fair_fit cannot be combined with prefix-calibration modes.")
    if prefix_calibration and hv2_prefix_only:
        raise ValueError("prefix_calibration and hv2_prefix_only cannot be enabled at the same time.")

    hr_q3_args = [
        '--ntscc-ckpt', 'checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_3_psnr.pth',
        '--kodak-protocol', 'witt_hr_center_crop',
    ]
    q3_resize128_args = [
        '--ntscc-ckpt', 'checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_3_psnr.pth',
    ]

    if run_tag == 'ideal_analog':
        return ['--no-digital-sic', '--aux-disable-iqpa', 'True']
    elif run_tag == 'precomp_analog':
        args = ['--no-digital-sic', '--aux-disable-iqpa', 'False']
        if realistic_q3_resize128_align:
            args += q3_resize128_args
        elif realistic_protocol_align:
            args += hr_q3_args
        return args
    elif run_tag == 'ideal_digital':
        return ['--backend', 'mp', '--aux-disable-iqpa', 'True']
    elif run_tag == 'precomp_digital':
        args = ['--backend', 'mp', '--aux-disable-iqpa', 'False']
        if prefix_calibration or mp_prefix_calibration:
            args += ['--fit-target', 'si_oracle', '--fit-window', 'prefix', '--fit-prefix-samples', str(prefix_samples)]
        elif fair_fit:
            args += ['--fit-target', 'observation']
        if digital_internal_normalize:
            args += ['--digital-internal-normalize']
        if realistic_q3_resize128_align:
            args += q3_resize128_args
        elif realistic_protocol_align:
            args += hr_q3_args
        return args
    elif run_tag == 'precomp_hammer_v2':
        args = ['--backend', 'hammerstein_v2', '--poly-orders', '1,5,9', '--aux-disable-iqpa', 'False']
        if prefix_calibration or hv2_prefix_only:
            args += ['--fit-target', 'si_oracle', '--fit-window', 'prefix', '--fit-prefix-samples', str(prefix_samples)]
        elif fair_fit:
            args += ['--fit-target', 'observation']
        if digital_internal_normalize:
            args += ['--digital-internal-normalize']
        if realistic_q3_resize128_align:
            args += q3_resize128_args
        elif realistic_protocol_align:
            args += hr_q3_args
        return args
    elif run_tag == 'precomp_kong':
        return ['--use-kong', '--aux-disable-iqpa', 'False']
    elif run_tag == 'ideal_diffusion':
        return ['--use-diffusion', '--aux-disable-iqpa', 'True']
    elif run_tag == 'precomp_diffusion':
        args = ['--use-diffusion', '--aux-disable-iqpa', 'False']
        if realistic_q3_resize128_align:
            args += q3_resize128_args
        elif realistic_protocol_align:
            args += hr_q3_args
        return args
    elif run_tag == 'toy_digital':
        args = ['--toy-analog', '--backend', 'mp']
        if fair_fit:
            args += ['--fit-target', 'observation']
        return args
    elif run_tag == 'toy_kong':
        return ['--toy-analog', '--use-kong']
    elif run_tag == 'old_sdd_digital':
        args = ['--old-sdd', '--backend', 'mp']
        if fair_fit:
            args += ['--fit-target', 'observation']
        return args
    elif run_tag == 'trackb_ref_q2':
        return [
            '--toy-analog',
            '--backend', 'mp',
            '--ntscc-ckpt', 'checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_2_psnr.pth',
        ]
    elif run_tag == 'trackb_ref_q3_prefix':
        return [
            '--toy-analog',
            '--backend', 'mp',
            '--ntscc-ckpt', 'checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_3_psnr.pth',
            '--kodak-protocol', 'witt_hr_center_crop',
            '--fit-target', 'si_oracle',
            '--fit-window', 'prefix',
            '--fit-prefix-samples', '1024',
        ]
    elif run_tag == 'trackb_ref_q3_midmid':
        return [
            '--toy-analog',
            '--backend', 'mp',
            '--ntscc-ckpt', 'checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_3_psnr.pth',
            '--kodak-protocol', 'witt_hr_center_crop',
            '--fit-target', 'si_oracle',
            '--fit-window', 'legacy',
            '--fit-prefix-samples', '1024',
            '--L', '3',
            '--lambda-reg', '1.0',
            '--poly-orders', '1',
        ]
    elif run_tag == 'trackb_ref_q3_midmid_crop':
        return [
            '--toy-analog',
            '--backend', 'mp',
            '--ntscc-ckpt', 'checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_3_psnr.pth',
            '--fit-target', 'si_oracle',
            '--fit-window', 'legacy',
            '--fit-prefix-samples', '1024',
            '--L', '3',
            '--lambda-reg', '1.0',
            '--poly-orders', '1',
        ]
    else:
        raise ValueError(f"Unknown run_tag: {run_tag}")


def safe_float(v):
    if v is None or isinstance(v, str):
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


def collect_image_pairs(local_dir, remote_dir, max_pairs=None):
    local_dir  = Path(local_dir)
    remote_dir = Path(remote_dir)
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    local_files  = sorted(p for p in local_dir.iterdir()  if p.suffix.lower() in exts)
    remote_files = sorted(p for p in remote_dir.iterdir() if p.suffix.lower() in exts)

    if not local_files:  raise FileNotFoundError(f"找不到圖片：{local_dir}")
    if not remote_files: raise FileNotFoundError(f"找不到圖片：{remote_dir}")

    lmap = {p.name: p for p in local_files}
    rmap = {p.name: p for p in remote_files}
    common = sorted(lmap.keys() & rmap.keys())

    if len(common) >= 2:
        pairs = [(str(lmap[n]), str(rmap[n])) for n in common]
    else:
        n     = min(len(local_files), len(remote_files))
        pairs = [(str(local_files[i]), str(remote_files[i])) for i in range(n)]

    return pairs[:max_pairs] if max_pairs else pairs


def run_one(rsi_scale, run_tag, local_img, remote_img, out_dir, fair_fit=False,
            prefix_calibration=False, hv2_prefix_only=False, mp_prefix_calibration=False,
            prefix_samples=2048, use_no_normalize=False,
            digital_internal_normalize=False, realistic_protocol_align=False,
            realistic_q3_resize128_align=False, diffusion_t_start=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "run_sdd_final.py",
        "--local",     local_img,
        "--remote",    remote_img,
        "--rsi-scale", str(rsi_scale),
    ] + get_cmd_args(
        run_tag,
        fair_fit=fair_fit,
        prefix_calibration=prefix_calibration,
        hv2_prefix_only=hv2_prefix_only,
        mp_prefix_calibration=mp_prefix_calibration,
        prefix_samples=prefix_samples,
        digital_internal_normalize=digital_internal_normalize,
        realistic_protocol_align=realistic_protocol_align,
        realistic_q3_resize128_align=realistic_q3_resize128_align,
    )

    if use_no_normalize:
        cmd.append("--no-normalize")

    with open(out_dir / "run_stdout.txt", "w") as f_out:
        t0  = time.time()
        env = None
        if diffusion_t_start is not None and "diffusion" in run_tag:
            env = dict(os.environ)
            env["T_START"] = str(diffusion_t_start)
        res = subprocess.run(cmd, stdout=f_out, stderr=subprocess.STDOUT, text=True, env=env)
        t1  = time.time()

    row = {
        "rsi_scale": rsi_scale, "run_tag": run_tag,
        "local_img": local_img, "remote_img": remote_img,
        "time_sec": t1 - t0,
        "sinr_pre": np.nan, "sinr_after_analog": np.nan,
        "sinr_after_digital": np.nan, "psnr": np.nan, "ms_ssim": np.nan,
    }

    if res.returncode != 0:
        print(f"    [ERROR] see {out_dir / 'run_stdout.txt'}")
        row["returncode"] = res.returncode
        with open(out_dir / "summary.json", "w") as f:
            json.dump(row, f, indent=2,
                      default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
        return row

    try:
        with open("bridge/meta.json") as f: am = json.load(f)
        row["sinr_pre"]          = safe_float(am.get("SINR_pre"))
        row["sinr_after_analog"] = safe_float(am.get("SINR_analog"))
    except Exception: pass

    try:
        with open("bridge_digital/metrics.json") as f: dm = json.load(f)
        row["sinr_after_digital"] = safe_float(dm.get("SINR_after_digital"))
    except Exception: pass

    try:
        with open("bridge_rx/metrics_remote.json") as f: rm = json.load(f)
        row["psnr"] = safe_float(rm.get("psnr"))
        row["ms_ssim"] = safe_float(rm.get("ms_ssim"))
    except Exception: pass

    for src, dst in [
        ("bridge/meta.json",                 "meta_analog.json"),
        ("bridge_digital/metrics.json",       "metrics_digital.json"),
        ("bridge_rx/metrics_remote.json",     "metrics_rx.json"),
        ("bridge_rx/img_recon_remote.png",    "img_recon_remote.png"),
        ("bridge_rx/img_original_remote.png", "img_original_remote.png"),
    ]:
        if Path(src).exists(): shutil.copy(src, out_dir / dst)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(row, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    return row


def load_existing_row(out_dir):
    p = out_dir / "summary.json"
    if p.exists():
        try:
            with open(p) as f: return json.load(f)
        except Exception: return None
    return None


def run_pairs_for_scale(rsi_scale, run_tag, pairs, out_root, force_rerun, fair_fit=False,
                        prefix_calibration=False, hv2_prefix_only=False,
                        mp_prefix_calibration=False, prefix_samples=2048,
                        use_no_normalize=False, digital_internal_normalize=False,
                        realistic_protocol_align=False,
                        realistic_q3_resize128_align=False, diffusion_t_start=None):
    if len(pairs) == 1:
        local_img, remote_img = pairs[0]
        run_name = f"rsi{rsi_scale:g}_{run_tag}"
        out_dir  = out_root / run_name

        if not force_rerun:
            cached = load_existing_row(out_dir)
            if cached is not None:
                print(f"  [CACHED] {run_name}  PSNR={safe_float(cached.get('psnr')):.2f}")
                return cached

        print(f"  [RUN]    {run_name}")
        return run_one(
            rsi_scale, run_tag, local_img, remote_img, out_dir,
            fair_fit=fair_fit, prefix_calibration=prefix_calibration,
            hv2_prefix_only=hv2_prefix_only,
            mp_prefix_calibration=mp_prefix_calibration,
            prefix_samples=prefix_samples, use_no_normalize=use_no_normalize,
            digital_internal_normalize=digital_internal_normalize,
            realistic_protocol_align=realistic_protocol_align,
            realistic_q3_resize128_align=realistic_q3_resize128_align,
            diffusion_t_start=diffusion_t_start,
        )

    psnr_list = []; ms_ssim_list = []; sinr_pres = []; sinr_anas = []; total_time = 0.0

    for idx, (local_img, remote_img) in enumerate(pairs):
        lname = Path(local_img).stem; rname = Path(remote_img).stem
        run_name = f"rsi{rsi_scale:g}_{run_tag}_pair{idx:03d}_{lname}_vs_{rname}"
        out_dir  = out_root / run_name

        if not force_rerun:
            cached = load_existing_row(out_dir)
            if cached is not None:
                print(f"  [CACHED] pair {idx+1}/{len(pairs)}  PSNR={safe_float(cached.get('psnr')):.2f}")
                row = cached
            else:
                print(f"  [RUN]    pair {idx+1}/{len(pairs)}: {lname} vs {rname}")
                row = run_one(
                    rsi_scale, run_tag, local_img, remote_img, out_dir,
                    fair_fit=fair_fit, prefix_calibration=prefix_calibration,
                    hv2_prefix_only=hv2_prefix_only,
                    mp_prefix_calibration=mp_prefix_calibration,
                    prefix_samples=prefix_samples, use_no_normalize=use_no_normalize,
                    digital_internal_normalize=digital_internal_normalize,
                    realistic_protocol_align=realistic_protocol_align,
                    realistic_q3_resize128_align=realistic_q3_resize128_align,
                    diffusion_t_start=diffusion_t_start,
                )
        else:
            print(f"  [RUN]    pair {idx+1}/{len(pairs)}: {lname} vs {rname}")
            row = run_one(
                rsi_scale, run_tag, local_img, remote_img, out_dir,
                fair_fit=fair_fit, prefix_calibration=prefix_calibration,
                hv2_prefix_only=hv2_prefix_only,
                mp_prefix_calibration=mp_prefix_calibration,
                prefix_samples=prefix_samples, use_no_normalize=use_no_normalize,
                digital_internal_normalize=digital_internal_normalize,
                realistic_protocol_align=realistic_protocol_align,
                realistic_q3_resize128_align=realistic_q3_resize128_align,
                diffusion_t_start=diffusion_t_start,
            )

        pv = safe_float(row.get("psnr"))
        if not np.isnan(pv): psnr_list.append(pv)
        sv = safe_float(row.get("ms_ssim"))
        if not np.isnan(sv): ms_ssim_list.append(sv)
        sinr_pres.append(safe_float(row.get("sinr_pre")))
        sinr_anas.append(safe_float(row.get("sinr_after_analog")))
        total_time += safe_float(row.get("time_sec", 0))

    avg_psnr = float(np.nanmean(psnr_list)) if psnr_list else np.nan
    print(f"  → Avg PSNR over {len(pairs)} pairs: {avg_psnr:.2f} dB")

    agg_dir = out_root / f"rsi{rsi_scale:g}_{run_tag}_AGG"
    agg_dir.mkdir(exist_ok=True)
    agg_row = {
        "rsi_scale": rsi_scale, "run_tag": run_tag, "n_pairs": len(pairs),
        "time_sec": total_time,
        "sinr_pre": float(np.nanmean(sinr_pres)),
        "sinr_after_analog": float(np.nanmean(sinr_anas)),
        "sinr_after_digital": np.nan,
        "psnr": avg_psnr,
        "ms_ssim": float(np.nanmean(ms_ssim_list)) if ms_ssim_list else np.nan,
        "psnr_list": psnr_list,
    }
    with open(agg_dir / "summary.json", "w") as f:
        json.dump(agg_row, f, indent=2,
                  default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    return agg_row


def build_sinr_lookup(rows):
    """
    建立 {rsi_scale: sinr_pre} 的查找表，以 precomp_analog 為基準。
    用於將 old_sdd 的 x 座標對齊到相同 rsi_scale 下其他線的 sinr_pre。
    """
    lookup = {}
    for r in rows:
        if r.get("run_tag") == "precomp_analog":
            s = r.get("rsi_scale")
            x = safe_float(r.get("sinr_pre"))
            if s is not None and not np.isnan(x):
                lookup[s] = x
    return lookup


def make_plots(rows, tags, labels, colors, markers, linestyles,
               title_suffix, out_root, mode):
    print("\nGenerating Plots...")

    # 建立 rsi_scale → sinr_pre 對照表（以 precomp_analog 為基準）
    sinr_lookup = build_sinr_lookup(rows)

    def collect_pts(tag, y_key):
        pts = []
        for r in rows:
            if r["run_tag"] == tag:
                if tag == 'old_sdd_digital':
                    x = sinr_lookup.get(r.get("rsi_scale"), np.nan)
                else:
                    x = safe_float(r.get("sinr_pre"))
                y = safe_float(r.get(y_key))
                if not np.isnan(x) and not np.isnan(y):
                    pts.append((x, y))
        pts.sort(key=lambda v: v[0])
        return pts

    if not (mode.startswith("fig4_ref") or mode.startswith("fig4_sweep")):
        # SINR plot（diffusion 無 SINR，跳過）
        plt.figure(figsize=(10, 7))
        for t in tags:
            if 'diffusion' in t:
                continue
            pts = collect_pts(t, "sinr_after_digital")
            if not pts:
                pts = collect_pts(t, "sinr_after_analog")
            if pts:
                x_vals, y_vals = zip(*pts)
                plt.plot(x_vals, y_vals, marker=markers[t], color=colors[t],
                         label=labels[t], linewidth=2.5, markersize=9,
                         linestyle=linestyles.get(t, '-'))

        plt.xlabel("Pre-Cancellation SINR (dB)", fontsize=12)
        plt.ylabel("Final Cleaned SINR before RX (dB)", fontsize=12)
        plt.title(f"Final SINR Capability vs Extreme Interference\n{title_suffix}",
                  fontsize=14, fontweight='bold')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=10)
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(out_root / f"plot_sinr_{mode}.png", dpi=300)
        plt.close()

    def plot_quality(y_key, ylabel, title_prefix, filename, xlim=None,
                     add_psnr_threshold=False):
        plt.figure(figsize=(10, 7))
        for t in tags:
            pts = collect_pts(t, y_key)
            if pts:
                x_vals, y_vals = zip(*pts)
                lw = 3.5 if 'diffusion' in t else 2.5
                ms = 12 if 'diffusion' in t else 9
                plt.plot(x_vals, y_vals, marker=markers[t], color=colors[t],
                         label=labels[t], linewidth=lw, markersize=ms,
                         linestyle=linestyles.get(t, '-'))

        if add_psnr_threshold:
            plt.axhline(y=30.0, color='green', linestyle=':', linewidth=2,
                        label='High Quality Threshold (30 dB)')
        plt.xlabel("Pre-Cancellation SINR (dB)", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{title_prefix}\n{title_suffix}",
                  fontsize=14, fontweight='bold')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=10)
        if xlim is None:
            plt.gca().invert_xaxis()
        else:
            plt.xlim(*xlim)
        plt.tight_layout()
        plt.savefig(out_root / filename, dpi=300)
        plt.close()

    # Default PSNR plot
    plot_quality(
        y_key="psnr",
        ylabel="Image PSNR (dB)",
        title_prefix="End-to-End Image Quality vs Extreme Interference",
        filename=f"plot_psnr_{mode}.png",
        xlim=None,
        add_psnr_threshold=True,
    )

    if not (mode.startswith("fig4_ref") or mode.startswith("fig4_sweep")):
        print(f"  plot_sinr_{mode}.png")
    print(f"  plot_psnr_{mode}.png")

    if mode.startswith("fig4_ref") or mode.startswith("fig4_sweep"):
        plot_quality(
            y_key="psnr",
            ylabel="Image PSNR (dB)",
            title_prefix="End-to-End Image Quality vs Interference (-30 to -65)",
            filename=f"plot_psnr_{mode}_neg30_to_neg65.png",
            xlim=(-30, -65),
            add_psnr_threshold=True,
        )
        plot_quality(
            y_key="psnr",
            ylabel="Image PSNR (dB)",
            title_prefix="End-to-End Image Quality vs Interference (-65 to -30)",
            filename=f"plot_psnr_{mode}_neg65_to_neg30.png",
            xlim=(-65, -30),
            add_psnr_threshold=True,
        )
        # SSIM-like dB plot, using the common transform -10*log10(1-SSIM)
        plt.figure(figsize=(10, 7))
        for t in tags:
            pts = collect_pts(t, "ms_ssim")
            if pts:
                x_vals, raw_y = zip(*pts)
                y_vals = []
                for y in raw_y:
                    y_clip = min(max(float(y), 0.0), 1.0 - 1e-8)
                    y_vals.append(-10.0 * np.log10(1.0 - y_clip))
                lw = 3.5 if 'diffusion' in t else 2.5
                ms = 12 if 'diffusion' in t else 9
                plt.plot(x_vals, y_vals, marker=markers[t], color=colors[t],
                         label=labels[t], linewidth=lw, markersize=ms,
                         linestyle=linestyles.get(t, '-'))
        plt.xlabel("Pre-Cancellation SINR (dB)", fontsize=12)
        plt.ylabel("SSIM-dB Proxy", fontsize=12)
        plt.title(f"SSIM-dB Proxy vs Interference (-65 to -30)\n{title_suffix}",
                  fontsize=14, fontweight='bold')
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(fontsize=10)
        plt.xlim(-65, -30)
        plt.tight_layout()
        plt.savefig(out_root / f"plot_ssim_{mode}_neg65_to_neg30.png", dpi=300)
        plt.close()

        print(f"  plot_psnr_{mode}_neg30_to_neg65.png")
        print(f"  plot_psnr_{mode}_neg65_to_neg30.png")
        print(f"  plot_ssim_{mode}_neg65_to_neg30.png")


def main():
    parser = argparse.ArgumentParser(
        description="SDD 壓力測試腳本（五條比較線版）",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--mode', type=str, required=True,
                        choices=['tf', 'ideal_comp', 'precomp_comp', 'ai_comp', 'ai_comp_v3c', 'fig4_ref', 'fig4_sweep', 'all'])

    src = parser.add_argument_group("圖片來源（三選一）")
    src.add_argument('--local-dir',  type=str, default=None)
    src.add_argument('--remote-dir', type=str, default=None)
    src.add_argument('--local',      type=str, default=None)
    src.add_argument('--remote',     type=str, default=None)

    parser.add_argument('--max-pairs',    type=int,   default=None)
    parser.add_argument('--rsi-scales',   type=float, nargs='+', default=None)
    parser.add_argument('--force-rerun',  action='store_true')
    parser.add_argument('--fair-fit',     action='store_true',
                        help="Use observation-only fit target for model-based baselines instead of the original default.")
    parser.add_argument('--prefix-calibration', action='store_true',
                        help="Use si_oracle prefix calibration for MP and Hammerstein v2 baselines.")
    parser.add_argument('--hv2-prefix-only', action='store_true',
                        help="Use prefix calibration only for Hammerstein v2, while keeping MP on its legacy definition.")
    parser.add_argument('--mp-prefix-calibration', action='store_true',
                        help="Use si_oracle prefix calibration for MP while keeping other baseline behavior unchanged.")
    parser.add_argument('--clean-split', action='store_true',
                        help="Convenience mode: physical no-normalize + digital internal normalize, with HV2 on prefix calibration only.")
    parser.add_argument('--prefix-samples', type=int, default=2048,
                        help="Calibration prefix length used with --prefix-calibration.")
    parser.add_argument('--digital-internal-normalize', action='store_true',
                        help="Forward --digital-internal-normalize to supported model-based digital SIC backends.")
    parser.add_argument('--diffusion-t-start', type=int, default=None,
                        help="Override T_START for diffusion runs only.")
    parser.add_argument('--trackb-variant', type=str, default='q2',
                        choices=['q2', 'q3_prefix', 'q3_midmid', 'q3_midmid_crop'],
                        help="Track B variant used when --mode fig4_sweep.")
    parser.add_argument('--align-realistic-to-trackb', action='store_true',
                        help="In fig4_sweep, run realistic lines with the HR/q3 protocol used by HR Track B variants.")
    parser.add_argument('--include-mp', action='store_true',
                        help="In fig4_sweep, include the Aux-TX + MP line.")
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument('--no-normalize', action='store_true',
                        help="Forward --no-normalize to run_sdd_final.py.")
    norm_group.add_argument('--normalize', action='store_true',
                        help="Explicitly keep semantic TX normalization enabled.")
    parser.add_argument('--out-suffix',   type=str,   default=None)

    # 僅重繪圖表，不重跑實驗（給有現成 CSV 的情況用）
    parser.add_argument('--plot-only',    type=str, default=None,
                        help="直接從指定 CSV 檔重繪圖表，不跑任何實驗。"
                             "例如：--plot-only results_sweep_ai_comp_generalization/sweep_results_ai_comp.csv")
    args = parser.parse_args()

    # ── plot-only 模式 ──
    if args.plot_only:
        _plot_from_csv(args)
        return

    if args.fair_fit and (args.prefix_calibration or args.hv2_prefix_only or args.mp_prefix_calibration):
        print("❌ --fair-fit cannot be used together with prefix-calibration modes.")
        return

    if args.prefix_calibration and args.hv2_prefix_only:
        print("❌ --prefix-calibration and --hv2-prefix-only cannot be used together.")
        return

    if args.prefix_calibration and args.mp_prefix_calibration:
        print("❌ --prefix-calibration already includes MP; do not combine it with --mp-prefix-calibration.")
        return

    if args.clean_split:
        if args.fair_fit or args.prefix_calibration:
            print("❌ --clean-split cannot be combined with --fair-fit or --prefix-calibration.")
            return
        if args.normalize:
            print("❌ --clean-split cannot be combined with --normalize.")
            return
        args.hv2_prefix_only = True
        args.no_normalize = True
        args.digital_internal_normalize = True

    if args.align_realistic_to_trackb and args.mode != 'fig4_sweep':
        print("--align-realistic-to-trackb is only supported with --mode fig4_sweep.")
        return

    if args.align_realistic_to_trackb and args.trackb_variant in {'q2', 'q3_midmid_crop'}:
        print("--align-realistic-to-trackb is intended for HR Track B variants, not q2 or q3_midmid_crop.")
        return

    if args.mode in {'ai_comp', 'ai_comp_v3c'}:
        tags = ['precomp_analog', 'precomp_digital', 'precomp_hammer_v2', 'precomp_diffusion']

        labels = {
            'precomp_analog':    'Aux-TX + Analog Only',
            'precomp_digital':   'Aux-TX + MP',
            'precomp_hammer_v2': 'Aux-TX + HW-Aware Hammerstein',
            'precomp_diffusion': 'Aux-TX + Diffusion (Ours)',
        }
        colors = {
            'precomp_analog':    'gray',
            'precomp_digital':   'orange',
            'precomp_hammer_v2': 'teal',
            'precomp_diffusion': 'blue',
        }
        markers = {
            'precomp_analog':    'v',
            'precomp_digital':   's',
            'precomp_hammer_v2': '^',
            'precomp_diffusion': '*',
        }
        linestyles = {
            'precomp_analog':    '-',
            'precomp_digital':   '-',
            'precomp_hammer_v2': '-',
            'precomp_diffusion': '-',
        }
        if args.mode == 'ai_comp_v3c':
            title_suffix = "V3c AI COMP (resize128/q3 fair protocol)"
        else:
            title_suffix = "Zero-Shot Generalization Test (Unseen Data)"
    elif args.mode == 'fig4_ref':
        tags = ['precomp_analog', 'precomp_digital', 'precomp_diffusion']

        labels = {
            'precomp_analog':    'Aux-TX + Analog Only',
            'precomp_digital':   'Aux-TX + MP',
            'precomp_diffusion': 'Aux-TX + Diffusion (Ours)',
        }
        colors = {
            'precomp_analog':    'gray',
            'precomp_digital':   'orange',
            'precomp_diffusion': 'blue',
        }
        markers = {
            'precomp_analog':    'v',
            'precomp_digital':   's',
            'precomp_diffusion': '*',
        }
        linestyles = {
            'precomp_analog':    '-',
            'precomp_digital':   '-',
            'precomp_diffusion': '-',
        }
        title_suffix = "SDD Fig.4-like Reference Sweep"
    elif args.mode == 'fig4_sweep':
        if args.trackb_variant == 'q2':
            trackb_tag = 'trackb_ref_q2'
        elif args.trackb_variant == 'q3_prefix':
            trackb_tag = 'trackb_ref_q3_prefix'
        elif args.trackb_variant == 'q3_midmid':
            trackb_tag = 'trackb_ref_q3_midmid'
        else:
            trackb_tag = 'trackb_ref_q3_midmid_crop'
        tags = ['precomp_analog', 'precomp_diffusion', trackb_tag]
        if args.include_mp:
            tags.insert(1, 'precomp_digital')

        labels = {
            'precomp_analog':    'Aux-TX Only',
            'precomp_digital':   'Aux-TX + MP',
            'precomp_diffusion': 'Aux-TX + Diffusion',
            'trackb_ref_q2':     'Paper-Align Ref (toy-analog, MP, NTSCC q2)',
            'trackb_ref_q3_prefix': 'Paper-Align Variant (toy-analog, MP prefix-fit, NTSCC q3 HR)',
            'trackb_ref_q3_midmid': 'Paper-Align V3 Candidate (toy-analog, MP mid-mid q3 HR)',
            'trackb_ref_q3_midmid_crop': 'Paper-Align V3c Candidate (toy-analog, MP mid-mid q3 resize128)',
        }
        colors = {
            'precomp_analog':    'gray',
            'precomp_digital':   'orange',
            'precomp_diffusion': 'blue',
            'trackb_ref_q2':     'black',
            'trackb_ref_q3_prefix': 'forestgreen',
            'trackb_ref_q3_midmid': 'darkorange',
            'trackb_ref_q3_midmid_crop': 'crimson',
        }
        markers = {
            'precomp_analog':    'v',
            'precomp_digital':   's',
            'precomp_diffusion': '*',
            'trackb_ref_q2':     'o',
            'trackb_ref_q3_prefix': '^',
            'trackb_ref_q3_midmid': 'D',
            'trackb_ref_q3_midmid_crop': 'P',
        }
        linestyles = {
            'precomp_analog':    '-',
            'precomp_digital':   '-',
            'precomp_diffusion': '-',
            'trackb_ref_q2':     '--',
            'trackb_ref_q3_prefix': '--',
            'trackb_ref_q3_midmid': '--',
            'trackb_ref_q3_midmid_crop': '--',
        }
        if args.trackb_variant == 'q2':
            title_suffix = "Final Fig.4-style SSIM Sweep (realistic lines + paper-alignment reference)"
        elif args.trackb_variant == 'q3_prefix':
            title_suffix = "Final Fig.4-style SSIM Sweep V2 (realistic lines + shape-oriented Track B variant)"
        elif args.trackb_variant == 'q3_midmid':
            title_suffix = "Final Fig.4-style SSIM Sweep V3 (realistic lines + morphology-balanced Track B candidate)"
        else:
            title_suffix = "Final Fig.4-style SSIM Sweep V3c (resize128/q3 fair three-line candidate)"
    else:
        print("目前僅支援 ai_comp 模式。")
        return

    if args.rsi_scales:
        rsi_scales = args.rsi_scales
    elif args.mode == 'fig4_ref':
        rsi_scales = FIG4_RSI_SCALE_LIST
    elif args.mode == 'ai_comp_v3c':
        rsi_scales = V3C_ANCHOR_RSI_SCALE_LIST
    elif args.mode == 'fig4_sweep':
        rsi_scales = FIG4_SWEEP_RSI_SCALE_LIST
    else:
        rsi_scales = RSI_SCALE_LIST

    if args.local_dir and args.remote_dir:
        all_pairs = collect_image_pairs(args.local_dir, args.remote_dir, args.max_pairs)
        print(f"目錄模式：{len(all_pairs)} 個圖片對")
        title_suffix = (f"Multi-Pair Test ({Path(args.local_dir).parent.name})"
                        f"  n={len(all_pairs)}")
    else:
        local_img  = args.local  if args.local  else DEFAULT_LOCAL
        remote_img = args.remote if args.remote else DEFAULT_REMOTE
        all_pairs  = [(local_img, remote_img)]
        print(f"單對模式：{Path(local_img).name}  vs  {Path(remote_img).name}")

    suffix   = f"_{args.out_suffix}" if args.out_suffix else ""
    out_root = Path(f"results_sweep_{args.mode}_generalization{suffix}")
    out_root.mkdir(exist_ok=True)

    print("=" * 80)
    print(f"🚀 SDD SWEEP  mode={args.mode.upper()}  "
          f"scales={[int(s) for s in rsi_scales]}")
    print(f"   tags={tags}")
    print(f"   fair_fit={args.fair_fit}")
    print(f"   prefix_calibration={args.prefix_calibration}")
    print(f"   hv2_prefix_only={args.hv2_prefix_only}")
    print(f"   mp_prefix_calibration={args.mp_prefix_calibration}")
    print(f"   clean_split={args.clean_split}")
    print(f"   no_normalize={args.no_normalize}")
    print(f"   digital_internal_normalize={args.digital_internal_normalize}")
    print(f"   diffusion_t_start={args.diffusion_t_start}")
    print(f"   align_realistic_to_trackb={args.align_realistic_to_trackb}")
    realistic_q3_resize128_align = (
        args.mode == 'ai_comp_v3c'
        or (args.mode == 'fig4_sweep' and args.trackb_variant == 'q3_midmid_crop')
    )
    print(f"   realistic_q3_resize128_align={realistic_q3_resize128_align}")
    if args.prefix_calibration or args.hv2_prefix_only or args.mp_prefix_calibration:
        print(f"   prefix_samples={args.prefix_samples}")
    if args.mode == 'fig4_sweep':
        print(f"   trackb_variant={args.trackb_variant}")
    print("=" * 80)

    rows = []
    for s in rsi_scales:
        for t in tags:
            print(f"\n{'─'*60}")
            print(f"  RSI={s:g}  tag={t}  ({len(all_pairs)} pair(s))")
            print(f"{'─'*60}")
            row = run_pairs_for_scale(
                s, t, all_pairs, out_root, args.force_rerun,
                fair_fit=args.fair_fit,
                prefix_calibration=args.prefix_calibration,
                hv2_prefix_only=args.hv2_prefix_only,
                mp_prefix_calibration=args.mp_prefix_calibration,
                prefix_samples=args.prefix_samples,
                use_no_normalize=args.no_normalize,
                digital_internal_normalize=args.digital_internal_normalize,
                realistic_protocol_align=args.align_realistic_to_trackb,
                realistic_q3_resize128_align=realistic_q3_resize128_align,
                diffusion_t_start=args.diffusion_t_start,
            )
            rows.append(row)

    # CSV
    csv_path = out_root / f"sweep_results_{args.mode}{suffix}.csv"
    with open(csv_path, "w") as f:
        f.write("rsi_scale,run_tag,n_pairs,time_sec,"
                "sinr_pre,sinr_after_analog,sinr_after_digital,psnr,ms_ssim\n")
        for r in rows:
            f.write(
                f"{r['rsi_scale']},{r['run_tag']},{r.get('n_pairs',1)},"
                f"{safe_float(r.get('time_sec',0)):.2f},"
                f"{safe_float(r.get('sinr_pre')):.4f},"
                f"{safe_float(r.get('sinr_after_analog')):.4f},"
                f"{safe_float(r.get('sinr_after_digital')):.4f},"
                f"{safe_float(r.get('psnr')):.4f},"
                f"{safe_float(r.get('ms_ssim')):.6f}\n"
            )
    print(f"\n✅ CSV：{csv_path}")

    make_plots(rows, tags, labels, colors, markers, linestyles,
               title_suffix, out_root, args.mode + suffix)

    # ── visual grid ───────────────────────────────────────────────────────
    try:
        from make_visual_grid import make_grid
        grid_path = out_root / f"visual_grid_{args.mode}{suffix}.png"
        make_grid(
            results_dir=str(out_root),
            tags=tags,
            rsi_scales=rsi_scales,
            output_path=str(grid_path),
            dpi=200,
        )
    except Exception as e:
        print(f"  [WARN] visual grid skipped: {e}")

    print(f"\n[OK] 結果目錄：{out_root}")
    print("SWEEP DONE")


def _plot_from_csv(args):
    """
    從現有 CSV 重繪圖表（不重跑實驗）。
    用法：python sweep.py --mode ai_comp --plot-only path/to/csv
    """
    import csv as csv_mod

    csv_path = Path(args.plot_only)
    if not csv_path.exists():
        print(f"❌ 找不到 CSV：{csv_path}")
        return

    rows = []
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for r in reader:
            rows.append({
                "rsi_scale":          safe_float(r.get("rsi_scale")),
                "run_tag":            r.get("run_tag", ""),
                "n_pairs":            int(r.get("n_pairs", 1)),
                "time_sec":           safe_float(r.get("time_sec", 0)),
                "sinr_pre":           safe_float(r.get("sinr_pre")),
                "sinr_after_analog":  safe_float(r.get("sinr_after_analog")),
                "sinr_after_digital": safe_float(r.get("sinr_after_digital")),
                "psnr":               safe_float(r.get("psnr")),
                "ms_ssim":            safe_float(r.get("ms_ssim")),
            })

    present_tags = list(dict.fromkeys(r["run_tag"] for r in rows))

    labels = {
        'precomp_analog':    'Aux-TX + Analog Only',
        'precomp_digital':   'Aux-TX + MP',
        'precomp_kong':      'Aux-TX + Kong-style NN SIC',
        'precomp_hammer_v2': 'Aux-TX + HW-Aware Hammerstein',
        'precomp_diffusion': 'Aux-TX + Diffusion (Ours)',
        'trackb_ref_q2':     'Paper-Align Ref (toy-analog, MP, NTSCC q2)',
        'trackb_ref_q3_prefix': 'Paper-Align Variant (toy-analog, MP prefix-fit, NTSCC q3 HR)',
        'trackb_ref_q3_midmid': 'Paper-Align V3 Candidate (toy-analog, MP mid-mid q3 HR)',
        'trackb_ref_q3_midmid_crop': 'Paper-Align V3c Candidate (toy-analog, MP mid-mid q3 resize128)',
    }
    colors = {
        'precomp_analog':    'gray',
        'precomp_digital':   'orange',
        'precomp_kong':      'darkred',
        'precomp_hammer_v2': 'teal',
        'precomp_diffusion': 'blue',
        'trackb_ref_q2':     'black',
        'trackb_ref_q3_prefix': 'forestgreen',
        'trackb_ref_q3_midmid': 'darkorange',
        'trackb_ref_q3_midmid_crop': 'crimson',
    }
    markers = {
        'precomp_analog':    'v',
        'precomp_digital':   's',
        'precomp_kong':      'P',
        'precomp_hammer_v2': '^',
        'precomp_diffusion': '*',
        'trackb_ref_q2':     'o',
        'trackb_ref_q3_prefix': '^',
        'trackb_ref_q3_midmid': 'D',
        'trackb_ref_q3_midmid_crop': 'P',
    }
    linestyles = {
        'precomp_analog':    '-',
        'precomp_digital':   '-',
        'precomp_kong':      '-',
        'precomp_hammer_v2': '-',
        'precomp_diffusion': '-',
        'trackb_ref_q2':     '--',
        'trackb_ref_q3_prefix': '--',
        'trackb_ref_q3_midmid': '--',
        'trackb_ref_q3_midmid_crop': '--',
    }

    out_dir    = csv_path.parent
    mode       = args.mode
    suffix     = f"_{args.out_suffix}" if args.out_suffix else ""
    title_suffix = "Zero-Shot Generalization Test (Unseen Data)"

    make_plots(rows, present_tags, labels, colors, markers, linestyles,
               title_suffix, out_dir, mode + suffix + "_replot")

    print(f"✅ 重繪完成：{out_dir}")


if __name__ == "__main__":
    main()
