#!/usr/bin/env python3
"""
run_sdd_final.py

Unified end-to-end SDD experiment runner:
1. Local / remote semantic TX
2. Analog SIC stage
3. Digital SIC backend
4. Semantic RX
5. Final report
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from utils.sic_eval import build_unified_digital_metrics

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.append(os.getcwd())
PYTHON_EXE = f'"{sys.executable}"'


try:
    import config

    DEFAULT_AUX_DISABLE = getattr(config, "AUX_DISABLE_IQPA", "Unknown")
except ImportError:
    DEFAULT_AUX_DISABLE = "Unknown"


def run_command(cmd, desc):
    print(f"[RUN] {desc}...", end=" ", flush=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print("FAILED")
        print("=" * 40 + " ERROR LOG " + "=" * 40)
        print(res.stderr)
        print(res.stdout)
        print("=" * 90)
        raise SystemExit(1)
    print("OK")


def force_cleanup(path_str):
    path = Path(path_str)
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def build_parser():
    parser = argparse.ArgumentParser(description="Run the full SDD pipeline.")
    parser.add_argument("--local", type=str, required=True)
    parser.add_argument("--remote", type=str, required=True)
    parser.add_argument(
        "--kodak-protocol",
        type=str,
        default="resize_128",
        choices=["resize_128", "witt_hr_center_crop"],
        help="Kodak preprocessing protocol used before semantic TX/RX.",
    )
    parser.add_argument(
        "--ntscc-ckpt",
        type=str,
        default="checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth",
        help="Override the pretrained NTSCC checkpoint used by both semantic TX and RX.",
    )
    parser.add_argument("--backend", type=str, default="mp",
                        choices=["wlls", "mp", "hammerstein", "hammerstein_v2", "hammerstein_v2_est"])
    parser.add_argument("--L", type=int, default=5, help="Memory length / tap count for digital SIC backends.")
    parser.add_argument("--lambda-reg", type=float, default=0.01, help="Ridge regularization for digital SIC backends.")
    parser.add_argument("--poly-orders", type=str, default="1,3,5", help="Comma-separated odd orders for MP-style backends.")
    parser.add_argument("--fit-target", type=str, default="si_oracle", choices=["si_oracle", "observation"],
                        help="Fit target for model-based digital SIC backends.")
    parser.add_argument("--fit-window", type=str, default="legacy", choices=["legacy", "full", "prefix"],
                        help="Coefficient fitting window for model-based digital SIC backends.")
    parser.add_argument("--fit-prefix-samples", type=int, default=1024,
                        help="Prefix/calibration sample count when --fit-window prefix.")
    parser.add_argument("--digital-internal-normalize", action="store_true",
                        help="Enable digital-backend-only internal normalization for supported model-based SIC backends.")
    parser.add_argument("--est-iq-grid", type=str, default="0.00,0.01,0.03,0.05",
                        help="Coarse IQ-imbalance candidate grid for hammerstein_v2_est.")
    parser.add_argument("--est-phase-grid-deg", type=str, default="0.0,1.0,3.0,5.0",
                        help="Coarse phase-error candidate grid (degrees) for hammerstein_v2_est.")
    parser.add_argument("--est-asat-grid", type=str, default="2.0,2.5,3.5,4.0",
                        help="Coarse PA saturation candidate grid for hammerstein_v2_est.")
    parser.add_argument("--est-val-ratio", type=float, default=0.25,
                        help="Validation ratio inside the calibration prefix for hammerstein_v2_est.")
    parser.add_argument("--rsi-scale", type=float, default=20.0)
    parser.add_argument("--no-digital-sic", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--use-diffusion", action="store_true", help="Use diffusion AI-SIC.")
    parser.add_argument("--use-kong", action="store_true", help="Use Kong v5 NN-aided SIC.")
    parser.add_argument("--use-cnn", action="store_true", help="Use v6a direct CNN SIC.")
    parser.add_argument("--use-cnn-mp", action="store_true", help="Use v6b MP + CNN post-filter SIC.")
    parser.add_argument("--use-cnn-mp-next", action="store_true", help="Use v6b-next MP + CNN correction SIC.")
    parser.add_argument("--use-cnn-mp-additive", action="store_true", help="Use v6b-additive MP + additive correction SIC.")

    analog_group = parser.add_mutually_exclusive_group()
    analog_group.add_argument("--toy-analog", action="store_true", help="Use oracle 50 dB toy analog SIC.")
    analog_group.add_argument("--old-sdd", action="store_true", help="Use the older realistic analog SIC baseline.")

    parser.add_argument(
        "--aux-disable-iqpa",
        type=str,
        choices=["True", "False"],
        default=None,
        help="Override AUX_DISABLE_IQPA in config.py.",
    )
    return parser


def build_ntscc_rx_tmp_script(args) -> str:
    return f"""
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from src.semantic.kodak_protocol_utils import load_kodak_image
from src.semantic.ntscc_rx_wrapper import NTSCCRXWrapper

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
sys.path.append(os.getcwd())

y = np.load('bridge_digital/y_clean.npy')
with open('bridge/meta_tx_remote.json') as f:
    tx = json.load(f)

img_gt, gt_path, protocol_meta = load_kodak_image(
    r"{args.remote}",
    protocol=r"{args.kodak_protocol}",
)
img_size = (img_gt.shape[0], img_gt.shape[1])

Path('bridge_rx').mkdir(exist_ok=True)
Image.fromarray((img_gt * 255).astype(np.uint8)).save('bridge_rx/img_original_remote.png')

dec = NTSCCRXWrapper(
    ckpt_path=r"{args.ntscc_ckpt}",
    device='cuda',
    img_size=img_size,
)
img, metrics = dec.decode(y, original_img=img_gt, img_size=img_size, cbr=1/16, meta_tx=tx)
metrics['source_image'] = gt_path
metrics['kodak_protocol'] = r"{args.kodak_protocol}"
metrics['protocol_meta'] = protocol_meta

Image.fromarray((img * 255).astype(np.uint8)).save('bridge_rx/img_recon_remote.png')
with open('bridge_rx/metrics_remote.json', 'w') as f:
    json.dump(metrics, f)
"""


def select_analog_mode(args):
    if args.toy_analog:
        return (
            "TOY 50dB oracle (IBFD-SC 2025 baseline)",
            "run_analog_semantic_toy.py",
            "run toy analog SIC",
        )
    if args.old_sdd:
        return (
            "Old SDD realistic (no PA, saturation baseline)",
            "run_analog_semantic_old_sdd.py",
            "run old SDD analog SIC",
        )
    return (
        "Aux-TX realistic (WL+NL FIR)",
        "run_analog_semantic.py",
        "run Aux-TX realistic analog SIC",
    )


def summarize_results(args, analog_mode_str, run_aux_status):
    print("\n" + "=" * 60)
    print("Final Report")
    print("=" * 60)
    try:
        with open("bridge/meta.json") as f:
            analog_meta = json.load(f)
        with open("bridge_digital/metrics.json") as f:
            digital_meta = json.load(f)
        with open("bridge_rx/metrics_remote.json") as f:
            rx_meta = json.load(f)
    except Exception as exc:
        print(f"Final report generation failed: {exc}")
        return

    sinr_pre = analog_meta.get("SINR_pre", 0.0)
    sinr_analog = analog_meta.get("SINR_analog", 0.0)
    gain_analog = sinr_analog - sinr_pre
    backend_name = digital_meta.get("backend", "None")

    physics_digital = digital_meta.get("SINR_after_digital_physics")
    symbol_noisy = digital_meta.get("symbol_sinr_noisy_db")
    symbol_clean = digital_meta.get("symbol_sinr_clean_db")
    mse_gain = digital_meta.get("mse_gain_db")
    corr_clean = digital_meta.get("corr_clean_abs")
    psnr_val = rx_meta.get("psnr")

    print("Summary:")
    print(f"  RSI scale: {analog_meta.get('rsi_scale', 0)}x")
    print(f"  Analog mode: {analog_mode_str}")
    if not args.toy_analog and not args.old_sdd:
        print(f"  AUX disable IQ/PA: {run_aux_status}")
    print(f"  Digital backend: {backend_name}")

    print("\nPhysics-side SINR:")
    print(f"  1. Before analog SIC: {sinr_pre:6.2f} dB")
    print(f"  2. After analog SIC:  {sinr_analog:6.2f} dB  (Gain: {gain_analog:+.2f} dB)")
    if physics_digital is not None:
        total_physics_gain = physics_digital - sinr_pre
        print(f"  3. After digital SIC: {physics_digital:6.2f} dB  [{backend_name}]")
        print(f"  Total physics gain:   {total_physics_gain:.2f} dB")
    else:
        print(f"  3. After digital SIC: N/A  [{backend_name}]")
        print("  Total physics gain:   N/A")

    print("\nSymbol fidelity:")
    if symbol_noisy is not None:
        print(f"  Noisy symbol SINR:    {symbol_noisy:6.2f} dB")
    else:
        print("  Noisy symbol SINR:    N/A")
    if symbol_clean is not None:
        print(f"  Clean symbol SINR:    {symbol_clean:6.2f} dB")
    else:
        print("  Clean symbol SINR:    N/A")
    if mse_gain is not None:
        print(f"  MSE gain:             {mse_gain:6.2f} dB")
    else:
        print("  MSE gain:             N/A")
    if corr_clean is not None:
        print(f"  |corr(clean,target)|: {corr_clean:.4f}")
    else:
        print("  |corr(clean,target)|: N/A")

    if psnr_val is not None and psnr_val >= 0:
        print(f"\nRX image PSNR: {psnr_val:.2f} dB")
        if psnr_val > 30.0:
            print("High-quality reconstruction achieved (PSNR > 30 dB).")
    else:
        print("\nRX image PSNR: N/A")


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Start each run from a clean artifact state so failed runs do not leak
    # stale bridge outputs into the next sweep row.
    for path_str in ["bridge", "bridge_digital", "bridge_rx", "_rx_tmp.py"]:
        force_cleanup(path_str)

    analog_mode_str, analog_script, analog_desc = select_analog_mode(args)

    if args.aux_disable_iqpa is not None:
        run_aux_status = args.aux_disable_iqpa == "True"
        status_source = "arg"
    else:
        run_aux_status = DEFAULT_AUX_DISABLE
        status_source = "config"

    print("\n" + "=" * 60)
    print(f"SDD Final Physics Simulation (RSI={args.rsi_scale}x)")
    print(f"  Analog mode: {analog_mode_str}")
    if not args.toy_analog and not args.old_sdd:
        print(f"  AUX_DISABLE_IQPA = {run_aux_status} ({status_source})")
    print(f"  Use diffusion: {args.use_diffusion}")
    print(f"  Use Kong:      {args.use_kong}")
    print(f"  Use CNN v6a:   {args.use_cnn}")
    print(f"  Use CNN v6b:   {args.use_cnn_mp}")
    print(f"  Use CNN next:  {args.use_cnn_mp_next}")
    print(f"  Use CNN add:   {args.use_cnn_mp_additive}")
    print(f"  Digital internal normalize: {args.digital_internal_normalize}")
    print(f"  Kodak protocol: {args.kodak_protocol}")
    print("=" * 60 + "\n")

    norm_flag = "--no-normalize" if args.no_normalize else ""

    run_command(
        f'{PYTHON_EXE} scripts/run_tx_kodak_batch.py --img "{args.local}" --ckpt "{args.ntscc_ckpt}" --kodak-protocol {args.kodak_protocol} --output bridge_tx {norm_flag}',
        "semantic TX for local image",
    )

    force_cleanup("bridge_tx_remote")
    run_command(
        f'{PYTHON_EXE} scripts/run_tx_kodak_batch.py --img "{args.remote}" --ckpt "{args.ntscc_ckpt}" --kodak-protocol {args.kodak_protocol} --output bridge_tx_remote {norm_flag}',
        "semantic TX for remote image",
    )

    cfg_path = Path("config.py")
    cfg_orig = cfg_path.read_text(encoding="utf-8")
    cfg_new = re.sub(r"(?m)^RSI_SCALE\s*=\s*[\d.]+", f"RSI_SCALE = {args.rsi_scale}", cfg_orig)
    if args.aux_disable_iqpa is not None and not args.toy_analog and not args.old_sdd:
        cfg_new = re.sub(
            r"(?m)^AUX_DISABLE_IQPA\s*=\s*(True|False)",
            f"AUX_DISABLE_IQPA = {args.aux_disable_iqpa}",
            cfg_new,
        )
    cfg_path.write_text(cfg_new, encoding="utf-8")
    try:
        run_command(f"{PYTHON_EXE} {analog_script}", analog_desc)
    finally:
        cfg_path.write_text(cfg_orig, encoding="utf-8")

    force_cleanup("bridge_digital")
    Path("bridge_digital").mkdir(exist_ok=True)

    if args.use_diffusion:
        run_command(f"{PYTHON_EXE} run_diffusion.py", "diffusion AI-SIC")
        y_clean = np.load("bridge_digital/y_clean.npy").flatten().astype(np.complex64)
        diffusion_metrics = build_unified_digital_metrics(
            backend_name="Diffusion",
            y_clean=y_clean,
            extra_metrics={
                "SINR_after_digital_physics": None,
            },
        )
        with open("bridge_digital/metrics.json", "w") as f:
            json.dump(diffusion_metrics, f, indent=2)
    elif args.use_kong:
        kong_env = f'KONG_LOCAL_IMG="{args.local}" KONG_RSI_SCALE="{args.rsi_scale}"'
        run_command(f'{kong_env} {PYTHON_EXE} run_kong_sic_v5.py', "Kong v5 NN-aided SIC")
    elif args.use_cnn:
        run_command(f"{PYTHON_EXE} run_cnn_sic.py", "CNN v6a direct residual SIC")
    elif args.use_cnn_mp:
        run_command(f"{PYTHON_EXE} run_cnn_sic_v6b.py", "CNN v6b MP + post-filter SIC")
    elif args.use_cnn_mp_next:
        run_command(f"{PYTHON_EXE} run_cnn_sic_v6b_next.py", "CNN v6b-next MP + additive correction SIC")
    elif args.use_cnn_mp_additive:
        run_command(f"{PYTHON_EXE} run_cnn_sic_v6b_additive.py", "CNN v6b-additive MP + additive correction SIC")
    elif args.no_digital_sic:
        print("[INFO] Digital SIC skipped (analog-only mode).")
        shutil.copy("bridge/y_adc.npy", "bridge_digital/y_clean.npy")
        with open("bridge/meta.json") as f:
            analog_meta = json.load(f)
        y_clean = np.load("bridge_digital/y_clean.npy").flatten().astype(np.complex64)
        analog_only_metrics = build_unified_digital_metrics(
            backend_name="None",
            y_clean=y_clean,
            extra_metrics={
                "SINR_after_digital_physics": analog_meta.get("SINR_analog"),
                "SINR_analog": analog_meta.get("SINR_analog"),
            },
        )
        with open("bridge_digital/metrics.json", "w") as f:
            json.dump(analog_only_metrics, f, indent=2)
    else:
        if Path("scripts/run_digital_sic.py").exists():
            internal_norm_flag = "--digital-internal-normalize" if args.digital_internal_normalize else ""
            run_command(
                f"{PYTHON_EXE} scripts/run_digital_sic.py --backend {args.backend} --L {args.L} "
                f"--lambda-reg {args.lambda_reg} --poly-orders {args.poly_orders} "
                f"--fit-target {args.fit_target} --fit-window {args.fit_window} "
                f"--fit-prefix-samples {args.fit_prefix_samples} "
                f"--est-iq-grid {args.est_iq_grid} "
                f"--est-phase-grid-deg {args.est_phase_grid_deg} "
                f"--est-asat-grid {args.est_asat_grid} "
                f"--est-val-ratio {args.est_val_ratio} "
                f"{internal_norm_flag}",
                f"digital SIC backend ({args.backend.upper()})",
            )
        else:
            print("[WARN] scripts/run_digital_sic.py not found, skip digital SIC.")
            rx_script = build_ntscc_rx_tmp_script(args)
            Path("_rx_tmp.py").write_text(rx_script)
            run_command(f"{PYTHON_EXE} _rx_tmp.py", "semantic RX")
            if Path("_rx_tmp.py").exists():
                Path("_rx_tmp.py").unlink()

            summarize_results(args, analog_mode_str, run_aux_status)
            return
    rx_script = build_ntscc_rx_tmp_script(args)

    Path("_rx_tmp.py").write_text(rx_script)
    run_command(f"{PYTHON_EXE} _rx_tmp.py", "semantic RX")
    if Path("_rx_tmp.py").exists():
        Path("_rx_tmp.py").unlink()

    summarize_results(args, analog_mode_str, run_aux_status)


if __name__ == "__main__":
    main()
