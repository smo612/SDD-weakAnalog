#!/usr/bin/env python3
"""
run_tx_kodak_batch.py

Encode a Kodak image with the real NTSCC semantic TX path and export the
standard bridge artifacts used by the SDD shell.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.tx_semantic import SemanticTX
from src.semantic.kodak_protocol_utils import load_kodak_image


NTSCC_CKPT = "checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth"


def process_single_image(
    img_name: str,
    ckpt_path: str,
    output_dir: str = "bridge_tx",
    cbr: float = 1 / 16,
    normalize_power: bool = True,
    kodak_protocol: str = "resize_128",
):
    print("=" * 60)
    print(f"Encode image: {img_name}")
    print(f"Kodak protocol: {kodak_protocol}")
    if not normalize_power:
        print("Power normalization disabled")
    print("=" * 60)

    img, img_path, protocol_meta = load_kodak_image(img_name, protocol=kodak_protocol)
    print(f"[Load Image] {img_path}")
    print(f"  Shape: {img.shape}, Range: [{img.min():.3f}, {img.max():.3f}]")

    tx = SemanticTX(
        ntscc_ckpt=ckpt_path,
        use_pilot=True,
        pilot_period=64,
        normalize_power=normalize_power,
    )

    result = tx.transmit(img, cbr=cbr, sps=1)
    result["meta"]["source_info"] = {
        "source_image": img_path,
        "image_name": img_name,
        "dataset": "Custom/Kodak",
        "kodak_protocol": kodak_protocol,
        "protocol_meta": protocol_meta,
    }

    tx.save_bridge(x_tx=result["x_tx"], meta=result["meta"], output_dir=output_dir)

    print(f"\nFinished: {img_name}")
    print(f"  Output: {output_dir}/")
    print(f"  Symbols: {len(result['x_tx'])}")
    return result


def process_all_kodak(
    ckpt_path: str,
    output_base: str = "bridge_tx_kodak",
    cbr: float = 1 / 16,
    normalize_power: bool = True,
    kodak_protocol: str = "resize_128",
):
    results = {}
    for i in range(1, 25):
        img_name = f"kodim{i:02d}"
        output_dir = f"{output_base}/{img_name}"
        try:
            result = process_single_image(
                img_name,
                ckpt_path,
                output_dir=output_dir,
                cbr=cbr,
                normalize_power=normalize_power,
                kodak_protocol=kodak_protocol,
            )
            results[img_name] = {
                "status": "success",
                "n_symbols": len(result["x_tx"]),
                "power": float(np.mean(np.abs(result["x_tx"]) ** 2)),
                "normalized": normalize_power,
                "kodak_protocol": kodak_protocol,
            }
            print("")
        except Exception as exc:
            print(f"\nFAILED {img_name}: {exc}")
            results[img_name] = {"status": "failed", "error": str(exc)}

    summary_path = Path(output_base) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("Batch finished")
    success_count = sum(1 for row in results.values() if row["status"] == "success")
    print(f"  Success: {success_count} / {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Encode Kodak image(s) with the real NTSCC TX path.")
    parser.add_argument("--img", type=str, help="Image path or Kodak image stem.")
    parser.add_argument("--all", action="store_true", help="Process all 24 Kodak images.")
    parser.add_argument("--cbr", type=float, default=1 / 16)
    parser.add_argument("--output", type=str, default="bridge_tx")
    parser.add_argument("--ckpt", type=str, default=NTSCC_CKPT)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument(
        "--kodak-protocol",
        type=str,
        default="resize_128",
        choices=["resize_128", "witt_hr_center_crop"],
        help="Kodak loader protocol used before semantic TX.",
    )
    args = parser.parse_args()

    ckpt_path = args.ckpt
    normalize_power = not args.no_normalize

    if not Path(ckpt_path).exists():
        print(f"Missing checkpoint: {ckpt_path}")
        sys.exit(1)

    if args.all:
        process_all_kodak(
            ckpt_path,
            output_base="bridge_tx_kodak",
            cbr=args.cbr,
            normalize_power=normalize_power,
            kodak_protocol=args.kodak_protocol,
        )
    elif args.img:
        process_single_image(
            args.img,
            ckpt_path,
            output_dir=args.output,
            cbr=args.cbr,
            normalize_power=normalize_power,
            kodak_protocol=args.kodak_protocol,
        )
    else:
        process_single_image(
            "kodim01",
            ckpt_path,
            output_dir=args.output,
            cbr=args.cbr,
            normalize_power=normalize_power,
            kodak_protocol=args.kodak_protocol,
        )


if __name__ == "__main__":
    main()
