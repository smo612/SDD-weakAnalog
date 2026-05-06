#!/usr/bin/env python3
"""
run_analog_semantic_toy.py - Toy Analog SIC (IBFD-SC baseline)

模擬 Shi et al., "In-Band Full-Duplex System for Semantic Communication"
IEEE IoT Journal 2025 (Table IV) 的類比消除設定：

  Analog domain SIC performance: 50 dB  (固定理想消除)

用途：作為 sweep 的 baseline 第四條線，
      代表「原始 SDD paper 的做法」(SIC_DB=50 dB oracle analog + MP digital)。

與 run_analog_semantic.py 的差異：
  use_realistic_analog_sic = False  →  toy k_amp 消除
  sic_db = SIC_DB_FIXED             →  從 config 讀取（預設 50 dB）
  enable_pa_nonlinearity = True     →  保持相同的 PA 失真（公平比較）
"""
import os
import warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings('ignore')

import numpy as np
import json
from pathlib import Path
import sys
import config as C
from sdd_channel_model_v5 import simulate_full_receive_signal


def load_tx_signal(bridge_tx_dir='bridge_tx'):
    path = Path(bridge_tx_dir)
    x_tx = np.load(path / 'x_tx.npy')
    with open(path / 'meta_tx.json', 'r') as f:
        meta = json.load(f)
    return x_tx, meta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--si-only', action='store_true',
                        help="SI-only 模式：x_remote 設為全零，用於 Kong v3 channel estimation")
    args, _ = parser.parse_known_args()

    SNR_DB    = C.SNR_DB
    SIC_DB    = C.SIC_DB_FIXED
    RSI_SCALE = C.RSI_SCALE
    TOY_SIC_DB = float(os.environ.get('TOY_SIC_DB', getattr(C, 'TOY_SIC_DB', 50.0)))

    x_self,   _           = load_tx_signal('bridge_tx')
    x_remote, meta_remote = load_tx_signal('bridge_tx_remote')

    # 長度對齊
    max_len = max(len(x_self), len(x_remote))
    if len(x_self) < max_len:
        x_self   = np.pad(x_self,   (0, max_len - len(x_self)),   'constant')
    if len(x_remote) < max_len:
        x_remote = np.pad(x_remote, (0, max_len - len(x_remote)), 'constant')

    # SI-only 模式：把 x_remote 設為全零，接收端只有 SI
    if args.si_only:
        x_remote = np.zeros_like(x_remote)
        print("[toy analog] SI-Only 模式：x_remote = 0（純 SI 接收，用於 Kong channel estimation）")

    # ── 執行物理模擬 (toy 類比消除) ──
    rx_out = simulate_full_receive_signal(
        x_remote=x_remote,
        x_self=x_self,
        snr_db=SNR_DB,
        rsi_scale=RSI_SCALE,
        sic_db=TOY_SIC_DB,             # ← 固定 50 dB 理想消除
        use_realistic_analog_sic=False, # ← toy 模式：不用 Aux-TX
        enable_pa_nonlinearity=True     # ← 保持 PA 非線性（公平比較）
    )

    y_main    = rx_out['y_main']
    y_si_after = rx_out['y_rsi_after_analog']
    noise_var  = rx_out['noise_var']
    info       = rx_out['analog_sic_info']

    y_adc = (y_main + y_si_after).astype(np.complex64)

    # 診斷
    P_main      = float(np.mean(np.abs(y_main)**2))
    P_si_before = float(np.mean(np.abs(rx_out['y_rsi_before_analog'])**2))
    P_si_after_ = float(np.mean(np.abs(y_si_after)**2))
    Pn          = float(noise_var)

    SINR_pre  = 10 * np.log10(P_main / (P_si_before + Pn + 1e-18))
    SINR_post = 10 * np.log10(P_main / (P_si_after_ + Pn + 1e-18))
    gain      = SINR_post - SINR_pre

    meta = {
        'snr_db':        float(SNR_DB),
        'rsi_scale':     float(RSI_SCALE),
        'SINR_pre':      float(SINR_pre),
        'SINR_analog':   float(SINR_post),
        'Supp_analog':   float(gain),
        'noise_var':     float(noise_var),
        'P_main':        float(P_main),
        'amp_scale':     1.0,
        'analog_mode':   'toy_50dB',   # 標記用途
        'toy_sic_db':    TOY_SIC_DB,
        'analog_sic_info': info,
    }

    # SI-only 模式：輸出到 bridge_si_only/，不要蓋掉主流程的 bridge/
    if args.si_only:
        out_dir = Path('bridge_si_only')
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / 'y_adc.npy',             y_adc)
        np.save(out_dir / 'y_si_after_analog.npy', y_si_after)
        with open(out_dir / 'meta.json', 'w') as f: json.dump(meta, f, indent=2)
        print(f"[toy analog SI-only] 輸出到 bridge_si_only/  "
              f"SINR_pre={SINR_pre:.2f} dB  SINR_analog={SINR_post:.2f} dB")
    else:
        Path('bridge').mkdir(exist_ok=True)
        np.save('bridge/y_adc.npy',              y_adc)
        np.save('bridge/y_si_after_analog.npy',  y_si_after)
        with open('bridge/meta.json',          'w') as f: json.dump(meta, f, indent=2)
        with open('bridge/meta_tx_remote.json','w') as f: json.dump(meta_remote, f, indent=2)
        print(f"[toy analog] SINR_pre={SINR_pre:.2f} dB  "
              f"SINR_analog={SINR_post:.2f} dB  "
              f"Supp={gain:.2f} dB  (toy {TOY_SIC_DB:.0f} dB)")


if __name__ == "__main__":
    main()
