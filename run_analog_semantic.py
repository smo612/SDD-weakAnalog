#!/usr/bin/env python3
"""
run_analog_semantic.py - Analog Physics Simulation (Fixed)
修正: 將 P_main 加入 meta.json，解決 Digital SIC 無法計算 SINR 的問題
"""
import os
import warnings
# 強制忽略警告
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
    with open(path / 'meta_tx.json', 'r') as f: meta = json.load(f)
    return x_tx, meta

def main():
    # 讀取配置
    SNR_DB = C.SNR_DB
    SIC_DB = C.SIC_DB_FIXED
    RSI_SCALE = C.RSI_SCALE
    
    # 載入信號
    x_self, _ = load_tx_signal('bridge_tx')
    x_remote, meta_remote = load_tx_signal('bridge_tx_remote')
    
    # --- 關鍵修正：長度對齊 (Padding) ---
    # 防止因直接截斷導致 NTSCC 封包破損而解碼失敗
    max_len = max(len(x_self), len(x_remote))
    if len(x_self) < max_len:
        x_self = np.pad(x_self, (0, max_len - len(x_self)), 'constant')
    if len(x_remote) < max_len:
        x_remote = np.pad(x_remote, (0, max_len - len(x_remote)), 'constant')
        
    # 截取有效部分
    n_sim = max_len 
    
    # 執行物理模擬
    rx_out = simulate_full_receive_signal(
        x_remote=x_remote,
        x_self=x_self,
        snr_db=SNR_DB,
        rsi_scale=RSI_SCALE,
        sic_db=SIC_DB,
        use_realistic_analog_sic=True,
        enable_pa_nonlinearity=True
    )
    
    y_main = rx_out['y_main']
    y_si_after = rx_out['y_rsi_after_analog']
    noise_var = rx_out['noise_var']
    info = rx_out['analog_sic_info']
    
    # 合成 ADC 輸入
    y_adc = (y_main + y_si_after).astype(np.complex64)
    
    # --- 物理診斷報告 ---
    P_main = np.mean(np.abs(y_main)**2)
    P_si_before = np.mean(np.abs(rx_out['y_rsi_before_analog'])**2)
    P_si_after = np.mean(np.abs(y_si_after)**2)
    Pn = noise_var
    
    SINR_pre = 10 * np.log10(P_main / (P_si_before + Pn + 1e-18))
    SINR_post = 10 * np.log10(P_main / (P_si_after + Pn + 1e-18))
    measured_gain = SINR_post - SINR_pre
    
    # 保存數據
    meta = {
        'snr_db': float(SNR_DB),
        'rsi_scale': float(RSI_SCALE),
        'SINR_pre': float(SINR_pre),
        'SINR_analog': float(SINR_post),
        'Supp_analog': float(measured_gain),
        'noise_var': float(noise_var),
        'P_main': float(P_main),   # <--- [關鍵修正] 加入這一行！
        'amp_scale': 1.0,
        'analog_sic_info': info
    }
    
    Path('bridge').mkdir(exist_ok=True)
    np.save('bridge/y_adc.npy', y_adc)
    np.save('bridge/y_si_after_analog.npy', y_si_after)
    with open('bridge/meta.json', 'w') as f: json.dump(meta, f, indent=2)
    with open('bridge/meta_tx_remote.json', 'w') as f: json.dump(meta_remote, f, indent=2)

if __name__ == "__main__":
    main()