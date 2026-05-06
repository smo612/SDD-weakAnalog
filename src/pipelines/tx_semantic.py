"""
tx_semantic.py - Semantic TX 管線（方案一修復版）

✅ 核心修復：
1. 添加 normalize_power 參數控制是否歸一化
2. 當 normalize_power=False 時，保持信號原始功率
3. 這樣可以測試 RSI_SCALE 的真實影響
"""
import numpy as np
import os
import json
from pathlib import Path
from typing import Dict, Optional

from src.semantic.ntscc_wrapper import NTSCCWrapper


class SemanticTX:
    """
    語義 TX 管線
    
    流程：
    1. NTSCC encoder: 影像 → latent [1, 256, 8, 8]
    2. Latent to baseband: latent → 複數符號（可選功率歸一）
    3. 導頻插入
    """
    
    def __init__(self,
                 ntscc_ckpt: str,
                 use_pilot: bool = True,
                 pilot_period: int = 64,
                 pilot_val: complex = 1.0 + 0.0j,
                 normalize_power: bool = True):  # ✅ 新增參數
        """
        Args:
            ntscc_ckpt: NTSCC checkpoint 路徑
            use_pilot: 是否插入導頻
            pilot_period: 導頻週期（每 N 個符號插入一個）
            pilot_val: 導頻值
            normalize_power: 是否歸一化功率（預設 True 保持向後兼容）
        """
        self.ntscc_ckpt = ntscc_ckpt
        self.use_pilot = use_pilot
        self.pilot_period = pilot_period
        self.pilot_val = pilot_val
        self.normalize_power = normalize_power  # ✅ 儲存設定
        
        # 建立 NTSCC wrapper
        print("[SemanticTX] 初始化 NTSCC encoder...")
        self.ntscc = NTSCCWrapper(ckpt_path=ntscc_ckpt)
        print("[SemanticTX] ✓ 初始化完成")
        
        # ✅ 顯示歸一化設定
        if not self.normalize_power:
            print("[SemanticTX] ⚠️  功率歸一化已禁用！信號將保持原始功率")
    
    def transmit(self, 
                 img: np.ndarray,
                 cbr: float = 1/16,
                 sps: int = 1,
                 seed: Optional[int] = None) -> Dict:
        """
        執行完整 TX 管線
        
        Args:
            img: [H, W, 3] RGB 影像, 0~1 範圍
            cbr: Channel Bandwidth Ratio
            sps: Samples per symbol（預留，目前固定=1）
            seed: 隨機種子（預留）
        
        Returns:
            dict with 'x_tx' and 'meta'
        """
        # 1. 語義編碼
        latent, ctx = self.ntscc.encode(img, cbr=cbr)
        
        # 2. Analog 調變（latent → 連續複數基帶）
        # ✅ 根據 normalize_power 設定決定是否歸一化
        if self.normalize_power:
            symbols, tx_scale, actual_power = self._latent_to_baseband(latent, target_power=1.0)
        else:
            symbols, tx_scale, actual_power = self._latent_to_baseband(latent, target_power=None)
        
        # 3. 插入導頻
        if self.use_pilot:
            x_tx = self._insert_pilots(symbols, self.pilot_period, self.pilot_val)
            n_pilots = len(x_tx) - len(symbols)
        else:
            x_tx = symbols
            n_pilots = 0
        
        # 4. 驗證功率
        final_power = float(np.mean(np.abs(x_tx)**2))
        
        # ✅ 根據是否歸一化調整驗證邏輯
        if self.normalize_power:
            is_power_ok = abs(final_power - 1.0) < 0.05
        else:
            is_power_ok = True  # 不歸一化時不檢查
        
        # 5. 建立完整 meta
        meta = {
            'tx_info': {
                'mode': 'semantic',
                'ntscc_mode': 'real',
                'ntscc_ckpt': str(self.ntscc_ckpt),
                'B': 1,
                'H': int(img.shape[0]),
                'W': int(img.shape[1]),
                'cbr': float(cbr),
                'sps': int(sps),
                'pulse_shaping': 'none',
                'seed': int(seed) if seed is not None else None,
                'normalize_power': bool(self.normalize_power),  # ✅ 記錄設定
            },
            'pilot_info': {
                'pilot_enabled': bool(self.use_pilot),
                'pilot_period': int(self.pilot_period) if self.use_pilot else None,
                'pilot_val': {
                    'real': float(self.pilot_val.real),
                    'imag': float(self.pilot_val.imag),
                    'magnitude': float(np.abs(self.pilot_val)),
                },
                'n_pilots': int(n_pilots),
            },
            'signal_info': {
                'latent_shape': [int(x) for x in ctx['latent_shape']],
                'flatten_order': 'even->I,odd->Q',
                'n_data_symbols': int(len(symbols)),
                'n_total_symbols': int(len(x_tx)),
                'original_power': float(actual_power),  # ✅ 原始功率
                'mean_power': float(final_power),       # ✅ 最終功率
                'power_check_passed': bool(is_power_ok),
                'tx_scale': float(tx_scale),
                'dtype': 'complex64',
            },
            'source_info': {
                'source_image': None,
                'image_name': None,
                'dataset': None,
            }
        }
        
        return {
            'x_tx': x_tx,
            'meta': meta
        }
    
    def _latent_to_baseband(self, 
                           latent: np.ndarray, 
                           target_power: Optional[float] = 1.0) -> tuple:
        """
        將 latent 轉為連續複數基帶
        
        ✅ 修復：添加 target_power=None 選項來禁用歸一化
        
        策略：
        - Flatten latent [1, 256, 8, 8] → [16384]
        - 偶數索引 → I（實部）
        - 奇數索引 → Q（虛部）
        - 如果 target_power 不為 None，則歸一化到目標功率
        - 如果 target_power 為 None，則保持原始功率
        
        Args:
            latent: [1, C, H, W] tensor
            target_power: 目標功率（None = 不歸一化）
        
        Returns:
            symbols: [N] complex64 array
            tx_scale: 縮放係數（供 RX 端反向還原）
            actual_power: 實際功率（歸一化前）
        """
        # Tensor → numpy
        if hasattr(latent, 'cpu'):
            latent_np = latent.cpu().numpy()
        else:
            latent_np = latent
        
        # Flatten
        flat = latent_np.flatten().astype(np.float32)
        
        # 偶數→I, 奇數→Q
        I = flat[0::2]
        Q = flat[1::2]
        
        # 組合複數
        symbols = (I + 1j * Q).astype(np.complex64)
        
        # 記錄原始功率
        original_power = float(np.mean(np.abs(symbols)**2))
        
        # ✅ 條件歸一化
        if target_power is not None:
            # 歸一化模式
            if original_power > 1e-10:
                tx_scale = np.sqrt(target_power / original_power)
                symbols = symbols * tx_scale
                print(f"[TX] 功率歸一化: {original_power:.6f} → {target_power:.6f} (scale={tx_scale:.6f})")
            else:
                tx_scale = 1.0
                print(f"[TX] ⚠️  原始功率過低 ({original_power:.3e})，跳過歸一化")
        else:
            # 不歸一化模式
            tx_scale = 1.0
            print(f"[TX] 保持原始功率: {original_power:.6f} (未歸一化)")
        
        return symbols, float(tx_scale), float(original_power)
    
    def _insert_pilots(self, 
                      data_symbols: np.ndarray, 
                      period: int, 
                      pilot_val: complex) -> np.ndarray:
        """
        插入導頻符號
        
        策略：每 period 個數據符號後插入一個導頻
        
        Args:
            data_symbols: [N] 數據符號
            period: 導頻週期
            pilot_val: 導頻值
        
        Returns:
            symbols_with_pilots: [N + N//period] 含導頻的符號
        """
        N = len(data_symbols)
        n_pilots = N // period
        
        output = []
        for i in range(0, N, period):
            chunk = data_symbols[i:i+period]
            output.append(chunk)
            if i + period < N:
                output.append(np.array([pilot_val], dtype=np.complex64))
        
        return np.concatenate(output)
    
    def save_bridge(self, x_tx: np.ndarray, meta: dict, output_dir: str = 'bridge_tx'):
        """
        儲存到 bridge/ 資料夾
        
        Args:
            x_tx: TX 符號 [N]
            meta: 元數據字典
            output_dir: 輸出資料夾
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 儲存符號
        np.save(output_path / 'x_tx.npy', x_tx)
        
        # 儲存 meta
        with open(output_path / 'meta_tx.json', 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n[SemanticTX] ✓ 儲存到 {output_dir}/")
        print(f"  - x_tx.npy: {len(x_tx)} 符號")
        print(f"  - meta_tx.json")