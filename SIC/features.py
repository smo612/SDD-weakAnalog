"""SIC/features.py - 特徵提取

提供：
- Memory Polynomial 特徵
- 短窗特徵（用於 NN）
- 禁止 per-sample 正規化
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def build_mp_features(x, poly_orders=[1, 3], memory_len=5, with_conj=True):
    """
    構建 Memory Polynomial 特徵矩陣
    
    特徵形式：x[n-m] * |x[n-m]|^(p-1)，包含共軛項
    
    Args:
        x: 發送信號 [N] complex
        poly_orders: 多項式階數列表，例如 [1, 3, 5]
        memory_len: 記憶長度 M
        with_conj: 是否包含共軛項
    
    Returns:
        Phi: 特徵矩陣 [N, K] complex
              K = len(poly_orders) * M * (2 if with_conj else 1)
    """
    N = len(x)
    M = memory_len
    P_list = poly_orders
    
    # 計算特徵維度
    n_poly = len(P_list)
    n_conj = 2 if with_conj else 1
    K = n_poly * M * n_conj
    
    Phi = np.zeros((N, K), dtype=np.complex64)
    
    col_idx = 0
    
    for p in P_list:
        for m in range(M):
            # 基本項：x[n-m] * |x[n-m]|^(p-1)
            x_delayed = np.roll(x, m)
            if m > 0:
                x_delayed[:m] = 0  # 清零無效樣本
            
            abs_x = np.abs(x_delayed)
            gain = abs_x ** (p - 1)
            
            # 正常項
            Phi[:, col_idx] = gain * x_delayed
            col_idx += 1
    
    if with_conj:
        # 共軛項
        for p in P_list:
            for m in range(M):
                x_delayed = np.roll(x, m)
                if m > 0:
                    x_delayed[:m] = 0
                
                abs_x = np.abs(x_delayed)
                gain = abs_x ** (p - 1)
                
                # 共軛項
                Phi[:, col_idx] = gain * np.conj(x_delayed)
                col_idx += 1
    
    return Phi


def build_short_window_features(x, y_res, window_L=13):
    """
    構建短窗特徵（用於 NN 殘差學習）
    
    特徵：{x[n-k], |x|²x[n-k], x*[n-k], x*|x|²[n-k]}, k=0..L-1
    
    Args:
        x: 發送信號 [N] complex
        y_res: 當前殘差 [N] complex（用於對齊檢查）
        window_L: 窗口長度
    
    Returns:
        features: [N, 4*L*2] float32（實部+虛部分開）
    """
    N = len(x)
    L = window_L
    
    # 初始化特徵矩陣
    n_complex_features = 4 * L  # x, |x|²x, x*, x*|x|² 各 L 個
    features_complex = np.zeros((N, n_complex_features), dtype=np.complex64)
    
    col_idx = 0
    
    # 1. x[n-k]
    for k in range(L):
        x_delayed = np.roll(x, k)
        if k > 0:
            x_delayed[:k] = 0
        features_complex[:, col_idx] = x_delayed
        col_idx += 1
    
    # 2. |x|²x[n-k]
    for k in range(L):
        x_delayed = np.roll(x, k)
        if k > 0:
            x_delayed[:k] = 0
        abs_sq = np.abs(x_delayed) ** 2
        features_complex[:, col_idx] = abs_sq * x_delayed
        col_idx += 1
    
    # 3. x*[n-k]
    for k in range(L):
        x_delayed = np.roll(x, k)
        if k > 0:
            x_delayed[:k] = 0
        features_complex[:, col_idx] = np.conj(x_delayed)
        col_idx += 1
    
    # 4. x*|x|²[n-k]
    for k in range(L):
        x_delayed = np.roll(x, k)
        if k > 0:
            x_delayed[:k] = 0
        abs_sq = np.abs(x_delayed) ** 2
        features_complex[:, col_idx] = abs_sq * np.conj(x_delayed)
        col_idx += 1
    
    # 轉換為實數特徵：[real, imag]
    features_real = np.concatenate([
        features_complex.real,
        features_complex.imag
    ], axis=1).astype(np.float32)
    
    return features_real


def normalize_features_dataset(features, stats=None):
    """
    使用資料集統計量正規化特徵（禁止 per-sample）
    
    Args:
        features: [N, D] float32
        stats: {'mean': [1, D], 'std': [1, D]} 或 None
    
    Returns:
        features_norm: [N, D]
        stats: 統計量（若輸入為 None 則計算）
    """
    if stats is None:
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        stats = {'mean': mean, 'std': std}
    else:
        mean = stats['mean']
        std = stats['std']
    
    features_norm = (features - mean) / std
    
    return features_norm, stats


def validate_feature_shape(features, expected_dim=None, name="features"):
    """
    驗證特徵形狀
    
    Args:
        features: [N, D]
        expected_dim: 期望的維度 D（可選）
        name: 特徵名稱
    
    Returns:
        is_valid: bool
    """
    if features.ndim != 2:
        logger.error(f"[{name}] 維度錯誤: {features.ndim}D，期望 2D")
        return False
    
    if expected_dim is not None:
        if features.shape[1] != expected_dim:
            logger.error(f"[{name}] 特徵維度錯誤: {features.shape[1]}，期望 {expected_dim}")
            return False
    
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        logger.error(f"[{name}] 包含 NaN 或 Inf")
        return False
    
    return True