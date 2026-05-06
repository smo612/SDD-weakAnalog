"""SIC/utils.py - 基礎工具函數

提供：
- α 計算（LS 幅度還原）
- 功率與 SINR 計算
- 指標計算
- 通用輔助函數
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_ls_alpha(r_hat, y_res, eps=1e-12):
    """
    計算 LS 幅度還原係數 α
    
    α = <r_hat, y_res> / (<r_hat, r_hat> + ε)
    
    Args:
        r_hat: 預測的干擾形狀 [N] complex
        y_res: 殘差信號 [N] complex
        eps: 防止除零
    
    Returns:
        alpha: 複數標量
    """
    numerator = np.vdot(r_hat, y_res)  # <r_hat, y_res>
    denominator = np.vdot(r_hat, r_hat) + eps  # <r_hat, r_hat>
    alpha = numerator / denominator
    return alpha


def compute_power(signal, eps=1e-12):
    """
    計算信號功率
    
    Args:
        signal: 複數信號 [N]
        eps: 最小值防止 log(0)
    
    Returns:
        power: 浮點數功率
    """
    power = float(np.mean(np.abs(signal) ** 2))
    return max(power, eps)


def compute_suppression_db(power_before, power_after, eps=1e-12):
    """
    計算抑制量（dB）
    
    Supp = 10 * log10(P_before / P_after)
    
    Args:
        power_before: 處理前功率
        power_after: 處理後功率
        eps: 防止 log(0)
    
    Returns:
        supp_db: 抑制量（dB）
    """
    power_before = max(power_before, eps)
    power_after = max(power_after, eps)
    supp_db = 10.0 * np.log10(power_before / power_after)
    return float(supp_db)


def compute_sinr_db(P_signal, P_interference, P_noise, eps=1e-12):
    """
    計算 SINR（dB）
    
    SINR = 10 * log10(P_signal / (P_interference + P_noise))
    
    Args:
        P_signal: 期望信號功率
        P_interference: 干擾功率
        P_noise: 噪聲功率
        eps: 防止 log(0)
    
    Returns:
        sinr_db: SINR（dB），若無效返回 None
    """
    if P_signal is None or P_signal < eps:
        return None
    
    denominator = P_interference + P_noise
    if denominator < eps:
        return None
    
    sinr_db = 10.0 * np.log10(P_signal / denominator)
    return float(sinr_db)


def compute_digital_sic_metrics(y_before, y_after, si_before, si_after, 
                                 r_hat_scaled, P_signal=None, 
                                 P_noise=None, alpha=None):
    """
    計算 Digital SIC 的完整指標
    
    Args:
        y_before: Digital SIC 前信號 [N]
        y_after: Digital SIC 後信號 [N]
        si_before: Digital SIC 前 SI [N]
        si_after: Digital SIC 後 SI [N]（真實殘差）
        r_hat_scaled: 估計的 SI（α * r_hat）[N]
        P_signal: 期望信號功率（可選）
        P_noise: 噪聲功率（可選）
        alpha: LS 係數（可選）
    
    Returns:
        metrics: 指標字典
    """
    eps = 1e-12
    
    # 1. 基本功率
    P_before = compute_power(y_before, eps)
    P_after = compute_power(y_after, eps)
    
    # 2. SI-only 功率
    P_si_before = compute_power(si_before, eps)
    P_si_after = compute_power(si_after, eps)
    
    # 3. Digital 抑制（SI-only）
    Digital_supp_si = compute_suppression_db(P_si_before, P_si_after, eps)
    
    # 4. Total 抑制（SI-only，從 Analog 前到 Digital 後）
    # 注意：這裡的 si_before 是 analog 後的 SI
    Total_supp_SI_only = Digital_supp_si  # 在這個階段就是 Digital supp
    
    # 5. SINR 計算
    if P_signal is not None and P_noise is not None:
        # 殘餘干擾 = P_after - P_signal - P_noise
        P_rsi = max(P_after - P_signal - P_noise, 0.0)
        SINR_after_digital = compute_sinr_db(P_signal, P_rsi, P_noise, eps)
    else:
        SINR_after_digital = None
    
    # 6. α 信息
    alpha_info = None
    if alpha is not None:
        alpha_info = {
            'real': float(alpha.real),
            'imag': float(alpha.imag),
            'abs': float(np.abs(alpha))
        }
    
    metrics = {
        'P_before': P_before,
        'P_after': P_after,
        'P_si_before': P_si_before,
        'P_si_after': P_si_after,
        'Digital_supp_si': Digital_supp_si,
        'Total_supp_SI_only': Total_supp_SI_only,
        'SINR_after_digital': SINR_after_digital,
        'alpha': alpha_info
    }
    
    return metrics


def align_signals(y, x, maxlag=256):
    """
    對齊兩個信號（基於互相關）
    
    Args:
        y: 接收信號 [N]
        x: 發送信號 [N]
        maxlag: 最大搜索延遲
    
    Returns:
        x_aligned: 對齊後的 x [N]
        delay: 延遲量（正數表示 x 需要前移）
    """
    corr = np.correlate(y, np.conj(x), mode='full')
    N = len(x)
    center = len(corr) // 2
    
    search_start = max(0, center - maxlag)
    search_end = min(len(corr), center + maxlag + 1)
    
    corr_window = corr[search_start:search_end]
    peak_idx_window = np.argmax(np.abs(corr_window))
    peak_idx_global = peak_idx_window + search_start
    
    delay = peak_idx_global - N + 1
    
    if delay > 0:
        x_aligned = np.pad(x, (delay, 0), mode='constant')[:-delay]
    elif delay < 0:
        x_aligned = np.pad(x, (0, -delay), mode='constant')[-delay:]
    else:
        x_aligned = x.copy()
    
    return x_aligned, delay


def normalize_signal(signal, method='none', stats=None):
    """
    信號正規化
    
    Args:
        signal: 輸入信號 [N] 或 [N, D]
        method: 'none', 'dataset' (使用全局統計)
        stats: {'mean': ..., 'std': ...} 當 method='dataset' 時使用
    
    Returns:
        signal_norm: 正規化後信號
        stats: 統計量（若 method='dataset' 且 stats=None）
    """
    if method == 'none':
        return signal, None
    
    elif method == 'dataset':
        if stats is None:
            # 計算統計量
            mean = np.mean(signal, axis=0, keepdims=True)
            std = np.std(signal, axis=0, keepdims=True)
            std = np.maximum(std, 1e-6)
            stats = {'mean': mean, 'std': std}
        else:
            mean = stats['mean']
            std = stats['std']
        
        signal_norm = (signal - mean) / std
        return signal_norm, stats
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def safe_divide(numerator, denominator, eps=1e-12, default=0.0):
    """
    安全除法
    
    Args:
        numerator: 分子
        denominator: 分母
        eps: 最小分母值
        default: 當分母過小時的預設值
    
    Returns:
        result: numerator / denominator 或 default
    """
    if np.abs(denominator) < eps:
        return default
    return numerator / denominator


def check_signal_quality(signal, name="signal"):
    """
    檢查信號質量（NaN, Inf, 過大值）
    
    Args:
        signal: 複數信號
        name: 信號名稱（用於 log）
    
    Returns:
        is_valid: bool
    """
    if np.any(np.isnan(signal)):
        logger.warning(f"[{name}] 包含 NaN")
        return False
    
    if np.any(np.isinf(signal)):
        logger.warning(f"[{name}] 包含 Inf")
        return False
    
    max_val = np.max(np.abs(signal))
    if max_val > 1e6:
        logger.warning(f"[{name}] 幅度過大: {max_val:.2e}")
        return False
    
    return True


def add_regularization(matrix, lambda_reg, eps=1e-12):
    """
    添加正則化到矩陣（Ridge）
    
    Args:
        matrix: 方陣 [K, K]
        lambda_reg: 正則化係數
        eps: 額外的數值穩定項
    
    Returns:
        matrix_reg: 正則化後的矩陣
    """
    K = matrix.shape[0]
    reg_matrix = (lambda_reg + eps) * np.eye(K, dtype=matrix.dtype)
    return matrix + reg_matrix