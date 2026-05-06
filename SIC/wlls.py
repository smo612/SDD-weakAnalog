"""SIC/wlls.py - WLLS Digital SIC（基準）

從現有 wlls_wrapper.py 抽取，符合統一 API
"""

import numpy as np
import logging
from .utils import (
    compute_ls_alpha,
    compute_digital_sic_metrics,
    add_regularization
)

logger = logging.getLogger(__name__)


class WLLSBackend:
    """WLLS Digital SIC Backend"""
    
    def __init__(self, L=5, lambda_reg=0.01, use_widely_linear=False):
        """
        初始化 WLLS Backend
        
        Args:
            L: 通道長度
            lambda_reg: Ridge 正則化係數
            use_widely_linear: 是否使用 widely-linear（預留，目前未實現）
        """
        self.L = L
        self.lambda_reg = lambda_reg
        self.use_widely_linear = use_widely_linear
        self.h_hat = None
        
        logger.info(f"[WLLS] 初始化: L={L}, λ={lambda_reg}")
    
    def fit(self, data_dict):
        """
        估計通道係數
        
        Args:
            data_dict: {
                'y': 接收信號 [N] complex,
                'x': 發送信號 [N] complex,
                'si_after_analog': SI-only [N] complex (optional)
            }
        
        Returns:
            self
        """
        y = data_dict['y']
        x = data_dict['x']
        
        N = len(y)
        L = self.L
        
        # 構建特徵矩陣 X [N, L]
        X = np.zeros((N, L), dtype=np.complex64)
        for n in range(N):
            for l in range(L):
                if n - l >= 0:
                    X[n, l] = x[n - l]
        
        # WLLS: h = (X^H X + λI)^{-1} X^H y
        XH_X = X.conj().T @ X
        XH_X_reg = add_regularization(XH_X, self.lambda_reg, eps=1e-12)
        XH_y = X.conj().T @ y
        
        try:
            self.h_hat = np.linalg.solve(XH_X_reg, XH_y)
        except np.linalg.LinAlgError:
            logger.warning("[WLLS] 矩陣奇異，使用 pinv")
            self.h_hat = np.linalg.pinv(XH_X_reg) @ XH_y
        
        logger.info(f"[WLLS] 通道估計完成: |h| = {np.linalg.norm(self.h_hat):.4f}")
        
        return self
    
    def predict(self, batch_dict):
        """
        預測並消除 SI
        
        Args:
            batch_dict: {
                'y': 接收信號 [N],
                'x': 發送信號 [N],
                'si_after_analog': SI-only [N] (用於計算 metrics),
                'P_signal': 期望信號功率 (optional),
                'P_noise': 噪聲功率 (optional)
            }
        
        Returns:
            r_hat: 估計的 SI [N] complex
            metrics: 指標字典
        """
        if self.h_hat is None:
            raise RuntimeError("[WLLS] 必須先呼叫 fit()")
        
        y = batch_dict['y']
        x = batch_dict['x']
        si_after_analog = batch_dict.get('si_after_analog', None)
        P_signal = batch_dict.get('P_signal', None)
        P_noise = batch_dict.get('P_noise', None)
        
        N = len(y)
        L = len(self.h_hat)
        
        # 估計 SI
        r_shape = np.zeros(N, dtype=np.complex64)
        for n in range(N):
            for l in range(L):
                if n - l >= 0:
                    r_shape[n] += self.h_hat[l] * x[n - l]
        
        # LS α 計算
        if si_after_analog is not None:
            y_res = si_after_analog  # WLLS 的殘差定義
        else:
            y_res = y  # 近似
        
        alpha = compute_ls_alpha(r_shape, y_res)
        r_hat = alpha * r_shape
        
        # 消除 SI
        y_clean = y - r_hat
        
        # 計算 metrics
        if si_after_analog is not None:
            si_residual = si_after_analog - r_hat
            metrics = compute_digital_sic_metrics(
                y_before=y,
                y_after=y_clean,
                si_before=si_after_analog,
                si_after=si_residual,
                r_hat_scaled=r_hat,
                P_signal=P_signal,
                P_noise=P_noise,
                alpha=alpha
            )
        else:
            # 近似計算
            metrics = {
                'Digital_supp_si': 0.0,
                'Total_supp_SI_only': 0.0,
                'SINR_after_digital': None,
                'gate_used': False,
                'alpha': {
                    'real': float(alpha.real),
                    'imag': float(alpha.imag),
                    'abs': float(np.abs(alpha))
                }
            }
            logger.warning("[WLLS] 無 si_after_analog，使用近似 metrics")
        
        metrics['gate_used'] = False  # WLLS 無 gate
        
        return r_hat, metrics