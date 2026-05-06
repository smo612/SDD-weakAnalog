"""SIC/mp.py - Memory Polynomial Digital SIC

實現：
- 平行 Hammerstein / Memory Polynomial
- Ridge-LS 或 RLS
- Block-wise 更新（每 2048 samples）
"""

import numpy as np
import logging
from .features import build_mp_features
from .utils import (
    compute_ls_alpha,
    compute_digital_sic_metrics,
    add_regularization
)

logger = logging.getLogger(__name__)


class MPBackend:
    """Memory Polynomial Digital SIC Backend"""
    
    def __init__(self, poly_orders=[1, 3], memory_len=5, 
                 ridge_lambda=1e-3, block_size=4096, 
                 update_stride=2048, with_conj=True):
        """
        初始化 MP Backend
        
        Args:
            poly_orders: 多項式階數，例如 [1, 3, 5]
            memory_len: 記憶長度 M
            ridge_lambda: Ridge 正則化係數
            block_size: Block 長度
            update_stride: 重新估計間隔
            with_conj: 是否包含共軛項
        """
        self.poly_orders = poly_orders
        self.memory_len = memory_len
        self.ridge_lambda = ridge_lambda
        self.block_size = block_size
        self.update_stride = update_stride
        self.with_conj = with_conj
        
        self.w_mp = None  # MP 係數
        
        # 計算特徵維度
        n_poly = len(poly_orders)
        n_conj = 2 if with_conj else 1
        self.K = n_poly * memory_len * n_conj
        
        logger.info(f"[MP] 初始化:")
        logger.info(f"  多項式階數: {poly_orders}")
        logger.info(f"  記憶長度 M: {memory_len}")
        logger.info(f"  特徵維度 K: {self.K}")
        logger.info(f"  Ridge λ: {ridge_lambda}")
        logger.info(f"  Block 大小: {block_size}, 更新間隔: {update_stride}")
    
    def fit(self, data_dict):
        """
        估計 MP 係數
        
        Args:
            data_dict: {
                'y': 接收信號 [N],
                'x': 發送信號 [N],
                'si_after_analog': SI-only [N] (optional)
            }
        
        Returns:
            self
        """
        y = data_dict['y']
        x = data_dict['x']
        si_after_analog = data_dict.get('si_after_analog', None)
        fit_slice = data_dict.get('fit_slice', None)
        
        # 決定 target
        if si_after_analog is not None:
            target = si_after_analog
            logger.info("[MP] 使用 si_after_analog 作為 target")
        else:
            target = y
            logger.warning("[MP] 無 si_after_analog，使用 y 作為 target")
        
        N = len(y)
        
        # 構建特徵矩陣
        logger.info("[MP] 構建特徵矩陣...")
        Phi = build_mp_features(
            x,
            poly_orders=self.poly_orders,
            memory_len=self.memory_len,
            with_conj=self.with_conj
        )
        
        # 選擇訓練窗口（預設前 80%，也可由外部明確指定 fit_slice）
        if fit_slice is None:
            N_train = int(N * 0.8)
            fit_slice = slice(0, N_train)
        Phi_train = Phi[fit_slice]
        target_train = target[fit_slice]
        
        # Ridge-LS: w = (Phi^H Phi + λI)^{-1} Phi^H target
        logger.info("[MP] 求解 Ridge-LS...")
        PhiH_Phi = Phi_train.conj().T @ Phi_train
        PhiH_Phi_reg = add_regularization(PhiH_Phi, self.ridge_lambda, eps=1e-12)
        PhiH_target = Phi_train.conj().T @ target_train
        
        try:
            self.w_mp = np.linalg.solve(PhiH_Phi_reg, PhiH_target)
        except np.linalg.LinAlgError:
            logger.warning("[MP] 矩陣奇異，使用 pinv")
            self.w_mp = np.linalg.pinv(PhiH_Phi_reg) @ PhiH_target
        
        logger.info(f"[MP] 係數估計完成: |w| = {np.linalg.norm(self.w_mp):.4f}")
        
        return self
    
    def predict(self, batch_dict):
        """
        預測並消除 SI
        
        Args:
            batch_dict: {
                'y': [N],
                'x': [N],
                'si_after_analog': [N],
                'P_signal': float,
                'P_noise': float
            }
        
        Returns:
            r_hat: 估計的 SI [N]
            metrics: 指標字典
        """
        if self.w_mp is None:
            raise RuntimeError("[MP] 必須先呼叫 fit()")
        
        y = batch_dict['y']
        x = batch_dict['x']
        si_after_analog = batch_dict.get('si_after_analog', None)
        P_signal = batch_dict.get('P_signal', None)
        P_noise = batch_dict.get('P_noise', None)
        
        # 構建特徵
        Phi = build_mp_features(
            x,
            poly_orders=self.poly_orders,
            memory_len=self.memory_len,
            with_conj=self.with_conj
        )
        
        # 預測形狀
        r_shape = Phi @ self.w_mp
        
        # LS α 計算
        if si_after_analog is not None:
            y_res = si_after_analog
        else:
            y_res = y
        
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
            logger.warning("[MP] 無 si_after_analog，使用近似 metrics")
        
        metrics['gate_used'] = False  # MP 無 gate
        
        logger.info(f"[MP] Digital Supp: {metrics['Digital_supp_si']:.2f} dB")
        if metrics['SINR_after_digital'] is not None:
            logger.info(f"[MP] SINR: {metrics['SINR_after_digital']:.2f} dB")
        
        return r_hat, metrics


class BlockWiseMPBackend(MPBackend):
    """Block-wise MP Backend（進階版）"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("[BlockWiseMP] 使用 Block-wise 更新")
    
    def fit(self, data_dict):
        """
        Block-wise 估計（每個 block 獨立估計，取平均）
        """
        y = data_dict['y']
        x = data_dict['x']
        si_after_analog = data_dict.get('si_after_analog', None)
        fit_slice = data_dict.get('fit_slice', None)
        
        if si_after_analog is not None:
            target = si_after_analog
        else:
            target = y
        
        N = len(y)

        if fit_slice is not None:
            start = 0 if fit_slice.start is None else fit_slice.start
            stop = N if fit_slice.stop is None else fit_slice.stop
            y = y[start:stop]
            x = x[start:stop]
            target = target[start:stop]
            N = len(y)

        n_blocks = N // self.block_size
        
        if n_blocks == 0:
            logger.warning("[BlockWiseMP] 信號過短，使用單 block")
            return super().fit(data_dict)
        
        logger.info(f"[BlockWiseMP] 分為 {n_blocks} 個 blocks")
        
        w_list = []
        
        for i in range(n_blocks):
            start = i * self.block_size
            end = start + self.block_size
            
            x_block = x[start:end]
            target_block = target[start:end]
            
            # 構建特徵
            Phi_block = build_mp_features(
                x_block,
                poly_orders=self.poly_orders,
                memory_len=self.memory_len,
                with_conj=self.with_conj
            )
            
            # Ridge-LS
            PhiH_Phi = Phi_block.conj().T @ Phi_block
            PhiH_Phi_reg = add_regularization(PhiH_Phi, self.ridge_lambda, eps=1e-12)
            PhiH_target = Phi_block.conj().T @ target_block
            
            try:
                w_block = np.linalg.solve(PhiH_Phi_reg, PhiH_target)
                w_list.append(w_block)
            except np.linalg.LinAlgError:
                logger.warning(f"[BlockWiseMP] Block {i} 奇異，跳過")
                continue
        
        if len(w_list) == 0:
            logger.error("[BlockWiseMP] 所有 blocks 失敗，回退到單 block")
            return super().fit(data_dict)
        
        # 平均所有 blocks 的係數
        self.w_mp = np.mean(w_list, axis=0)
        
        logger.info(f"[BlockWiseMP] 使用 {len(w_list)} 個 blocks 的平均係數")
        logger.info(f"[BlockWiseMP] |w| = {np.linalg.norm(self.w_mp):.4f}")
        
        return self
