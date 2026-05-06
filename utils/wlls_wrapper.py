"""
WLLS Digital SIC Wrapper - 修正版
主要修正：
1. SINR 計算公式錯誤（分子分母反了）
2. 預設參數改為最佳配置 (L=5, λ=0.01)
3. 新增從 analog meta 讀取期望信號功率以計算精確 SINR
"""

import numpy as np


class WLLSDigitalSIC:
    """WLLS-based Digital SIC（保守版本）"""
    
    def __init__(self, L=5, lambda_reg=0.01, use_widely_linear=False,
                 holdout_ratio=0.2, skip_samples=10, version='conservative'):
        """
        初始化 WLLS Digital SIC
        
        Args:
            L: 通道基底長度（預設 5，最佳配置）
            lambda_reg: 正則化參數（預設 0.01，最佳配置）
            use_widely_linear: 是否使用 widely-linear（預設 False）
            holdout_ratio: Holdout 測試比例
            skip_samples: 跳過前面不穩定樣本
            version: 版本（'conservative' 或 'pure_linear'）
        """
        self.L = L
        self.lambda_reg = lambda_reg
        self.use_widely_linear = use_widely_linear
        self.holdout_ratio = holdout_ratio
        self.skip_samples = skip_samples
        self.version = version
        
        print(f"[WLLSDigitalSIC] 初始化")
        print(f"  版本: {self.version}")
        print(f"  L={self.L}, λ={self.lambda_reg}, widely_linear={self.use_widely_linear}")
        print(f"  holdout={self.holdout_ratio*100:.0f}%, skip={self.skip_samples}")
        print()
    
    def estimate_channel(self, y, x):
        """
        估計通道係數（WLLS）
        
        Args:
            y: 接收信號
            x: 發送信號
        
        Returns:
            h_hat: 估計的通道係數 (L,)
        """
        N_total = len(y)
        N_train = int(N_total * (1 - self.holdout_ratio))
        
        # 訓練窗口（跳過開頭不穩定部分）
        start_idx = self.skip_samples
        end_idx = N_train
        
        y_train = y[start_idx:end_idx]
        x_train = x[start_idx:end_idx]
        N = len(y_train)
        
        # 構造基底矩陣 X (N, L)
        X = np.zeros((N, self.L), dtype=np.complex64)
        for n in range(N):
            for l in range(self.L):
                if n - l >= 0:
                    X[n, l] = x_train[n - l]
        
        # WLLS: h = (X^H X + λI)^{-1} X^H y
        XH_X = X.conj().T @ X
        reg_matrix = self.lambda_reg * np.eye(self.L, dtype=np.complex64)
        XH_y = X.conj().T @ y_train
        
        h_hat = np.linalg.solve(XH_X + reg_matrix, XH_y)
        
        return h_hat
    
    def apply_sic(self, y, x, h):
        """
        應用 Digital SIC
        
        Args:
            y: 接收信號
            x: 發送信號
            h: 通道估計
        
        Returns:
            y_clean: SIC 後信號
            y_si_est: 估計的 SI
        """
        N = len(y)
        y_si_est = np.zeros(N, dtype=np.complex64)
        
        # 估計 SI
        for n in range(N):
            for l in range(len(h)):
                if n - l >= 0:
                    y_si_est[n] += h[l] * x[n - l]
        
        # 消除 SI
        y_clean = y - y_si_est
        
        return y_clean, y_si_est
    
    def compute_metrics(self, y_adc, y_clean, y_si_est, 
                       noise_var, amp_scale,
                       y_si_before_digital=None,
                       P_signal=None,
                       test_window_only=True):
        """
        計算性能指標
        
        Args:
            y_adc: Digital SIC 前信號
            y_clean: Digital SIC 後信號
            y_si_est: 估計的 SI
            noise_var: 噪聲方差（原始）
            amp_scale: Analog 縮放
            y_si_before_digital: SI-only 波形（可選，用於精確計算）
            P_signal: 期望信號功率（從 analog meta 讀取）
            test_window_only: 是否只在 holdout 窗口計算
        
        Returns:
            metrics: 指標字典
        """
        if test_window_only:
            # 只在 holdout 窗口計算
            N_total = len(y_adc)
            N_train = int(N_total * (1 - self.holdout_ratio))
            y_adc_test = y_adc[N_train:]
            y_clean_test = y_clean[N_train:]
            y_si_est_test = y_si_est[N_train:]
            
            if y_si_before_digital is not None:
                y_si_before_test = y_si_before_digital[N_train:]
            else:
                y_si_before_test = None
        else:
            y_adc_test = y_adc
            y_clean_test = y_clean
            y_si_est_test = y_si_est
            y_si_before_test = y_si_before_digital
        
        # 功率計算
        P_before = float(np.mean(np.abs(y_adc_test) ** 2))
        P_after = float(np.mean(np.abs(y_clean_test) ** 2))
        P_si_est = float(np.mean(np.abs(y_si_est_test) ** 2))
        
        # Digital 域噪聲功率
        Pn_digital = 2.0 * noise_var * (amp_scale ** 2)
        
        # Digital Gain (總功率減少)
        digital_gain = float(10 * np.log10((P_before + 1e-20) / (P_after + 1e-20)))
        
        # Digital Supp (SI-only，精確計算)
        if y_si_before_test is not None:
            P_si_before_digital = float(np.mean(np.abs(y_si_before_test) ** 2))
            # SI after = |y_si_before - y_si_est|^2
            si_residual = y_si_before_test - y_si_est_test
            P_si_after_digital = float(np.mean(np.abs(si_residual) ** 2))
            
            if P_si_before_digital > 1e-20 and P_si_after_digital > 1e-20:
                digital_supp_si = float(10 * np.log10(P_si_before_digital / P_si_after_digital))
            else:
                digital_supp_si = 0.0
            
            digital_supp_note = "SI-only (精確)"
        else:
            # 近似：假設 y_si_est 就是 SI before
            P_si_before_digital = P_si_est
            P_si_after_digital = P_after - P_before + P_si_est  # 近似
            
            if P_si_before_digital > 1e-20 and P_si_after_digital > 1e-20:
                digital_supp_si = float(10 * np.log10(P_si_before_digital / P_si_after_digital))
            else:
                digital_supp_si = 0.0
            
            digital_supp_note = "近似"
        
        # ===== 修正 SINR 和 AA 計算 =====
        
        # 殘餘干擾功率（RSI = 總功率 - 噪聲）
        P_rsi_after = max(P_after - Pn_digital, 0.0)
        
        # AA = RSI / Noise
        if Pn_digital > 1e-20 and P_rsi_after > 1e-20:
            AA_approx = float(10 * np.log10(P_rsi_after / Pn_digital))
        else:
            AA_approx = None
        
        # ===== SINR 計算 =====
        # 如果有期望信號功率，計算精確 SINR
        if P_signal is not None and P_signal > 1e-20:
            # 精確計算：SINR = Signal / (RSI + Noise)
            # RSI = P_after - P_signal - Pn_digital
            P_rsi_only = max(P_after - P_signal - Pn_digital, 0.0)
            
            if P_rsi_only + Pn_digital > 1e-20:
                SINR_after = float(10 * np.log10(P_signal / (P_rsi_only + Pn_digital)))
                SINR_note = "精確"
            else:
                SINR_after = None
                SINR_note = "N/A"
        else:
            # ✅ 修正：近似計算 SINR = RSI / Noise
            # 假設 RSI ≈ 期望信號（當 Digital SIC 效果好時）
            if P_rsi_after > 1e-20 and Pn_digital > 1e-20:
                SINR_after = float(10 * np.log10(P_rsi_after / Pn_digital))
                SINR_note = "近似（假設SI已被充分抑制）"
            else:
                SINR_after = None
                SINR_note = "N/A"
        
        metrics = {
            'P_before': P_before,
            'P_after': P_after,
            'P_si_est': P_si_est,
            'P_si_before_digital': P_si_before_digital,
            'P_si_after_digital': P_si_after_digital,
            'Digital_gain': digital_gain,
            'Digital_supp_si': digital_supp_si,
            'Digital_supp_note': digital_supp_note,
            'AA_approx': AA_approx,
            'SINR_after': SINR_after,
            'SINR_note': SINR_note,
            'Pn_digital': Pn_digital,
            'P_rsi_after': P_rsi_after
        }
        
        return metrics
    
    def process(self, y_adc, x_tx, noise_var, amp_scale, 
                y_si_after_analog=None,
                P_signal=None,
                return_full_info=False):
        """
        完整 Digital SIC 流程
        
        Args:
            y_adc: ADC 輸出（analog SIC 後）
            x_tx: 發送信號
            noise_var: 噪聲方差
            amp_scale: Analog 階段縮放
            y_si_after_analog: SI-only 波形（從 analog 階段，可選）
            P_signal: 期望信號功率（從 analog meta）
            return_full_info: 是否返回詳細信息
        
        Returns:
            y_clean: SIC 後信號
            metrics: 指標字典
            (可選) info: 詳細信息
        """
        print(f"\n{'='*60}")
        print(f"執行 Digital SIC (WLLS)")
        print(f"{'='*60}")
        
        # 1. 估計通道
        print(f"[1/3] 估計通道...")
        h_hat = self.estimate_channel(y_adc, x_tx)
        print(f"  ✓ 估計到 {len(h_hat)} tap 通道")
        print(f"  h_hat = {h_hat[:min(3, len(h_hat))]}")
        
        # 2. 應用 SIC
        print(f"[2/3] 應用 Digital SIC...")
        y_clean, y_si_est = self.apply_sic(y_adc, x_tx, h_hat)
        print(f"  ✓ SIC 完成")
        
        # 3. 計算指標
        print(f"[3/3] 計算指標（holdout window）...")
        metrics = self.compute_metrics(
            y_adc, y_clean, y_si_est,
            noise_var, amp_scale,
            y_si_before_digital=y_si_after_analog,
            P_signal=P_signal,
            test_window_only=True
        )
        
        # 添加配置信息到 metrics
        metrics['L'] = self.L
        metrics['lambda'] = self.lambda_reg
        metrics['widely_linear'] = self.use_widely_linear
        
        # 輸出報告
        self._print_report(metrics, noise_var, amp_scale)
        
        if return_full_info:
            info = {
                'h_hat': h_hat,
                'y_si_est': y_si_est,
                'y_clean': y_clean
            }
            return y_clean, metrics, info
        else:
            return y_clean, metrics
    
    def _print_report(self, metrics, noise_var, amp_scale):
        """輸出報告（對齊 run_analog.py 格式）"""
        print(f"\n{'='*60}")
        print(f"===== DIGITAL SIC REPORT =====")
        print(f"{'='*60}")
        print(f"WLLS 配置：L={self.L}, λ={self.lambda_reg}, "
              f"widely_linear={self.use_widely_linear}")
        print(f"測試窗：holdout {self.holdout_ratio*100:.0f}%")
        print(f"")
        print(f"功率分析（holdout window）:")
        print(f"  P_before (analog後): {metrics['P_before']:.6e}")
        print(f"  P_after  (digital後): {metrics['P_after']:.6e}")
        print(f"  P_si_est (估計SI):   {metrics['P_si_est']:.6e}")
        
        if metrics.get('P_si_before_digital') is not None:
            print(f"  P_si (before digital): {metrics['P_si_before_digital']:.6e}")
            print(f"  P_si (after digital):  {metrics['P_si_after_digital']:.6e}")
        
        print(f"")
        print(f"抑制效果:")
        print(f"  Digital_gain (Total):     {metrics['Digital_gain']:>6.2f} dB")
        print(f"  Digital_supp (SI-only):   {metrics['Digital_supp_si']:>6.2f} dB  [{metrics['Digital_supp_note']}]")
        print(f"")
        print(f"殘餘干擾分析:")
        if metrics['AA_approx'] is not None:
            print(f"  AA (RSI/Noise after): {metrics['AA_approx']:>6.2f} dB")
        else:
            print(f"  AA (RSI/Noise after): N/A")
        
        if metrics['SINR_after'] is not None:
            print(f"  SINR after digital:   {metrics['SINR_after']:>6.2f} dB  [{metrics['SINR_note']}]")
        else:
            print(f"  SINR after digital:   N/A")
        print(f"")
        print(f"噪聲底 (digital域): {metrics['Pn_digital']:.6e}")
        print(f"")
        
        if metrics['Digital_supp_note'] == "SI-only (精確)":
            print(f"✅ 使用 SI-only 波形計算，Digital_supp 為精確值")
        else:
            print(f"⚠️  未使用 SI-only 波形，Digital_supp 為近似值")
        
        print(f"{'='*60}\n")


def sweep_wlls_parameters(y_adc, x_tx, noise_var, amp_scale,
                          y_si_after_analog=None,
                          P_signal=None,
                          L_options=[3, 4, 5],
                          lambda_options=[1e-2, 3e-2, 1e-1]):
    """
    掃描 WLLS 參數，找到最佳配置
    
    Args:
        y_adc: ADC 輸出
        x_tx: 發送信號
        noise_var: 噪聲方差
        amp_scale: Analog 縮放
        y_si_after_analog: SI-only 波形（可選）
        P_signal: 期望信號功率
        L_options: L 候選值
        lambda_options: 正則化候選值
    
    Returns:
        best_config: 最佳配置
        all_results: 所有結果
    """
    print(f"\n{'='*60}")
    print(f"WLLS 參數掃描")
    print(f"{'='*60}")
    print(f"L 候選: {L_options}")
    print(f"λ 候選: {lambda_options}")
    print(f"")
    
    all_results = []
    
    for L in L_options:
        for lam in lambda_options:
            print(f"測試: L={L}, λ={lam:.0e}")
            
            sic = WLLSDigitalSIC(
                L=L,
                lambda_reg=lam,
                use_widely_linear=False,
                holdout_ratio=0.2,
                skip_samples=10,
                version='conservative'
            )
            
            _, metrics = sic.process(
                y_adc, x_tx, noise_var, amp_scale,
                y_si_after_analog=y_si_after_analog,
                P_signal=P_signal,
                return_full_info=False
            )
            
            all_results.append({
                'L': L,
                'lambda': lam,
                'Digital_gain': metrics['Digital_gain'],
                'Digital_supp_si': metrics['Digital_supp_si'],
                'AA_approx': metrics['AA_approx']
            })
    
    # 找最佳配置（根據 Digital_supp_si）
    best_idx = np.argmax([r['Digital_supp_si'] for r in all_results])
    best_config = all_results[best_idx]
    
    print(f"\n{'='*60}")
    print(f"最佳配置:")
    print(f"  L = {best_config['L']}")
    print(f"  λ = {best_config['lambda']:.0e}")
    print(f"  Digital_gain = {best_config['Digital_gain']:.2f} dB")
    print(f"  Digital_supp (SI-only) = {best_config['Digital_supp_si']:.2f} dB")
    if best_config['AA_approx'] is not None:
        print(f"  AA (approx) = {best_config['AA_approx']:.2f} dB")
    print(f"{'='*60}\n")
    
    return best_config, all_results