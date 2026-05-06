#!/usr/bin/env python3
"""
Digital SIC - Unified Runner (MP & WLLS)
支援後端切換: --backend [mp|wlls]
修正: 防止 SINR 為 None 時導致 Crash
"""

import numpy as np
import json
import argparse
from pathlib import Path
import sys

# 將專案根目錄加入路徑
sys.path.append(str(Path(__file__).parent.parent))

from utils.wlls_wrapper import WLLSDigitalSIC, sweep_wlls_parameters
from utils.sic_eval import build_unified_digital_metrics
from SIC.features import build_mp_features

# 嘗試導入 MP
try:
    from SIC.mp import MPBackend
    HAS_MP = True
except ImportError:
    print("⚠️ 找不到 SIC.mp 模組，MP 後端將不可用")
    HAS_MP = False

# ✅ 導入導頻校正工具 (選擇性)
try:
    from utils.pilot_correction import pilot_based_correction
    PILOT_CORRECTION_AVAILABLE = True
except ImportError:
    PILOT_CORRECTION_AVAILABLE = False


def load_analog_output(bridge_dir='bridge'):
    """載入 analog 階段輸出"""
    bridge_path = Path(bridge_dir)
    y_adc = np.load(bridge_path / 'y_adc.npy')
    
    # 載入 SI-only 波形（如果存在）
    # 嘗試多種可能的命名
    possible_names = ['y_si_after_analog.npy', 'y_rsi_after_analog.npy']
    y_si_after_analog = None
    for name in possible_names:
        p = bridge_path / name
        if p.exists():
            y_si_after_analog = np.load(p)
            break
            
    with open(bridge_path / 'meta.json', 'r') as f:
        meta = json.load(f)
        
    P_signal = meta.get('P_main', None)
    return y_adc, y_si_after_analog, meta, P_signal


def load_tx_signal(bridge_tx_dir='bridge_tx'):
    """載入 TX 信號"""
    path = Path(bridge_tx_dir)
    x_tx = np.load(path / 'x_tx.npy')
    return x_tx


def align_lengths(y, x):
    n_min = min(len(y), len(x))
    return y[:n_min], x[:n_min]


def parse_poly_orders(order_str):
    values = []
    for raw in str(order_str).split(','):
        raw = raw.strip()
        if raw:
            values.append(int(raw))
    if not values:
        raise ValueError("poly_orders must contain at least one integer")
    return values


def parse_float_grid(grid_str):
    values = []
    for raw in str(grid_str).split(','):
        raw = raw.strip()
        if raw:
            values.append(float(raw))
    if not values:
        raise ValueError("grid string must contain at least one numeric value")
    return values


def resolve_fit_slice(num_samples, fit_window, fit_prefix_samples):
    if fit_window == 'legacy':
        fit_len = max(1, int(num_samples * 0.8))
        return slice(0, fit_len), fit_len

    if fit_window == 'full':
        return slice(0, num_samples), num_samples

    fit_len = max(1, min(int(fit_prefix_samples), int(num_samples)))
    return slice(0, fit_len), fit_len


def _signal_std_abs(signal):
    if signal is None:
        return 0.0
    if len(signal) == 0:
        return 0.0
    return float(np.std(np.abs(signal)))


def _compute_internal_scale(*signals):
    scales = [_signal_std_abs(sig) for sig in signals if sig is not None]
    return max(max(scales, default=0.0), 1e-8)


def _normalize_complex_signal(signal, scale_ref):
    if signal is None:
        return None
    return (signal / scale_ref).astype(np.complex64)


def _restore_physical_scale(signal, scale_ref):
    if signal is None:
        return None
    return (signal * scale_ref).astype(np.complex64)


def _iq_imbalance_widely_linear(x, iq_imbal, iq_phase_rad):
    alpha = (1.0 + iq_imbal) * np.exp(1j * iq_phase_rad / 2.0)
    beta = (1.0 - iq_imbal) * np.exp(-1j * iq_phase_rad / 2.0)
    return 0.5 * (alpha * x + beta * np.conj(x))


def _rapp_pa(x, asat, p):
    r = np.abs(x)
    denom = (1.0 + (r / (asat + 1e-18)) ** (2.0 * p)) ** (1.0 / (2.0 * p))
    return x / (denom + 1e-18)


def _apply_backoff(x, bo_db):
    return x * (10 ** (-bo_db / 20.0))


def _main_pa_reference_from_params(x_tx, iq_imbal, iq_phase_rad, bo_db, pa_p, asat_main):
    x_bo = _apply_backoff(x_tx.astype(np.complex64), bo_db).astype(np.complex64)
    x_iq = _iq_imbalance_widely_linear(x_bo, iq_imbal, iq_phase_rad).astype(np.complex64)
    x_pa = _rapp_pa(x_iq, asat_main, pa_p).astype(np.complex64)

    return x_iq, x_pa


def _main_pa_reference_from_config(x_tx):
    import config as C

    iq_imbal = float(getattr(C, "IQ_IMBALANCE", getattr(C, "RSI_IQ_AMBAL", 0.02)))
    if hasattr(C, "IQ_PHASE"):
        iq_phase_rad = float(getattr(C, "IQ_PHASE"))
    else:
        iq_phase_deg = float(getattr(C, "RSI_IQ_PHERR_DEG", 2.0))
        iq_phase_rad = np.deg2rad(iq_phase_deg)
    bo_db = float(getattr(C, "BO_DB", 0.0))
    pa_p = float(getattr(C, "PA_P", getattr(C, "RSI_RAPP_P", 2.0)))
    # Use RSI_RAPP_Asat directly (absolute saturation amplitude, matches the analog simulator)
    asat_main = float(getattr(C, "RSI_RAPP_Asat", 3.0))

    x_iq, x_pa = _main_pa_reference_from_params(
        x_tx=x_tx,
        iq_imbal=iq_imbal,
        iq_phase_rad=iq_phase_rad,
        bo_db=bo_db,
        pa_p=pa_p,
        asat_main=asat_main,
    )

    return x_iq, x_pa, {
        "iq_imbalance": iq_imbal,
        "iq_phase_rad": iq_phase_rad,
        "bo_db": bo_db,
        "pa_p": pa_p,
        "asat_main": asat_main,
    }


def _solve_ridge_ls(Phi_train, target_train, lambda_reg):
    phi_h_phi = Phi_train.conj().T @ Phi_train
    lam_I = lambda_reg * np.eye(Phi_train.shape[1], dtype=np.complex128)
    phi_h_phi_reg = phi_h_phi.astype(np.complex128) + lam_I
    phi_h_tgt = Phi_train.conj().T @ target_train

    try:
        return np.linalg.solve(phi_h_phi_reg, phi_h_tgt.astype(np.complex128))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(phi_h_phi_reg) @ phi_h_tgt.astype(np.complex128)


def _compute_physics_metrics(y_si_after_analog, r_hat, P_signal, noise_var, sinr_analog_val):
    sinr_after_physics = None
    digital_supp_si = None

    if y_si_after_analog is not None:
        si_res = y_si_after_analog - r_hat
        P_si_res = float(np.mean(np.abs(si_res) ** 2))
        P_sig = float(P_signal) if P_signal is not None else 1.0
        sinr_after_physics = float(
            10 * np.log10(P_sig / (P_si_res + noise_var + 1e-12))
        )
        P_si_before = float(np.mean(np.abs(y_si_after_analog) ** 2))
        digital_supp_si = float(
            10 * np.log10(P_si_before / (P_si_res + 1e-12))
        )

    digital_gain = (
        sinr_after_physics - sinr_analog_val
        if sinr_after_physics is not None else 0.0
    )
    return sinr_after_physics, digital_supp_si, digital_gain


def to_json_serializable(value):
    if value is None:
        return None
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.complex64) or isinstance(value, np.complex128):
        return {'real': float(value.real), 'imag': float(value.imag)}
    else:
        return value


def save_digital_output(y_clean, h_hat, metrics, meta_analog, backend_name, output_dir='bridge_digital'):
    """保存 Digital SIC 輸出"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path / 'y_clean.npy', y_clean)
    if h_hat is not None:
        np.save(output_path / 'h_hat.npy', h_hat)
    
    # 計算總抑制
    analog_supp = meta_analog.get('Supp_analog', 0.0)
    digital_supp_si = metrics.get('Digital_supp_si', 0.0)
    
    total_supp_si_only = None
    if analog_supp is not None and digital_supp_si is not None:
        try:
            total_supp_si_only = float(analog_supp + digital_supp_si)
        except:
            total_supp_si_only = 0.0
    
    sinr_final_physics = metrics.get('SINR_after_digital')
    if sinr_final_physics is None:
        sinr_final_physics = metrics.get('SINR_after')

    extra_metrics = {
        'snr_db': to_json_serializable(meta_analog.get('snr_db')),
        'Supp_analog': to_json_serializable(analog_supp),
        'Digital_gain': to_json_serializable(metrics.get('Digital_gain')),
        'Digital_gain_physics': to_json_serializable(metrics.get('Digital_gain')),
        'Digital_supp_si': to_json_serializable(digital_supp_si),
        'SINR_after_digital_physics': to_json_serializable(sinr_final_physics),
        'Total_supp_SI_only': to_json_serializable(total_supp_si_only),
        'params': {
            'L': metrics.get('L'),
            'lambda': metrics.get('lambda'),
            'poly_orders': metrics.get('poly_orders'),
            'fit_target': metrics.get('fit_target'),
            'fit_window': metrics.get('fit_window'),
            'fit_prefix_samples': metrics.get('fit_prefix_samples'),
        },
        'pilot_correction': metrics.get('pilot_correction_info'),
        'reference_mode': metrics.get('reference_mode'),
        'reference_info': metrics.get('reference_info'),
    }

    meta_digital = build_unified_digital_metrics(
        backend_name=backend_name,
        y_clean=y_clean,
        extra_metrics=extra_metrics,
    )
    
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(meta_digital, f, indent=2)
    
    print(f"\n✅ {backend_name} Digital SIC 完成")
    print(f"  Analog Gain: {analog_supp:.2f} dB")
    print(f"  Digital Gain: {metrics.get('Digital_gain', 0):.2f} dB")
    if sinr_final_physics is not None:
        print(f"  Final SINR (physics): {sinr_final_physics:.2f} dB")
    else:
        print(f"  Final SINR (physics): N/A (SI reference missing)")
    if meta_digital.get('symbol_sinr_clean_db') is not None:
        print(f"  Symbol SINR: {meta_digital['symbol_sinr_clean_db']:.2f} dB")
    if meta_digital.get('mse_gain_db') is not None:
        print(f"  MSE gain: {meta_digital['mse_gain_db']:.2f} dB")


def main():
    parser = argparse.ArgumentParser(description='Digital SIC Runner')
    
    # 新增 backend 參數
    parser.add_argument('--backend', type=str, default='wlls',
                       choices=['wlls', 'mp', 'hammerstein', 'hammerstein_v2', 'hammerstein_v2_est'],
                       help='選擇數位消除演算法: wlls / mp / hammerstein')

    # 通用參數
    parser.add_argument('--bridge-dir', type=str, default='bridge')
    parser.add_argument('--bridge-tx-dir', type=str, default='bridge_tx')
    parser.add_argument('--output-dir', type=str, default='bridge_digital')
    
    # 演算法參數
    parser.add_argument('--L', type=int, default=5, help='記憶長度/通道階數')
    parser.add_argument('--lambda-reg', type=float, default=0.01, help='正則化係數')
    parser.add_argument('--widely-linear', action='store_true', default=True)
    parser.add_argument('--poly-orders', type=str, default='1,3,5')
    parser.add_argument('--fit-target', type=str, default='si_oracle',
                       choices=['si_oracle', 'observation'])
    parser.add_argument('--fit-window', type=str, default='legacy',
                       choices=['legacy', 'full', 'prefix'],
                       help='Coefficient fitting window: legacy(80%%), full packet, or prefix-only calibration.')
    parser.add_argument('--fit-prefix-samples', type=int, default=1024,
                       help='Number of prefix samples used when --fit-window prefix.')
    parser.add_argument('--digital-internal-normalize', action='store_true',
                       help='Normalize signals internally inside supported digital backends, then rescale outputs back to physical amplitude.')
    parser.add_argument('--est-iq-grid', type=str, default='0.00,0.01,0.03,0.05',
                       help='Coarse IQ-imbalance candidate grid for hammerstein_v2_est.')
    parser.add_argument('--est-phase-grid-deg', type=str, default='0.0,1.0,3.0,5.0',
                       help='Coarse IQ phase-error candidate grid (degrees) for hammerstein_v2_est.')
    parser.add_argument('--est-asat-grid', type=str, default='2.0,2.5,3.5,4.0',
                       help='Coarse PA saturation candidate grid for hammerstein_v2_est.')
    parser.add_argument('--est-val-ratio', type=float, default=0.25,
                       help='Validation ratio inside the calibration prefix for hammerstein_v2_est candidate selection.')
    
    # 導頻校正
    parser.add_argument('--pilot-correction', action='store_true', default=True)
    parser.add_argument('--pilot-period', type=int, default=64)
    parser.add_argument('--n-pilots', type=int, default=127)
    
    # WLLS 專用
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--holdout-ratio', type=float, default=0.2)
    parser.add_argument('--skip-samples', type=int, default=64)
    parser.add_argument('--version', type=str, default='conservative')

    args = parser.parse_args()
    
    print("="*60)
    print(f"Digital SIC Runner - Backend: {args.backend.upper()}")
    print("="*60)

    # 1. 載入數據
    y_adc, y_si_after_analog, meta_analog, P_signal = load_analog_output(args.bridge_dir)
    x_tx = load_tx_signal(args.bridge_tx_dir)
    y_adc, x_tx = align_lengths(y_adc, x_tx)
    
    if y_si_after_analog is not None:
        n_min = min(len(y_adc), len(y_si_after_analog))
        y_si_after_analog = y_si_after_analog[:n_min]
        y_adc = y_adc[:n_min]
        x_tx = x_tx[:n_min]
    
    # 2. 導頻校正 (共通步驟)
    pilot_info = None
    if args.pilot_correction and PILOT_CORRECTION_AVAILABLE:
        try:
            print("[導頻校正] 執行中...")
            y_adc, alpha_est = pilot_based_correction(y_adc, args.pilot_period, args.n_pilots)
            if y_si_after_analog is not None:
                y_si_after_analog = y_si_after_analog / alpha_est
            pilot_info = {'enabled': True, 'alpha_abs': float(np.abs(alpha_est))}
            print("  ✅ 校正完成")
        except Exception as e:
            print(f"  ⚠️ 校正失敗: {e}")
            pilot_info = {'enabled': False, 'error': str(e)}

    noise_var = meta_analog.get('noise_var', 0.0)
    amp_scale = meta_analog.get('amp_scale', 1.0)
    poly_orders = parse_poly_orders(args.poly_orders)
    use_oracle_target = (args.fit_target == 'si_oracle')
    fit_slice, fit_sample_count = resolve_fit_slice(len(y_adc), args.fit_window, args.fit_prefix_samples)
    internal_norm_supported = {'mp', 'hammerstein_v2'}
    use_internal_norm = bool(args.digital_internal_normalize and args.backend in internal_norm_supported)
    internal_scale_ref = None

    if args.digital_internal_normalize and not use_internal_norm:
        print(f"[INFO] --digital-internal-normalize is currently implemented only for {sorted(internal_norm_supported)}.")

    # 3. 執行選定的後端
    if args.backend == 'mp':
        if not HAS_MP:
            raise ImportError("無法載入 MP Backend")
            
        print(f"[MP] 開始執行 Memory Polynomial SIC (Order={poly_orders}, M={args.L})...")
        if args.fit_window == 'prefix':
            print(f"[MP] Fitting on prefix only: first {fit_sample_count} samples.")
        elif args.fit_window == 'full':
            print("[MP] Fitting on the full packet.")
        else:
            print(f"[MP] Fitting on the legacy window: first {fit_sample_count} samples.")
        
        # 初始化 MP
        mp = MPBackend(
            poly_orders=poly_orders,
            memory_len=args.L, 
            ridge_lambda=args.lambda_reg
        )
        
        # 準備資料
        x_fit = x_tx
        y_fit = y_adc
        si_fit = y_si_after_analog if use_oracle_target else None
        if use_internal_norm:
            internal_scale_ref = _compute_internal_scale(x_tx, y_adc, si_fit)
            print(f"[MP] Internal normalization enabled (scale_ref={internal_scale_ref:.6e}).")
            x_fit = _normalize_complex_signal(x_tx, internal_scale_ref)
            y_fit = _normalize_complex_signal(y_adc, internal_scale_ref)
            si_fit = _normalize_complex_signal(si_fit, internal_scale_ref)

        data_train = {
            'y': y_fit,
            'x': x_fit,
            'si_after_analog': si_fit,
            'fit_slice': fit_slice,
        }
        
        # 訓練
        mp.fit(data_train)
        
        # 預測
        batch_predict = {
            'y': y_fit,
            'x': x_fit,
            'si_after_analog': si_fit,
            'P_signal': P_signal,
            'P_noise': noise_var
        }

        r_hat_internal, _predict_metrics = mp.predict(batch_predict)
        if use_internal_norm:
            r_hat = _restore_physical_scale(r_hat_internal, internal_scale_ref)
        else:
            r_hat = r_hat_internal.astype(np.complex64)
        h_hat = np.array([0]) 

        sinr_analog = meta_analog.get('SINR_analog', 0.0) or 0.0
        sinr_after_physics, digital_supp_si, digital_gain = _compute_physics_metrics(
            y_si_after_analog=y_si_after_analog,
            r_hat=r_hat,
            P_signal=P_signal,
            noise_var=noise_var,
            sinr_analog_val=sinr_analog,
        )

        metrics = {
            'SINR_after_digital': sinr_after_physics,
            'SINR_after_digital_physics': sinr_after_physics,
            'Digital_gain': digital_gain,
            'Digital_supp_si': digital_supp_si,
        }

        metrics['pilot_correction_info'] = pilot_info
        metrics['L'] = args.L
        metrics['lambda'] = args.lambda_reg
        metrics['poly_orders'] = poly_orders
        metrics['fit_target'] = args.fit_target
        metrics['fit_window'] = args.fit_window
        metrics['fit_prefix_samples'] = fit_sample_count
        metrics['internal_normalization_enabled'] = use_internal_norm
        metrics['internal_scale_ref'] = internal_scale_ref
        metrics['internal_scale_policy'] = (
            'max_std_abs(x_tx,y_adc,target)' if use_internal_norm else None
        )

        y_clean = y_adc - r_hat

    elif args.backend == 'hammerstein':
        import config as C

        print(f"[Hammerstein] RAPP(p={C.RSI_RAPP_P}, Asat={C.RSI_RAPP_Asat}), "
              f"WL FIR M={args.L}, lambda={args.lambda_reg}")

        # 1. 用已知 PA 參數計算 post-PA 參考訊號 x_pa = RAPP(x_tx)
        r = np.abs(x_tx)
        x_pa = (x_tx / np.maximum(
            (1.0 + (r / C.RSI_RAPP_Asat) ** (2.0 * C.RSI_RAPP_P))
            ** (1.0 / (2.0 * C.RSI_RAPP_P)),
            1e-18
        )).astype(np.complex64)

        M = args.L
        N = len(x_pa)

        # 2. 建 widely-linear delay matrix [x_pa taps, conj(x_pa) taps]
        cols = []
        for m in range(M):
            d = np.roll(x_pa, m).astype(np.complex64)
            if m > 0:
                d[:m] = 0
            cols.append(d)
        for m in range(M):
            d = np.roll(x_pa, m).astype(np.complex64)
            if m > 0:
                d[:m] = 0
            cols.append(np.conj(d))
        Phi = np.column_stack(cols)  # [N, 2*M]

        # 3. 選估測目標（同 MP：有 SI reference 用 SI，沒有用 y_adc）
        if use_oracle_target and y_si_after_analog is not None:
            target = y_si_after_analog.astype(np.complex64)
            print("[Hammerstein] Using si_after_analog as LS target.")
        else:
            target = y_adc.astype(np.complex64)
            print("[Hammerstein] Using observed y_adc as LS target.")

        if args.fit_window == 'prefix':
            print(f"[Hammerstein] Fitting on prefix only: first {fit_sample_count} samples.")
        elif args.fit_window == 'full':
            print("[Hammerstein] Fitting on the full packet.")
        else:
            print(f"[Hammerstein] Fitting on the legacy window: first {fit_sample_count} samples.")

        # 4. Ridge-LS，支援 full-packet 或 prefix-only 校正
        Phi_tr = Phi[fit_slice]
        tgt_tr = target[fit_slice]

        PhiH_Phi = Phi_tr.conj().T @ Phi_tr
        lam_I = args.lambda_reg * np.eye(2 * M, dtype=np.complex128)
        PhiH_Phi_reg = PhiH_Phi.astype(np.complex128) + lam_I
        PhiH_tgt = Phi_tr.conj().T @ tgt_tr

        try:
            w = np.linalg.solve(PhiH_Phi_reg, PhiH_tgt.astype(np.complex128))
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(PhiH_Phi_reg) @ PhiH_tgt.astype(np.complex128)

        # 5. 消除
        r_hat = (Phi @ w.astype(np.complex64)).astype(np.complex64)
        y_clean = (y_adc - r_hat).astype(np.complex64)
        h_hat = w

        # 6. Physics SINR（同 MP 計算方式）
        sinr_after_physics = None
        digital_supp_si = None
        if y_si_after_analog is not None:
            si_res = y_si_after_analog - r_hat
            P_si_res = float(np.mean(np.abs(si_res) ** 2))
            P_sig = float(P_signal) if P_signal is not None else 1.0
            sinr_after_physics = float(
                10 * np.log10(P_sig / (P_si_res + noise_var + 1e-12))
            )
            P_si_before = float(np.mean(np.abs(y_si_after_analog) ** 2))
            digital_supp_si = float(
                10 * np.log10(P_si_before / (P_si_res + 1e-12))
            )

        sinr_analog_val = meta_analog.get('SINR_analog', 0.0) or 0.0
        digital_gain = (
            sinr_after_physics - sinr_analog_val
            if sinr_after_physics is not None else 0.0
        )

        metrics = {
            'SINR_after_digital': sinr_after_physics,
            'SINR_after_digital_physics': sinr_after_physics,
            'Digital_gain': digital_gain,
            'Digital_supp_si': digital_supp_si,
            'L': M,
            'lambda': args.lambda_reg,
            'poly_orders': [1],
            'fit_target': args.fit_target,
            'fit_window': args.fit_window,
            'fit_prefix_samples': fit_sample_count,
            'pilot_correction_info': pilot_info,
        }

    elif args.backend == 'hammerstein_v2':
        print(f"[Hammerstein v2] IQ -> RAPP -> MP basis, orders={poly_orders}, M={args.L}, lambda={args.lambda_reg}")

        x_iq, x_pa, ref_info = _main_pa_reference_from_config(x_tx)
        if use_oracle_target and y_si_after_analog is not None:
            target = y_si_after_analog.astype(np.complex64)
            print("[Hammerstein v2] Using si_after_analog as LS target.")
        else:
            target = y_adc.astype(np.complex64)
            print("[Hammerstein v2] Using observed y_adc as LS target.")

        x_pa_fit = x_pa
        target_fit = target
        if use_internal_norm:
            internal_scale_ref = _compute_internal_scale(x_pa, y_adc, target)
            print(f"[Hammerstein v2] Internal normalization enabled (scale_ref={internal_scale_ref:.6e}).")
            x_pa_fit = _normalize_complex_signal(x_pa, internal_scale_ref)
            target_fit = _normalize_complex_signal(target, internal_scale_ref)

        Phi = build_mp_features(
            x_pa_fit,
            poly_orders=poly_orders,
            memory_len=args.L,
            with_conj=True,
        ).astype(np.complex64)

        if args.fit_window == 'prefix':
            print(f"[Hammerstein v2] Fitting on prefix only: first {fit_sample_count} samples.")
        elif args.fit_window == 'full':
            print("[Hammerstein v2] Fitting on the full packet.")
        else:
            print(f"[Hammerstein v2] Fitting on the legacy window: first {fit_sample_count} samples.")

        Phi_tr = Phi[fit_slice]
        tgt_tr = target_fit[fit_slice]

        PhiH_Phi = Phi_tr.conj().T @ Phi_tr
        lam_I = args.lambda_reg * np.eye(Phi.shape[1], dtype=np.complex128)
        PhiH_Phi_reg = PhiH_Phi.astype(np.complex128) + lam_I
        PhiH_tgt = Phi_tr.conj().T @ tgt_tr

        try:
            w = np.linalg.solve(PhiH_Phi_reg, PhiH_tgt.astype(np.complex128))
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(PhiH_Phi_reg) @ PhiH_tgt.astype(np.complex128)

        r_hat_internal = (Phi @ w.astype(np.complex64)).astype(np.complex64)
        if use_internal_norm:
            r_hat = _restore_physical_scale(r_hat_internal, internal_scale_ref)
        else:
            r_hat = r_hat_internal.astype(np.complex64)
        y_clean = (y_adc - r_hat).astype(np.complex64)
        h_hat = w

        sinr_after_physics = None
        digital_supp_si = None
        if y_si_after_analog is not None:
            si_res = y_si_after_analog - r_hat
            P_si_res = float(np.mean(np.abs(si_res) ** 2))
            P_sig = float(P_signal) if P_signal is not None else 1.0
            sinr_after_physics = float(
                10 * np.log10(P_sig / (P_si_res + noise_var + 1e-12))
            )
            P_si_before = float(np.mean(np.abs(y_si_after_analog) ** 2))
            digital_supp_si = float(
                10 * np.log10(P_si_before / (P_si_res + 1e-12))
            )

        sinr_analog_val = meta_analog.get('SINR_analog', 0.0) or 0.0
        digital_gain = (
            sinr_after_physics - sinr_analog_val
            if sinr_after_physics is not None else 0.0
        )

        metrics = {
            'SINR_after_digital': sinr_after_physics,
            'SINR_after_digital_physics': sinr_after_physics,
            'Digital_gain': digital_gain,
            'Digital_supp_si': digital_supp_si,
            'L': args.L,
            'lambda': args.lambda_reg,
            'poly_orders': poly_orders,
            'fit_target': args.fit_target,
            'fit_window': args.fit_window,
            'fit_prefix_samples': fit_sample_count,
            'reference_mode': 'iq_then_rapp_then_mp_basis',
            'reference_info': ref_info,
            'pilot_correction_info': pilot_info,
            'internal_normalization_enabled': use_internal_norm,
            'internal_scale_ref': internal_scale_ref,
            'internal_scale_policy': (
                'max_std_abs(x_ref,y_adc,target)' if use_internal_norm else None
            ),
        }

    elif args.backend == 'hammerstein_v2_est':
        import itertools
        import config as C

        iq_grid = parse_float_grid(args.est_iq_grid)
        phase_grid_deg = parse_float_grid(args.est_phase_grid_deg)
        asat_grid = parse_float_grid(args.est_asat_grid)

        bo_db = float(getattr(C, "BO_DB", 0.0))
        pa_p = float(getattr(C, "PA_P", getattr(C, "RSI_RAPP_P", 2.0)))

        print(
            f"[Hammerstein v2 est] Coarse hardware search, orders={poly_orders}, "
            f"M={args.L}, lambda={args.lambda_reg}"
        )
        print(
            f"[Hammerstein v2 est] iq_grid={iq_grid}, phase_grid_deg={phase_grid_deg}, "
            f"asat_grid={asat_grid}, val_ratio={args.est_val_ratio}"
        )

        if use_oracle_target and y_si_after_analog is not None:
            target = y_si_after_analog.astype(np.complex64)
            print("[Hammerstein v2 est] Using si_after_analog as calibration target.")
        else:
            target = y_adc.astype(np.complex64)
            print("[Hammerstein v2 est] Using observed y_adc as calibration target.")

        cal_start = 0 if fit_slice.start is None else fit_slice.start
        cal_stop = len(target) if fit_slice.stop is None else fit_slice.stop
        cal_len = max(1, cal_stop - cal_start)

        val_len = max(1, int(round(cal_len * args.est_val_ratio)))
        if val_len >= cal_len:
            val_len = max(1, cal_len // 2)
        train_stop = cal_stop - val_len
        if train_stop <= cal_start:
            train_stop = min(cal_stop, cal_start + max(1, cal_len // 2))
        fit_slice_train = slice(cal_start, train_stop)
        fit_slice_val = slice(train_stop, cal_stop)

        best = None
        n_candidates = 0
        for iq_imbal, phase_deg, asat_main in itertools.product(iq_grid, phase_grid_deg, asat_grid):
            iq_phase_rad = np.deg2rad(phase_deg)
            _, x_pa = _main_pa_reference_from_params(
                x_tx=x_tx,
                iq_imbal=iq_imbal,
                iq_phase_rad=iq_phase_rad,
                bo_db=bo_db,
                pa_p=pa_p,
                asat_main=asat_main,
            )
            Phi = build_mp_features(
                x_pa,
                poly_orders=poly_orders,
                memory_len=args.L,
                with_conj=True,
            ).astype(np.complex64)

            w = _solve_ridge_ls(Phi[fit_slice_train], target[fit_slice_train], args.lambda_reg)
            pred_val = (Phi[fit_slice_val] @ w.astype(np.complex64)).astype(np.complex64)
            tgt_val = target[fit_slice_val]
            val_mse = float(np.mean(np.abs(tgt_val - pred_val) ** 2))
            n_candidates += 1

            if best is None or val_mse < best['val_mse']:
                best = {
                    'val_mse': val_mse,
                    'iq_imbalance': float(iq_imbal),
                    'phase_deg': float(phase_deg),
                    'iq_phase_rad': float(iq_phase_rad),
                    'asat_main': float(asat_main),
                    'Phi': Phi,
                }

        if best is None:
            raise RuntimeError("No candidate evaluated in hammerstein_v2_est")

        print(
            "[Hammerstein v2 est] Selected coarse params: "
            f"iq={best['iq_imbalance']}, phase_deg={best['phase_deg']}, "
            f"asat={best['asat_main']}, val_mse={best['val_mse']:.6e}"
        )

        Phi = best['Phi']
        w = _solve_ridge_ls(Phi[fit_slice], target[fit_slice], args.lambda_reg)
        r_hat = (Phi @ w.astype(np.complex64)).astype(np.complex64)
        y_clean = (y_adc - r_hat).astype(np.complex64)
        h_hat = w

        sinr_analog_val = meta_analog.get('SINR_analog', 0.0) or 0.0
        sinr_after_physics, digital_supp_si, digital_gain = _compute_physics_metrics(
            y_si_after_analog=y_si_after_analog,
            r_hat=r_hat,
            P_signal=P_signal,
            noise_var=noise_var,
            sinr_analog_val=sinr_analog_val,
        )

        metrics = {
            'SINR_after_digital': sinr_after_physics,
            'SINR_after_digital_physics': sinr_after_physics,
            'Digital_gain': digital_gain,
            'Digital_supp_si': digital_supp_si,
            'L': args.L,
            'lambda': args.lambda_reg,
            'poly_orders': poly_orders,
            'fit_target': args.fit_target,
            'fit_window': args.fit_window,
            'fit_prefix_samples': fit_sample_count,
            'reference_mode': 'coarse_estimated_iq_then_rapp_then_mp_basis',
            'reference_info': {
                'selected_iq_imbalance': best['iq_imbalance'],
                'selected_phase_deg': best['phase_deg'],
                'selected_asat_main': best['asat_main'],
                'candidate_count': n_candidates,
                'candidate_grids': {
                    'iq_imbalance': iq_grid,
                    'phase_deg': phase_grid_deg,
                    'asat_main': asat_grid,
                },
                'selection_val_mse': best['val_mse'],
                'selection_train_samples': int(train_stop - cal_start),
                'selection_val_samples': int(cal_stop - train_stop),
                'bo_db': bo_db,
                'pa_p': pa_p,
            },
            'pilot_correction_info': pilot_info,
        }

    elif args.backend == 'wlls':
        print(f"[WLLS] 開始執行 (L={args.L}, lambda={args.lambda_reg})...")
        sic = WLLSDigitalSIC(
            L=args.L,
            lambda_reg=args.lambda_reg,
            use_widely_linear=args.widely_linear,
            version=args.version
        )
        
        y_clean, metrics, info = sic.process(
            y_adc, x_tx, noise_var, amp_scale,
            y_si_after_analog=y_si_after_analog,
            P_signal=P_signal,
            return_full_info=True
        )
        h_hat = info['h_hat']
        metrics['pilot_correction_info'] = pilot_info

    # 4. 保存結果
    save_digital_output(
        y_clean, h_hat, metrics, meta_analog, 
        backend_name=args.backend.upper(),
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
