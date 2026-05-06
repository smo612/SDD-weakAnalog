# sdd_channel_model_v5.py
# v5_postpa: Active Analog Cancellation using post-PA reference
#
# Patch v5.1:
#  - Keep AUX_DISABLE_IQPA default True (avoid double distortion)
#  - Fix safety flip (corr<0 -> flip)
#  - NEW: when PA is OFF (linear SI), force ASIC_P_use=1 to avoid ill-conditioned
#         high-order nonlinear basis exploding the LS solution.
#
# Patch v5.2:
#  - Modified AUX_DISABLE_IQPA=False behavior: Aux-TX ONLY includes IQ Imbalance,
#    completely bypassing Aux PA to match ASIC paper Fig 1 and prevent double distortion.
#
# Patch v5.3 (Current):
#  - 🌟 Fix 1: 解除 LS 估測的雜訊底噪封印 (預設 ASIC_EST_SNR_DB = 100.0)
#  - 🌟 Fix 2: 實作數位端 IQ 預補償 (IQ Pre-compensation)，完美抵銷 Aux 硬體 IQ 瑕疵

import numpy as np
import config as C
from analog_sic_estimator import ls_estimation, generate_aux_signal


def _rand_rayleigh_taps(L: int) -> np.ndarray:
    h = (np.random.randn(L) + 1j * np.random.randn(L)).astype(np.complex128)
    h /= np.sqrt(np.sum(np.abs(h) ** 2) + 1e-18)
    return h


def _rand_rician_taps(L: int, K_dB: float) -> np.ndarray:
    K = 10 ** (K_dB / 10.0)
    los = np.ones(L, dtype=np.complex128)
    nlos = (np.random.randn(L) + 1j * np.random.randn(L)).astype(np.complex128)
    nlos /= np.sqrt(np.sum(np.abs(nlos) ** 2) + 1e-18)
    h = np.sqrt(K / (K + 1)) * los + np.sqrt(1 / (K + 1)) * nlos
    h /= np.sqrt(np.sum(np.abs(h) ** 2) + 1e-18)
    return h


def _circ_conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.complex128)
    for k in range(len(h)):
        y += h[k] * np.roll(x, k)
    return y


def _iq_imbalance_widely_linear(x: np.ndarray, iq_ambal: float, iq_phase: float) -> np.ndarray:
    alpha = (1 + iq_ambal) * np.exp(1j * iq_phase / 2)
    beta = (1 - iq_ambal) * np.exp(-1j * iq_phase / 2)
    return 0.5 * (alpha * x + beta * np.conj(x))


def _rapp_pa(x: np.ndarray, Asat: float, p: float) -> np.ndarray:
    r = np.abs(x)
    denom = (1 + (r / (Asat + 1e-18)) ** (2 * p)) ** (1 / (2 * p))
    return x / (denom + 1e-18)


def _apply_backoff(x: np.ndarray, bo_db: float) -> np.ndarray:
    return x * (10 ** (-bo_db / 20.0))


def _power(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x) ** 2) + 1e-18)


def simulate_full_receive_signal(
    x_remote: np.ndarray,
    x_self: np.ndarray,
    snr_db: float,
    rsi_scale: float,
    sic_db: float,
    main_channel_type: str = "flat",
    main_K_dB: float = 5.0,
    rsi_channel_type: str = "rayleigh",
    rsi_K_dB: float = 0.0,
    use_realistic_analog_sic: bool = True,
    enable_pa_nonlinearity: bool = True
):
    np.random.seed(int(getattr(C, "SEED", 2025))) # 強制每次產生相同的隨機通道
    RSI_NUM_TAPS = int(getattr(C, "RSI_NUM_TAPS", 3))
    MAIN_NUM_TAPS = int(getattr(C, "MAIN_NUM_TAPS", 1))

    IQ_AMBAL = float(getattr(C, "IQ_IMBALANCE", 0.02))
    IQ_PHASE = float(getattr(C, "IQ_PHASE", 0.02))
    BO_DB = float(getattr(C, "BO_DB", 0.0))

    PA_P = float(getattr(C, "PA_P", 2.2))
    ASAT_FACTOR = float(getattr(C, "ASAT_FACTOR", 2.0))

    AUX_PA_P = float(getattr(C, "AUX_PA_P", 2.4))
    AUX_ASAT_FACTOR = float(getattr(C, "AUX_ASAT_FACTOR", 2.0))

    ASIC_L = int(getattr(C, "ASIC_L", 4))
    ASIC_P = int(getattr(C, "ASIC_P", 7))
    ASIC_NSYM = int(getattr(C, "ASIC_NSYM", 500))
    ASIC_RIDGE = float(getattr(C, "ASIC_RIDGE", 1e-6))
    
    # 🌟 Fix 1: 預設使用 100.0 dB 極高估測 SNR，解除 LS 演算法的 25dB 雜訊天花板限制
    EST_SNR_DB = float(getattr(C, "ASIC_EST_SNR_DB", 30.0))

    FORCE_LINEAR_WHEN_PA_OFF = bool(getattr(C, "ASIC_FORCE_LINEAR_WHEN_PA_OFF", True))
    ASIC_P_use = 1 if (FORCE_LINEAR_WHEN_PA_OFF and (not enable_pa_nonlinearity)) else ASIC_P

    AUX_GAIN_ERR_STD = float(getattr(C, "AUX_GAIN_ERR_STD", 0.02))
    AUX_PHASE_ERR_STD = float(getattr(C, "AUX_PHASE_ERR_STD", 0.02))
    AUX_COUPLER_MEAN_DB = float(getattr(C, "AUX_COUPLER_MEAN_DB", -3.0))
    AUX_COUPLER_STD_DB = float(getattr(C, "AUX_COUPLER_STD_DB", 1.0))

    AUX_DISABLE_IQPA = bool(getattr(C, "AUX_DISABLE_IQPA", True))
    SAFETY_SIGN_FLIP = bool(getattr(C, "ASIC_SAFETY_SIGN_FLIP", True))

    # ---- Channels
    if main_channel_type == "flat":
        h_main = np.array([1 + 0j], dtype=np.complex128)
    elif main_channel_type == "rayleigh":
        h_main = _rand_rayleigh_taps(MAIN_NUM_TAPS)
    elif main_channel_type == "rician":
        h_main = _rand_rician_taps(MAIN_NUM_TAPS, main_K_dB)
    else:
        h_main = np.array([1 + 0j], dtype=np.complex128)

    if rsi_channel_type == "rayleigh":
        h_rsi = _rand_rayleigh_taps(RSI_NUM_TAPS)
    elif rsi_channel_type == "rician":
        h_rsi = _rand_rician_taps(RSI_NUM_TAPS, rsi_K_dB)
    else:
        h_rsi = _rand_rayleigh_taps(RSI_NUM_TAPS)

    # ---- Signals
    x_remote = x_remote.astype(np.complex128)
    x_self = x_self.astype(np.complex128)
    y_main = _circ_conv(x_remote, h_main)

    # ---- SI before analog (PA-first chain)
    x_self_bo = _apply_backoff(x_self, BO_DB)
    x_self_iq = _iq_imbalance_widely_linear(x_self_bo, IQ_AMBAL, IQ_PHASE)

    if enable_pa_nonlinearity:
        Asat_main = ASAT_FACTOR * np.sqrt(_power(x_self_iq))
        x_self_pa = _rapp_pa(x_self_iq, Asat_main, PA_P)
    else:
        x_self_pa = x_self_iq

    y_rsi_before_analog = _circ_conv(x_self_pa, h_rsi) * np.sqrt(rsi_scale)

    # ---- ADC noise var
    P_sig = _power(y_main)
    noise_var = P_sig / (10 ** (snr_db / 10.0) + 1e-18)

    waveforms = {
        "y_main": y_main.copy(),
        "y_rsi_before_analog": y_rsi_before_analog.copy(),
        "x_self_pa": x_self_pa.copy(),
    }

    analog_sic_info = {
        "mode": "realistic_asic_postpa" if use_realistic_analog_sic else "toy_kmatch",
        "ASIC_L": ASIC_L,
        "ASIC_P": ASIC_P,
        "ASIC_P_use": ASIC_P_use,
        "ASIC_NSYM": ASIC_NSYM,
        "BO_DB": BO_DB,
        "enable_pa_nonlinearity": bool(enable_pa_nonlinearity),
        "AUX_DISABLE_IQPA": bool(AUX_DISABLE_IQPA),
        "ASIC_FORCE_LINEAR_WHEN_PA_OFF": bool(FORCE_LINEAR_WHEN_PA_OFF),
    }

    if use_realistic_analog_sic:
        N = len(x_self)
        Nsym = min(ASIC_NSYM, N)

        # ---- Aux coupler + mismatch (fixed per call)
        coupler_db = np.random.randn() * AUX_COUPLER_STD_DB + AUX_COUPLER_MEAN_DB
        coupler_lin = 10 ** (coupler_db / 20.0)
        h_aux_flat = coupler_lin * np.exp(1j * 2 * np.pi * np.random.rand())
        g_err = 1.0 + np.random.randn() * AUX_GAIN_ERR_STD
        p_err = np.random.randn() * AUX_PHASE_ERR_STD
        h_aux_effective = h_aux_flat * g_err * np.exp(1j * p_err)

        analog_sic_info.update({
            "h_aux_flat_mag_db": float(20 * np.log10(np.abs(h_aux_flat) + 1e-18)),
            "g_err": float(g_err),
            "p_err_rad": float(p_err),
            "EST_SNR_DB": float(EST_SNR_DB),
        })

        # ---- Training target
        y_tx_train_rx = y_rsi_before_analog[:Nsym]

        P_est_sig = _power(y_tx_train_rx)
        est_noise_var = P_est_sig / (10 ** (EST_SNR_DB / 10.0) + 1e-18)
        w_train = (np.random.randn(Nsym) + 1j * np.random.randn(Nsym)) * np.sqrt(est_noise_var / 2)
        w_train = w_train.astype(np.complex128)
        y_target = y_tx_train_rx + w_train

        # ---- POST-PA reference for LS
        x_ref_train = x_self_pa[:Nsym]
        w_hat = ls_estimation(y_target, x_ref_train, L=ASIC_L, P=ASIC_P_use, ridge=ASIC_RIDGE)

        # ---- Generate aux waveform
        x_aux_digital = generate_aux_signal(x_self_pa, w_hat, L=ASIC_L, P=ASIC_P_use)

        # ---- Cancellation path
        if AUX_DISABLE_IQPA:
            # 理想狀態：不經過任何硬體瑕疵
            y_cancel_arrived = x_aux_digital * h_aux_effective
        else:
            # 🌟 Fix 2: 數位端 IQ 預補償 (Pre-compensation)
            # 1. 補償 Backoff 帶來的振幅縮放
            backoff_lin = 10 ** (-BO_DB / 20.0)
            z_desired = x_aux_digital / (backoff_lin + 1e-18)
            
            # 2. 計算硬體的 IQ 參數
            alpha_iq = (1 + IQ_AMBAL) * np.exp(1j * IQ_PHASE / 2)
            beta_iq = (1 - IQ_AMBAL) * np.exp(-1j * IQ_PHASE / 2)
            
            # 3. 執行反矩陣運算 (Inverse Widely Linear Transform)
            # 確保硬體輸出剛好還原為 z_desired
            denom_iq = np.abs(alpha_iq)**2 - np.abs(beta_iq)**2 + 1e-18
            x_aux_pre = (np.conj(alpha_iq) * z_desired - beta_iq * np.conj(z_desired)) / denom_iq
            
            # --- 以下進入模擬硬體發射 ---
            x_aux_bo = _apply_backoff(x_aux_pre, BO_DB)
            y_aux_iq = _iq_imbalance_widely_linear(x_aux_bo, IQ_AMBAL, IQ_PHASE)
            
            y_aux_pa = y_aux_iq  # 依照 V5.2 邏輯，Aux不經過 PA
            
            y_cancel_arrived = y_aux_pa * h_aux_effective

        # ---- alpha calibration
        y_cancel_train = y_cancel_arrived[:Nsym]
        denom = np.vdot(y_cancel_train, y_cancel_train) + 1e-18
        alpha = np.vdot(y_cancel_train, y_tx_train_rx) / denom
        y_cancel_arrived = alpha * y_cancel_arrived

        analog_sic_info["alpha_real"] = float(np.real(alpha))
        analog_sic_info["alpha_imag"] = float(np.imag(alpha))

        # ---- Safety flip
        if SAFETY_SIGN_FLIP:
            corr = np.vdot(y_tx_train_rx, y_cancel_arrived[:Nsym])
            if np.real(corr) < 0:
                y_cancel_arrived = -y_cancel_arrived
                analog_sic_info["safety_flip"] = True
            else:
                analog_sic_info["safety_flip"] = False

        # ---- RF combiner fixed subtraction
        y_rsi_after_analog = y_rsi_before_analog - y_cancel_arrived

        # ---- Metrics
        P_before = _power(y_rsi_before_analog)
        P_after = _power(y_rsi_after_analog)
        supp_db = 10 * np.log10(P_before / (P_after + 1e-18))

        analog_sic_info.update({
            "P_rsi_before": float(P_before),
            "P_rsi_after": float(P_after),
            "analog_supp_db": float(supp_db),
            "actual_suppression_db": float(supp_db),
            "est_noise_var": float(est_noise_var),
        })

        waveforms.update({
            "y_tx_train_rx": y_tx_train_rx.copy(),
            "x_aux_digital": x_aux_digital.copy(),
            "y_cancel_arrived": y_cancel_arrived.copy(),
            "w_hat": w_hat.copy(),
        })

    else:
        k_amp = 10 ** (-sic_db / 20.0)
        y_rsi_after_analog = y_rsi_before_analog * k_amp

        P_before = _power(y_rsi_before_analog)
        P_after = _power(y_rsi_after_analog)
        supp_db = 10 * np.log10(P_before / (P_after + 1e-18))

        analog_sic_info.update({
            "k_amp": float(k_amp),
            "P_rsi_before": float(P_before),
            "P_rsi_after": float(P_after),
            "analog_supp_db": float(supp_db),
            "actual_suppression_db": float(supp_db),
        })
        waveforms["y_cancel_arrived"] = (y_rsi_before_analog - y_rsi_after_analog).copy()

    waveforms["residual_after_analog"] = y_rsi_after_analog.copy()

    # ---- SINR stats
    P_main = _power(y_main)
    P_noise = float(noise_var)
    P_si_pre = _power(y_rsi_before_analog)
    P_si_post = _power(y_rsi_after_analog)
    sinr_pre = 10 * np.log10(P_main / (P_si_pre + P_noise + 1e-18))
    sinr_post = 10 * np.log10(P_main / (P_si_post + P_noise + 1e-18))

    analog_sic_info.update({
        "P_main": float(P_main),
        "noise_var": float(P_noise),
        "SINR_pre_db": float(sinr_pre),
        "SINR_analog_db": float(sinr_post),
    })

    channel_info = {
        "h_main": h_main,
        "h_rsi": h_rsi,
        "main_channel_type": main_channel_type,
        "rsi_channel_type": rsi_channel_type,
        "rsi_scale": float(rsi_scale),
    }

    return {
        "y_main": y_main,
        "y_rsi_before_analog": y_rsi_before_analog,
        "y_rsi_after_analog": y_rsi_after_analog,
        "noise_var": float(noise_var),
        "channel_info": channel_info,
        "analog_sic_info": analog_sic_info,
        "debug_waveforms": waveforms
    }