# analog_sic_estimator.py
# WL + Nonlinear + FIR (Memory Polynomial style) LS Estimator
# - ls_estimation(): solve Phi(x) w ≈ y_target
# - generate_aux_signal(): generate waveform Phi(x) w
#
# IMPORTANT:
#   y_target 應該是「SI 在 RX 看到的樣子」(SI-only training segment)
#   不是 y_tx + y_aux 的混合，避免 w 的物理意義跑偏。

import numpy as np


def _build_wl_nl_fir_matrix(x: np.ndarray, L: int, P: int) -> np.ndarray:
    """
    Build design matrix Phi for widely-linear nonlinear FIR basis.
    Basis orders: odd {1,3,5,...,P} (if P is even, we stop at P-1)
    For each delay l in [0..L-1], include:
        u_p_l = x[n-l] * |x[n-l]|^{p-1}
        v_p_l = conj(x[n-l]) * |x[n-l]|^{p-1}
    Columns: [u_{p=1,l=0..L-1}, v_{p=1,l=0..L-1}, u_{p=3,...}, v_{p=3,...}, ...]
    """
    x = x.astype(np.complex128)
    N = len(x)
    P = int(P)
    L = int(L)
    if L <= 0:
        raise ValueError("L must be positive.")
    if P <= 0:
        raise ValueError("P must be positive.")
    # use odd orders
    orders = list(range(1, P + 1, 2))

    # Prepare delayed versions
    Xd = np.zeros((N, L), dtype=np.complex128)
    for l in range(L):
        Xd[:, l] = np.roll(x, l)  # circular delay (consistent with your channel sim)

    # Build columns
    cols = []
    for p in orders:
        amp = np.abs(Xd) ** (p - 1)  # (N,L), real
        U = Xd * amp                 # (N,L)
        V = np.conj(Xd) * amp        # (N,L)
        cols.append(U)
        cols.append(V)

    Phi = np.concatenate(cols, axis=1)  # (N, 2*L*len(orders))
    return Phi


def ls_estimation(y_target: np.ndarray,
                  x_ref: np.ndarray,
                  L: int = 4,
                  P: int = 7,
                  ridge: float = 1e-6) -> np.ndarray:
    """
    Solve for w in Phi(x_ref) w ≈ y_target

    ridge: Tikhonov regularization weight (improves stability under noise/collinearity)

    Returns:
        w: complex weights, shape (M,)
    """
    y = y_target.astype(np.complex128).reshape(-1)
    x = x_ref.astype(np.complex128).reshape(-1)
    if len(y) != len(x):
        raise ValueError("y_target and x_ref must have the same length.")

    Phi = _build_wl_nl_fir_matrix(x, L=L, P=P)  # (N,M)

    # Ridge LS: (Phi^H Phi + λ I) w = Phi^H y
    # Use normal eq with small ridge for speed; OK for moderate M.
    M = Phi.shape[1]
    A = Phi.conj().T @ Phi
    b = Phi.conj().T @ y
    A = A + (ridge * np.eye(M, dtype=np.complex128))

    w = np.linalg.solve(A, b)
    return w


def generate_aux_signal(x_ref: np.ndarray,
                        w: np.ndarray,
                        L: int = 4,
                        P: int = 7) -> np.ndarray:
    """
    Generate waveform x_aux = Phi(x_ref) w

    NOTE:
      This is the *digital* waveform you will feed to the aux chain (IQ/PA etc).
    """
    x = x_ref.astype(np.complex128).reshape(-1)
    w = w.astype(np.complex128).reshape(-1)
    Phi = _build_wl_nl_fir_matrix(x, L=L, P=P)
    y = Phi @ w
    return y.astype(np.complex128)


# -----------------------------
# Quick unit test
# -----------------------------
if __name__ == "__main__":
    np.random.seed(0)

    print("產生測試訊號...")
    N = 1024
    L = 4
    P = 7

    # Random QPSK-ish
    bits_i = np.random.randint(0, 2, size=N) * 2 - 1
    bits_q = np.random.randint(0, 2, size=N) * 2 - 1
    x = (bits_i + 1j * bits_q).astype(np.complex128) / np.sqrt(2)

    # Synthetic target: some WL+NL FIR mapping
    true_w = (np.random.randn(2 * L * len(range(1, P + 1, 2))) +
              1j * np.random.randn(2 * L * len(range(1, P + 1, 2)))) * 0.05
    true_w = true_w.astype(np.complex128)
    y = generate_aux_signal(x, true_w, L=L, P=P)

    print("測試 LS Estimation...")
    w_hat = ls_estimation(y, x, L=L, P=P, ridge=1e-8)
    print("FIR 權重形狀:", w_hat.shape)

    print("測試 Aux 訊號生成...")
    y_hat = generate_aux_signal(x, w_hat, L=L, P=P)
    print("生成的輔助消除訊號形狀:", y_hat.shape)

    err = np.mean(np.abs(y - y_hat) ** 2) / (np.mean(np.abs(y) ** 2) + 1e-18)
    assert err < 1e-6, f"Unit test failed, NMSE={err}"
    print("✅ 測試通過！模組運作正常。")