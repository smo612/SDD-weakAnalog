import numpy as np

def ls_estimation(y_rx, data_tx_cp, data_aux_cp, L=4, P=7):
    """
    對應 C++ 檔案 estimation.hpp 中的 LS_estimation 函數
    
    Args:
        y_rx: 估計階段接收到的訊號 (1D complex array)
        data_tx_cp: 主發射訊號 (1D complex array)
        data_aux_cp: 輔助發射訊號 (1D complex array)
        L: FIR 濾波器長度 (對應 Lw)
        P: 多項式最高階數 (1, 3, 5, 7)
        
    Returns:
        W: 算出的 FIR 濾波器權重，形狀為 (Nw, L)
    """
    NN = len(data_tx_cp)
    Nw = ((P + 1) * (P + 3)) // 4  # 基底函數的數量
    
    # 建立 X 矩陣，對應 C++ 的 x_line，形狀為 (NN, L * Nw + 2)
    X = np.zeros((NN, L * Nw + 2), dtype=np.complex128)
    
    # 建立多項式特徵 (對應 4 層 for 迴圈)
    for l in range(1, P + 1, 2):
        for m in range(0, l + 1):
            col_base = (((l * l) + 3) // 4 + (m - 1)) * L
            
            # 向量化計算: mypow(x, m) * mypow(conj(x), l-m)
            tx_m = data_tx_cp ** m
            tx_conj_lm = np.conj(data_tx_cp) ** (l - m)
            feature_seq = tx_m * tx_conj_lm
            
            # 對應延遲 (j)
            for j in range(L):
                col_idx = col_base + j
                if j == 0:
                    X[:, col_idx] = feature_seq
                else:
                    X[j:, col_idx] = feature_seq[:-j]
                    # i < j 的部分預設為 0，NumPy 初始化已處理

    # 加上 Aux 訊號本身與其共軛
    X[:, L * Nw] = data_aux_cp
    X[:, L * Nw + 1] = np.conj(data_aux_cp)
    
    # 使用 NumPy 解最小平方法 (對應 LAPACKE_zgelss)
    w, _, _, _ = np.linalg.lstsq(X, y_rx, rcond=1e-8)
    
    # 提取 c 矩陣與 Caux1, Caux2
    c = np.zeros((Nw, L), dtype=np.complex128)
    for i in range(Nw):
        c[i, :] = w[i * L : (i + 1) * L]
        
    Caux1 = w[L * Nw]
    Caux2 = w[L * Nw + 1]
    
    # 計算最終的 FIR 濾波器權重 W
    W = np.zeros((Nw, L), dtype=np.complex128)
    denomi = np.abs(Caux2)**2 - np.abs(Caux1)**2
    
    for l in range(1, P + 1, 2):
        for m in range(0, l + 1):
            j = ((l * l) + 3) // 4 + m - 1
            k = ((l * l) + 3) // 4 + l - m - 1
            W[j, :] = (np.conj(Caux1) * c[j, :] - Caux2 * np.conj(c[k, :])) / denomi
            
    return W

def generate_aux_signal(data_tx, W, L=4, P=7):
    """
    對應 ope2.hpp 中 Cancellation phase 產生 wave_aux 的邏輯
    利用算出來的 FIR 權重 W，對發射訊號進行預先補償
    """
    NN = len(data_tx)
    wave_aux = np.zeros(NN, dtype=np.complex128)
    
    for l in range(1, P + 1, 2):
        for m in range(0, l + 1):
            k = ((l * l) + 3) // 4 + m - 1
            
            # 產生特徵
            tx_m = data_tx ** m
            tx_conj_lm = np.conj(data_tx) ** (l - m)
            feature_seq = tx_m * tx_conj_lm
            
            # 套用 FIR 濾波器 (卷積)
            for j in range(L):
                if j == 0:
                    wave_aux += W[k, j] * feature_seq
                else:
                    wave_aux[j:] += W[k, j] * feature_seq[:-j]
                    
    return wave_aux

# 簡單的單元測試
if __name__ == "__main__":
    np.random.seed(42)
    N = 1024
    print("產生測試訊號...")
    tx = np.random.randn(N) + 1j * np.random.randn(N)
    aux = np.random.randn(N) + 1j * np.random.randn(N)
    y = np.random.randn(N) + 1j * np.random.randn(N)  # 假的接收訊號
    
    print("測試 LS Estimation...")
    W = ls_estimation(y, tx, aux, L=4, P=7)
    print(f"FIR 濾波器權重形狀: {W.shape}")
    
    print("測試 Aux 訊號生成...")
    aux_cancellation = generate_aux_signal(tx, W, L=4, P=7)
    print(f"生成的輔助消除訊號形狀: {aux_cancellation.shape}")
    print("✅ 測試通過！模組運作正常。")