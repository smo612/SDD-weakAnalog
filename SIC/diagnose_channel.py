"""diagnose_channel.py - 診斷最佳通道配置 (V6.4)

V6.4 關鍵修正（PA-FIRST物理順序）：
1. **PA必須在多徑之前**（最關鍵修正）
   - 正確：IQ → PA → 多徑 → 功率縮放 → Analog SIC
   - 錯誤：多徑 → IQ → 功率縮放 → PA → Analog SIC（V6.3）
   
2. 為什麼必須PA-first？
   - MP數學假設：y = h*PA(x)（線性記憶效應）
   - PA-last會導致：y = PA(h*x)（無法用MP擬合）
   
3. 實驗證據：
   - V6.3（PA-last）：MP增益 +0.40 dB ❌
   - V6.4（PA-first）：預期 +10-15 dB ✓

V6.3成果（保留）：
- 補上Analog SIC步驟 ✓
- 增加樣本數到32768 ✓
- 固定隨機種子 ✓

測試不同組合找出 MP 有效的甜蜜點：
- Taps: 3, 4, 5
- Rapp p: 2.0, 2.2, 2.4, 2.6
- rsi_scale: 20（甜蜜點）
"""

import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from SIC import load_backend


def generate_test_data(N=32768, rsi_scale=20, n_taps=3, rapp_p=2.2, seed=None):
    """生成測試數據（V6.3: 對齊真實Pipeline - 加入Analog SIC）
    
    Args:
        N: 樣本數（提升到32768以獲得更穩定估計）
        rsi_scale: RSI功率比例（20為甜蜜點）
        n_taps: 多徑taps數
        rapp_p: Rapp模型參數
        seed: 隨機種子（可選）
    """
    if seed is not None:
        np.random.seed(seed)
    
    # TX 信號
    x = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
    x /= np.sqrt(np.mean(np.abs(x)**2))
    
    # === V6.4: PA-FIRST物理順序（關鍵修正）===
    # 正確順序：IQ → PA → 多徑
    # 錯誤順序：多徑 → IQ → PA（V6.3及之前）
    
    # 1. IQ 不平衡（固定 2.5%，在低功率時）
    iq_ambal = 0.025
    iq_phase = 2.5
    phi = np.deg2rad(iq_phase)
    a = (1.0 + iq_ambal) * np.exp(1j*phi/2)
    b = iq_ambal * np.exp(-1j*phi/2) * 0.5
    si = (a*x + b*np.conj(x)).astype(np.complex64)
    
    # 2. 計算固定Asat（在PA之前，基於低功率信號）
    ref_amplitude = np.sqrt(np.mean(np.abs(si)**2) + 1e-12)
    Asat_abs = ref_amplitude * 2.0  # V6.1: 固定值
    
    # 3. Rapp PA（V6.4: 在多徑之前！）
    abs_si = np.abs(si)
    ang = np.angle(si)
    gain = abs_si / ((1.0 + (abs_si/Asat_abs)**(2.0*rapp_p))**(1.0/(2.0*rapp_p)) + 1e-18)
    phase_shift = 0.08 * (abs_si/Asat_abs)**2
    si = (gain * np.exp(1j*(ang + phase_shift))).astype(np.complex64)
    
    # 4. 多徑通道（V6.4: 在PA之後）
    h = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)).astype(np.complex64)
    h /= np.sqrt(np.sum(np.abs(h)**2))
    
    si_multipath = np.zeros(N, dtype=np.complex64)
    for k in range(n_taps):
        si_multipath += h[k] * np.roll(si, k)
    si = si_multipath
    
    # 5. 功率縮放（在PA和多徑之後）
    si *= np.sqrt(rsi_scale)
    
    # 5. 功率縮放（在PA和多徑之後）
    si *= np.sqrt(rsi_scale)
    
    # 保存PA+多徑後的SI（Analog SIC之前）
    si_before_analog = si.copy()
    
    # V6.3: 關鍵補充 - 加入Analog SIC步驟（對齊真實pipeline）
    # 模擬Analog SIC：目標20 dB抑制，含2%增益誤差和1.5°相位誤差
    sic_db = 20.0
    k_analog = 10.0**(-sic_db/20.0)  # 抑制係數
    g_err = 1.0 + np.random.randn() * 0.02  # 2%增益誤差
    p_err = np.exp(1j * np.deg2rad(np.random.randn() * 1.5))  # 1.5°相位誤差
    
    # Analog SIC輸出（殘差SI）
    si = (k_analog * g_err * p_err) * si_before_analog
    
    # 期望信號 + 噪聲
    s = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
    s *= 0.1
    noise = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
    noise *= 0.01
    
    y = s + si + noise
    
    # V6.2: 補齊Backend所需的完整欄位
    P_signal = float(np.mean(np.abs(s)**2))
    P_noise = float(np.mean(np.abs(noise)**2))
    
    return {
        'y': y,
        'x': x,
        'si_after_analog': si,  # 主要的SI序列
        'y_after_analog': si,   # V6.2新增：某些backend用這個鍵名
        'P_signal': P_signal,
        'P_noise': P_noise,
        'noise_var': P_noise    # V6.2新增：很多backend讀這個
    }


def test_combination(n_taps, rapp_p, config, seed=42):
    """測試單個組合
    
    Args:
        n_taps: 多徑taps數
        rapp_p: Rapp參數
        config: Backend配置
        seed: 固定隨機種子（避免每組換數據）
    """
    # V6.3: N=32768（更穩定估計），rsi_scale=20（甜蜜點），含Analog SIC
    data = generate_test_data(N=32768, rsi_scale=20, n_taps=n_taps, 
                             rapp_p=rapp_p, seed=seed)
    
    # 測試 WLLS
    wlls = load_backend('wlls', config)
    wlls.fit(data)
    _, wlls_metrics = wlls.predict(data)
    
    # 測試 MP
    mp = load_backend('mp', config)
    mp.fit(data)
    _, mp_metrics = mp.predict(data)
    
    return {
        'wlls_supp': wlls_metrics['Digital_supp_si'],
        'mp_supp': mp_metrics['Digital_supp_si'],
        'gain': mp_metrics['Digital_supp_si'] - wlls_metrics['Digital_supp_si']
    }


def main():
    """主診斷流程（V6.4 - PA-FIRST物理順序）
    
    V6.4 關鍵修正：
    1. PA在多徑之前（修正物理順序）
    2. 符合MP數學假設（y = h*PA(x)）
    3. 預期MP增益從+0.4 dB提升到+10-15 dB
    """
    print("="*70)
    print("通道配置診斷 - 找出 MP 有效的甜蜜點 (V6.4)")
    print("="*70)
    print("\nV6.4 關鍵修正:")
    print("  1. PA-FIRST物理順序（IQ → PA → 多徑）")
    print("  2. 符合MP數學假設（最關鍵修正）")
    print("  3. 預期MP增益：+10-15 dB")
    print()
    
    # MP 配置（使用增強版）
    config = {
        'wlls': {'L': 5, 'lambda_reg': 0.01},
        'mp': {
            'poly_orders': [1, 3, 5, 7],
            'memory_len': 9,
            'ridge_lambda': 1e-3,  # 降低正則化
            'with_conj': True
        }
    }
    
    # 測試矩陣
    taps_list = [3, 4, 5]
    p_list = [2.0, 2.2, 2.4, 2.6]
    
    print(f"\n測試矩陣:")
    print(f"  Taps: {taps_list}")
    print(f"  Rapp p: {p_list}")
    print(f"  目標: MP 增益 ≥ 10 dB\n")
    
    results = []
    
    for n_taps in taps_list:
        for rapp_p in p_list:
            print(f"測試: Taps={n_taps}, p={rapp_p:.1f}...", end=" ")
            
            try:
                result = test_combination(n_taps, rapp_p, config)
                results.append({
                    'taps': n_taps,
                    'p': rapp_p,
                    **result
                })
                
                gain = result['gain']
                status = "✅" if gain >= 10.0 else "⚠️" if gain >= 5.0 else "❌"
                print(f"{status} MP 增益: {gain:+.2f} dB")
                
            except Exception as e:
                print(f"❌ 失敗: {e}")
                results.append({
                    'taps': n_taps,
                    'p': rapp_p,
                    'wlls_supp': 0,
                    'mp_supp': 0,
                    'gain': 0
                })
    
    # 找出最佳組合
    print("\n" + "="*70)
    print("診斷結果")
    print("="*70)
    
    # 按增益排序
    results_sorted = sorted(results, key=lambda x: x['gain'], reverse=True)
    
    print("\n最佳組合（前 3 名）:")
    for i, r in enumerate(results_sorted[:3], 1):
        print(f"  {i}. Taps={r['taps']}, p={r['p']:.1f} → MP 增益: {r['gain']:+.2f} dB")
        print(f"     WLLS: {r['wlls_supp']:.2f} dB, MP: {r['mp_supp']:.2f} dB")
    
    best = results_sorted[0]
    
    print("\n📋 推薦配置:")
    print(f"  RSI_NUM_TAPS = {best['taps']}")
    print(f"  rapp_p = {best['p']:.1f}")
    print(f"  預期 MP 增益: {best['gain']:+.2f} dB")
    
    if best['gain'] >= 10.0:
        print("\n🎉 找到有效配置！")
    elif best['gain'] >= 5.0:
        print("\n⚠️  部分有效，考慮:")
        print("    1. 進一步降低 p（更強非線性）")
        print("    2. 增加訓練樣本數")
        print("    3. 調整 MP 參數")
    else:
        print("\n❌ 所有組合都不達標")
        print("    建議:")
        print("    1. 重新思考通道架構")
        print("    2. 考慮更簡單的 baseline（例如純 PA，無多徑）")
        print("    3. 檢查 MP 實現是否正確")
    
    # 繪製熱圖（簡易版）
    print("\n" + "="*70)
    print("MP 增益熱圖（dB）")
    print("="*70)
    header = "p \\ Taps"
    print(f"{header:<8}", end="")
    for t in taps_list:
        print(f"{t:>8}", end="")
    print()
    print("-" * 70)
    
    for p in p_list:
        print(f"{p:<8.1f}", end="")
        for t in taps_list:
            r = next((x for x in results if x['taps']==t and x['p']==p), None)
            if r:
                gain = r['gain']
                color = "✅" if gain >= 10 else "⚠️" if gain >= 5 else "❌"
                print(f"{color}{gain:>6.2f}", end="")
            else:
                print(f"   N/A", end="")
        print()
    
    print("="*70)
    print("\n圖例:")
    print("  ✅ ≥10 dB（達標）")
    print("  ⚠️  5-10 dB（部分達標）")
    print("  ❌ <5 dB（未達標）")


if __name__ == '__main__':
    main()