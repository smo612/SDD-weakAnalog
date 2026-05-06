import os
import json
import torch
import numpy as np
from tqdm import tqdm
from modules_CDiff import UNet
from CDiff import Diffusion
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def process_complex_to_1d(signal, target_len=8704):
    """將通訊複數訊號轉換為 1D Tensor (1, 2, Length)"""
    pad_size = target_len - len(signal)
    signal_padded = np.pad(signal, (0, pad_size), mode='constant')
    real_part = np.real(signal_padded)
    imag_part = np.imag(signal_padded)
    tensor_1d = torch.tensor(np.stack([real_part, imag_part]), dtype=torch.float32)
    return tensor_1d.unsqueeze(0) # 增加 Batch 維度

def process_1d_to_complex(tensor_1d, original_len=8319):
    """將 1D Tensor 還原為通訊複數訊號"""
    tensor_1d = tensor_1d.squeeze(0).cpu().numpy()
    complex_signal = tensor_1d[0] + 1j * tensor_1d[1]
    return complex_signal[:original_len]

def get_adaptive_t_start(sinr):
    """
    🌟 終極包絡線版自適應 SDEdit (Adaptive Threshold Logic)
    根據前端類比消除後的 SINR 品質，果斷切換修復力道，達成全區間最佳解。

    分段設計（對應 sinr_analog）：
      sinr >= 20 dB  →  T=2   直通（RSI 極小，訊號乾淨，不需修復）
      20 > sinr >= 14.5  →  T=30  輕修（過渡帶，類比端稍有殘餘，輕度補正）
      14.5 > sinr >= 10  →  T=150 中修（RSI 100萬左右，MP 開始崩潰）
      sinr < 10       →  T=200 重修（RSI 3.15M+，極端重症全力重構）
    """
    variant = os.environ.get("ADAPTIVE_VARIANT", "v1").lower()
    if variant == "v2":
        # v2 refines the empirical v1 ladder around the medium-residual band.
        if sinr >= 20.0:
            return 2
        elif sinr >= 17.0:
            return 15
        elif sinr >= 14.0:
            return 80
        elif sinr >= 10.0:
            return 150
        else:
            return 200

    if sinr >= 20.0:
        # 【極輕度區間】RSI 極小，訊號健康，直通保留細節
        return 2
    elif sinr >= 14.5:
        # 【輕度過渡帶】類比端有少量殘餘，輕修補正，確保不輸 Analog+MP
        return 30
    elif sinr >= 10.0:
        # 【黃金交叉區間】RSI 100萬，MP 開始崩潰，啟動中重度 AI 修復
        return 150
    else:
        # 【極端重症區間】RSI 3.15M ~ 10M，全功率深度重構
        return 200

def main():
    device = "cuda"
    
    # 1. 讀取前端與條件訊號
    cond_np = np.load('bridge_tx/x_tx.npy')
    noisy_np = np.load('bridge/y_adc.npy')

    # ==========================================
    # 🌟 讀取前端體檢報告，動態決定修復力道
    # ==========================================
    auto_t_start = 200
    try:
        with open('bridge/meta.json', 'r') as f:
            meta = json.load(f)
            sinr_analog = meta.get("SINR_analog", 0)
            auto_t_start = get_adaptive_t_start(sinr_analog)
            print(f"📊 [體檢報告] 前端類比 SINR = {sinr_analog:.2f} dB")
    except Exception as e:
        print("⚠️ 無法讀取 meta.json，使用預設重症模式 T_start=200")

    # 允許環境變數強制覆寫，否則使用自適應計算出的最佳步數
    T_start = int(os.environ.get("T_START", auto_t_start))
    print(f"🚀 啟動 1D AI-SIC SDEdit (自適應模式: 執行 {T_start} 步修復) ...")

    # ==========================================
    # 神經網路推論區
    # ==========================================
    # 特徵正規化 (非常重要，確保模型吃到的 Scale 穩定)
    scale = np.std(noisy_np) + 1e-8
    
    # 轉換為 1D 並在 Channel 維度拼接 Condition
    cond = process_complex_to_1d(cond_np / scale).to(device)
    noisy = process_complex_to_1d(noisy_np / scale).to(device)
    condition = torch.cat([noisy, cond], dim=1) 

    # 載入 1D 特化版權重
    # model_path = "ddpm_models/SDD_Diffusion_VER4_RSI315W_1D/ckpt_epoch_190.pt"
    model_path = "ddpm_models/SDD_Diffusion_VER4_RSI315W_DIV2K_FINAL_OLD/ckpt_epoch_199.pt"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型權重: {model_path}")
        exit(1)
        
    model = UNet(c_in=6, c_out=2, device=device).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    diffusion = Diffusion(device=device)
    
    # 根據動態的 T_start 決定初始加噪程度，產生 x_{T_start}
    t_start_tensor = (torch.ones(1) * T_start).long().to(device)
    alpha_hat_t = diffusion.alpha_hat[t_start_tensor][:, None, None]
    noise_init = torch.randn_like(noisy)
    x = torch.sqrt(alpha_hat_t) * noisy + torch.sqrt(1 - alpha_hat_t) * noise_init

    # 執行 Denoising Loop (局部修復)
    with torch.no_grad():
        for i in tqdm(reversed(range(1, T_start)), total=T_start-1, desc=f"Denoising (T={T_start})", leave=False):
            t = (torch.ones(1) * i).long().to(device)
            predicted_noise = model(x, t, x_hat=condition)
            
            alpha = diffusion.alpha[t][:, None, None]
            alpha_hat = diffusion.alpha_hat[t][:, None, None]
            beta = diffusion.beta[t][:, None, None]
            
            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

    # 還原回通訊複數訊號並乘回 scale
    denoised_complex = process_1d_to_complex(x) * scale
    
    # 輸出給下一站 (Demodulation)
    os.makedirs('bridge_digital', exist_ok=True)
    output_path = "bridge_digital/y_clean.npy"
    np.save(output_path, denoised_complex)
    
if __name__ == "__main__":
    main()
