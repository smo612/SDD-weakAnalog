import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from modules_CDiff import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class SDDDataset(Dataset):
    def __init__(self, pt_file):
        logging.info(f"載入 SDD 資料集: {pt_file}")
        # 載入資料 (忽略 FutureWarning)
        self.data = torch.load(pt_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['target'], item['cond'], item['noisy']

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def add_noise_ddpm(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        noise = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=250, size=(n,))    

def train(args):
    device = args.device
    
    # 🌟 讀取我們剛才產生的乾淨訓練集
    dataset = SDDDataset("sdd_diffusion_dataset_train.pt")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    diffusion = Diffusion(device=device)
    model = UNet(c_in=6, c_out=2, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    all_epoch_losses = []

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        batch_losses = []
        
        for i, (target, cond, noisy) in enumerate(pbar):
            target = target.to(device)
            cond = cond.to(device)
            noisy = noisy.to(device)

            t = diffusion.sample_timesteps(target.shape[0]).to(device)
            x_t, noise = diffusion.add_noise_ddpm(target, t)

            condition = torch.cat([noisy, cond], dim=1)
            predicted_noise = model(x_t, t, x_hat=condition)

            loss = mse(noise, predicted_noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            batch_losses.append(loss_val)
            pbar.set_postfix(MSE=loss_val)
            logger.add_scalar("MSE", loss_val, global_step=epoch * l + i)

        all_epoch_losses.append(batch_losses)

        # 🌟 每 10 個 Epoch 存一次檔，並且確保最後一個 Epoch 一定會存檔
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            os.makedirs(os.path.join("ddpm_models", args.run_name), exist_ok=True)
            torch.save(model.state_dict(), os.path.join("ddpm_models", args.run_name, f"ckpt_epoch_{epoch}.pt"))
            logging.info(f"模型權重已儲存: ckpt_epoch_{epoch}.pt")

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # 🌟 最終長訓參數設定
    args.run_name = "SDD_Diffusion_VER4_RSI315W_DIV2K_FINAL" 
    args.epochs = 200       # 預計跑 5.5 到 6 小時
    args.batch_size = 4     # 保守的 Batch Size，避免半夜 OOM 爆顯存
    args.device = "cuda"
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()