import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Dynamic 1D self-attention for time-series / RF signals.
    """
    def __init__(self, channels, size=None, num_heads=4):
        super().__init__()
        self.channels = channels
        self.size = size  # kept only for backward compatibility
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )
        self.ln1 = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # 🌟 降維：x 的 shape 現在是 [B, C, L] (Batch, Channels, Length)
        b, c, l = x.shape
        if c != self.channels:
            raise ValueError(
                f"SelfAttention expected channels={self.channels}, but got {c}."
            )

        x_seq = x.transpose(1, 2)   # 轉換為 [B, L, C] 餵給 Attention
        x_ln = self.ln1(x_seq)
        attn_out, _ = self.mha(x_ln, x_ln, x_ln)
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.ff_self(x_seq)

        out = x_seq.transpose(1, 2) # 轉回 [B, C, L]
        return out


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if mid_channels is None:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # 🌟 Conv2d -> Conv1d
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False), # 🌟 Conv2d -> Conv1d
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), # 🌟 MaxPool2d -> MaxPool1d
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        # 🌟 降維：Time embedding 只需要擴張一個維度 [:, :, None]，並 repeat 到 L 長度
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        # 🌟 降維：2D bilinear -> 1D linear
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)

        # 🌟 降維：時間長度對齊 (1D)
        if x.shape[-1] != skip_x.shape[-1]:
            x = F.interpolate(
                x,
                size=skip_x.shape[-1],
                mode="linear",
                align_corners=True
            )

        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)

        # 🌟 降維：Time embedding 只需要擴張一個維度
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Encoder(nn.Module):
    def __init__(self, c_in, time_dim, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)   
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

    def forward(self, x, t):
        x1 = self.inc(x)          # [B, 64, L]
        x2 = self.down1(x1, t)    # [B, 128, L/2]
        x2 = self.sa1(x2)

        x3 = self.down2(x2, t)    # [B, 256, L/4]
        x3 = self.sa2(x3)

        x4 = self.down3(x3, t)    # [B, 256, L/8]
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        return x4, x3, x2, x1


class Channel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + torch.randn_like(x)


class Decoder(nn.Module):
    def __init__(self, c_out, time_dim, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        # self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1) # 🌟 Conv2d -> Conv1d

    def forward(self, x4, x3, x2, x1, t):
        x = self.up1(x4, x3, t)
        x = self.sa4(x)

        x = self.up2(x, x2, t)
        x = self.sa5(x)

        x = self.up3(x, x1, t)
        # x = self.sa6(x)

        output = self.outc(x)
        return output


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class UNet(nn.Module):
    def __init__(self, c_in=6, c_out=2, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.encoder = Encoder(c_in, time_dim=time_dim, device=device)
        self.Channel = Channel()
        self.decoder = Decoder(c_out, time_dim, device)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, x_hat=None, y=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None and hasattr(self, "label_emb"):
            t = t + self.label_emb(y)

        if x_hat is not None:
            x = torch.cat([x, x_hat], dim=1)

        x4, x3, x2, x1 = self.encoder(x, t)
        x4 = self.Channel(x4)
        output = self.decoder(x4, x3, x2, x1, t)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.encoder = Encoder(c_in, time_dim, device)
        self.Channel = Channel()
        self.decoder = Decoder(c_out, time_dim, device)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None and hasattr(self, "label_emb"):
            t = t + self.label_emb(y)

        x4, x3, x2, x1 = self.encoder(x, t)
        x4 = self.Channel(x4)
        output = self.decoder(x4, x3, x2, x1, t)
        return output

if __name__ == '__main__':
    # 🌟 快速測試腳本：你可以直接執行這份檔案，看看維度對不對
    net = UNet(c_in=6, c_out=2, device="cpu")
    print("AI-SIC 1D 模型總參數量:", sum([p.numel() for p in net.parameters()]))
    # 模擬輸入：Batch=2, Channels=2(noisy)+4(condition), Length=8704
    x = torch.randn(2, 2, 8704)
    x_hat = torch.randn(2, 4, 8704) 
    t = x.new_tensor([200] * x.shape[0]).long()
    output = net(x, t, x_hat=x_hat)
    print("輸出維度 (應為 Batch, 2, 8704):", output.shape)