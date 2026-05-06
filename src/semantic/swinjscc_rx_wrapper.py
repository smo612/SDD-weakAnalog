import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
SWIN_ROOT = REPO_ROOT / "external" / "SwinJSCC"
if str(SWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SWIN_ROOT))

from net.decoder import create_decoder  # noqa: E402


class SwinJSCCRXWrapper:
    """Minimal decoder wrapper matching the bounded SwinJSCC Fig.4 branch."""

    def __init__(self, ckpt_path: str, device: str = None):
        self.ckpt_path = str(ckpt_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "SwinJSCC_w/o_SAandRA"
        self.model_size = "base"
        self.given_snr = 10
        self.channel_dim = 192
        self.decoder = self._load_decoder()
        self.decoder.eval()

    def _decoder_kwargs(self):
        return dict(
            model=self.model_name,
            img_size=(128, 128),
            embed_dims=[320, 256, 192, 128],
            depths=[2, 6, 2, 2],
            num_heads=[10, 8, 6, 4],
            C=self.channel_dim,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )

    def _load_decoder(self):
        ckpt_path = Path(self.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing SwinJSCC checkpoint: {ckpt_path}")

        decoder = create_decoder(**self._decoder_kwargs()).to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        decoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("decoder."):
                new_key = key[len("decoder."):]
                if "attn_mask" in new_key:
                    continue
                decoder_state[new_key] = value
        if not decoder_state:
            raise ValueError("Checkpoint does not contain decoder.* keys")

        missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
        missing_real = [k for k in missing if "attn_mask" not in k]
        if missing_real or unexpected:
            raise RuntimeError(
                f"Failed to load SwinJSCC decoder cleanly: missing={missing_real[:8]}, unexpected={unexpected[:8]}"
            )

        decoder.update_resolution(8, 8)
        return decoder

    def decode(
        self,
        y_clean: np.ndarray,
        original_img: Optional[np.ndarray] = None,
        img_size: Tuple[int, int] = (128, 128),
        cbr: float = 1 / 16,
        meta_tx: Optional[dict] = None,
    ):
        latent_shape = None
        if meta_tx and "signal_info" in meta_tx:
            latent_shape = meta_tx["signal_info"].get("latent_shape")
        if latent_shape is None:
            latent_shape = [1, 64, self.channel_dim]

        tx_scale = 1.0
        if meta_tx and "signal_info" in meta_tx:
            tx_scale = float(meta_tx["signal_info"].get("tx_scale", 1.0) or 1.0)
        y_scaled = y_clean / tx_scale if tx_scale > 0 else y_clean

        y_hat = self._symbols_to_latent(y_scaled, latent_shape)
        with torch.no_grad():
            y_tensor = torch.from_numpy(y_hat).to(self.device)
            x_hat = self.decoder(y_tensor, self.given_snr, self.model_name)
        img_recon = self._postprocess(x_hat.cpu().numpy())

        metrics = {}
        if original_img is not None:
            metrics = self._compute_metrics(original_img, img_recon)
        return img_recon, metrics

    def _symbols_to_latent(self, y_data: np.ndarray, latent_shape):
        total_real = int(np.prod(latent_shape))
        expected_complex = total_real // 2
        y = y_data[:expected_complex]
        if len(y) != expected_complex:
            raise ValueError(f"Expected {expected_complex} complex symbols, got {len(y)}")
        i = np.real(y).astype(np.float32)
        q = np.imag(y).astype(np.float32)
        v = np.empty(i.size * 2, dtype=np.float32)
        v[0::2] = i
        v[1::2] = q
        return v.reshape(*latent_shape)

    def _postprocess(self, x_hat: np.ndarray) -> np.ndarray:
        img = x_hat[0].transpose(1, 2, 0)
        return np.clip(img, 0.0, 1.0)

    def _compute_metrics(self, img_original: np.ndarray, img_recon: np.ndarray) -> dict:
        from skimage.metrics import structural_similarity as ssim

        img_original = np.clip(img_original, 0.0, 1.0)
        img_recon = np.clip(img_recon, 0.0, 1.0)
        mse = np.mean((img_original - img_recon) ** 2)
        psnr = 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)
        ssim_val = ssim(img_original, img_recon, data_range=1.0, channel_axis=2)
        return {"psnr": float(psnr), "mse": float(mse), "ms_ssim": float(ssim_val)}
