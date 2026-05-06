"""
NTSCC decoder wrapper used by the Fig.4 RX path.

The original local branch assumed 128x128 inputs and an [1, 256, 8, 8] latent.
This cleaned version preserves that default while allowing dynamic resolutions
whose height and width are divisible by 16.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from layer.synthesis_transform import SynthesisTransform


class NTSCCRXWrapper:
    def __init__(
        self,
        ckpt_path: str,
        device: str = None,
        img_size: Tuple[int, int] = (128, 128),
    ):
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_img_size = tuple(int(x) for x in img_size)
        self.model = self._load_decoder(ckpt_path)
        self.model.eval()

    def _load_decoder(self, ckpt_path: str):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        gs_kwargs = {
            "img_size": self.current_img_size,
            "embed_dims": [256, 256, 256, 256],
            "depths": [4, 2, 1, 1],
            "num_heads": [8, 8, 8, 8],
            "window_size": 8,
            "mlp_ratio": 4.0,
            "norm_layer": nn.LayerNorm,
            "patch_norm": True,
        }

        model = SynthesisTransform(**gs_kwargs).to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint

        gs_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("gs.") and "attn_mask" not in key:
                gs_state_dict[key[3:]] = value

        missing_keys, unexpected_keys = model.load_state_dict(gs_state_dict, strict=False)
        missing_real = [key for key in missing_keys if "attn_mask" not in key]
        if missing_real or unexpected_keys:
            raise RuntimeError(
                f"Failed to load decoder cleanly: missing={missing_real[:8]}, unexpected={unexpected_keys[:8]}"
            )

        latent_h = self.current_img_size[0] // 16
        latent_w = self.current_img_size[1] // 16
        model.update_resolution(latent_h, latent_w)
        return model

    def decode(
        self,
        y_clean: np.ndarray,
        original_img: np.ndarray = None,
        img_size: Tuple[int, int] = (128, 128),
        cbr: float = 1 / 16,
        meta_tx: Optional[dict] = None,
    ):
        self._ensure_resolution(img_size)

        if meta_tx and "pilot_info" in meta_tx:
            pilot_info = meta_tx["pilot_info"]
            pilot_enabled = pilot_info.get("pilot_enabled", False)
            pilot_period = int(pilot_info.get("pilot_period", 64) or 64)
            n_pilots = int(pilot_info.get("n_pilots", 0) or 0)
        else:
            pilot_enabled = len(y_clean) > self._expected_complex_symbols(img_size)
            pilot_period = 64
            n_pilots = len(y_clean) - self._expected_complex_symbols(img_size) if pilot_enabled else 0

        n_data_expected = None
        if meta_tx and "signal_info" in meta_tx:
            n_data_expected = int(meta_tx["signal_info"].get("n_data_symbols", 0) or 0)
        if not n_data_expected:
            n_data_expected = self._expected_complex_symbols(img_size)

        if pilot_enabled and n_pilots > 0:
            y_data = self._strip_pilots(y_clean, pilot_period, n_pilots, n_data_expected)
            if len(y_data) != n_data_expected:
                raise ValueError(f"Pilot stripping mismatch: expected {n_data_expected}, got {len(y_data)}")
        else:
            y_data = y_clean[:n_data_expected] if len(y_clean) > n_data_expected else y_clean

        tx_scale = meta_tx.get("signal_info", {}).get("tx_scale", 0.0) if meta_tx else 0.0
        if tx_scale and tx_scale > 0:
            y_data = y_data / tx_scale
        else:
            rms = np.sqrt(np.mean(np.abs(y_data) ** 2))
            if rms > 1e-8:
                y_data = y_data / rms

        y_hat = self._symbols_to_latent(y_data, img_size, cbr)
        with torch.no_grad():
            y_hat_tensor = torch.from_numpy(y_hat).to(self.device)
            x_hat_tensor = self.model(y_hat_tensor, out_conv=True)

        x_hat = x_hat_tensor.cpu().numpy()
        img_recon = self._postprocess(x_hat)

        metrics = {}
        if original_img is not None:
            metrics = self._compute_metrics(original_img, img_recon)
        return img_recon, metrics

    def _ensure_resolution(self, img_size: Tuple[int, int]) -> None:
        img_size = tuple(int(x) for x in img_size)
        if img_size[0] % 16 != 0 or img_size[1] % 16 != 0:
            raise ValueError(f"img_size must be divisible by 16, got {img_size}")
        if img_size != self.current_img_size:
            latent_h = img_size[0] // 16
            latent_w = img_size[1] // 16
            self.model.update_resolution(latent_h, latent_w)
            self.current_img_size = img_size

    def _expected_complex_symbols(self, img_size: Tuple[int, int]) -> int:
        height, width = img_size
        latent_h = height // 16
        latent_w = width // 16
        latent_c = 256
        return (latent_h * latent_w * latent_c) // 2

    def _strip_pilots(self, y: np.ndarray, period: int, n_pilots: int, n_data_expected: int) -> np.ndarray:
        if period <= 0 or n_pilots <= 0:
            return y[:n_data_expected].astype(np.complex64, copy=False)

        out = []
        idx = 0
        pilots_removed = 0
        total = len(y)
        data_len = 0
        while idx < total and data_len < n_data_expected:
            chunk_end = min(idx + period, total)
            if chunk_end > idx:
                chunk = y[idx:chunk_end]
                out.append(chunk)
                data_len += len(chunk)
            idx = chunk_end
            if pilots_removed < n_pilots and idx < total:
                idx += 1
                pilots_removed += 1

        y_data = np.concatenate(out) if out else np.empty(0, dtype=y.dtype)
        if len(y_data) > n_data_expected:
            y_data = y_data[:n_data_expected]
        return y_data.astype(np.complex64, copy=False)

    def _symbols_to_latent(self, y_data: np.ndarray, img_size: Tuple[int, int], cbr: float) -> np.ndarray:
        height, width = img_size
        latent_h = height // 16
        latent_w = width // 16
        latent_c = 256
        expected_complex = (latent_h * latent_w * latent_c) // 2
        if len(y_data) != expected_complex:
            raise ValueError(f"Expected {expected_complex} complex symbols, got {len(y_data)}")

        i_part = np.real(y_data).astype(np.float32)
        q_part = np.imag(y_data).astype(np.float32)
        flat = np.empty(i_part.size * 2, dtype=np.float32)
        flat[0::2] = i_part
        flat[1::2] = q_part
        return flat.reshape(1, latent_c, latent_h, latent_w)

    def _postprocess(self, x_hat: np.ndarray) -> np.ndarray:
        img_recon = x_hat[0].transpose(1, 2, 0)
        return np.clip(img_recon, 0.0, 1.0)

    def _compute_metrics(self, img_original: np.ndarray, img_recon: np.ndarray) -> dict:
        img_original = np.clip(img_original, 0.0, 1.0)
        img_recon = np.clip(img_recon, 0.0, 1.0)
        mse = np.mean((img_original - img_recon) ** 2)
        psnr = 100.0 if mse < 1e-10 else 10 * np.log10(1.0 / mse)
        metrics = {"psnr": float(psnr), "mse": float(mse)}
        try:
            from skimage.metrics import structural_similarity as ssim

            try:
                ms_ssim = ssim(img_original, img_recon, data_range=1.0, channel_axis=2)
            except TypeError:
                ms_ssim = ssim(img_original, img_recon, data_range=1.0, multichannel=True)
            metrics["ms_ssim"] = float(ms_ssim)
        except ImportError:
            pass
        return metrics
