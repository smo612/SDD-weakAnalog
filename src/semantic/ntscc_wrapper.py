"""
NTSCC encoder wrapper used by the Fig.4 semantic TX path.

This version keeps backward-compatible 128x128 behavior by default, but it can
also adapt to other resolutions that are divisible by 16. That allows bounded
HR protocol checks without changing the checkpoint family itself.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from layer.analysis_transform import AnalysisTransform


class NTSCCWrapper:
    def __init__(
        self,
        ckpt_path: str,
        device: str = None,
        img_size: Tuple[int, int] = (128, 128),
    ):
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_img_size = tuple(int(x) for x in img_size)
        self.model = self._load_encoder(ckpt_path)
        self.model.eval()

    def _load_encoder(self, ckpt_path: str):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ga_kwargs = {
            "img_size": self.current_img_size,
            "embed_dims": [256, 256, 256, 256],
            "depths": [1, 1, 2, 4],
            "num_heads": [8, 8, 8, 8],
            "window_size": 8,
            "mlp_ratio": 4.0,
            "qkv_bias": True,
            "qk_scale": None,
            "norm_layer": nn.LayerNorm,
            "patch_norm": True,
        }

        model = AnalysisTransform(**ga_kwargs).to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        encoder_state_dict = self._extract_encoder_weights(checkpoint)
        missing, unexpected = model.load_state_dict(encoder_state_dict, strict=False)
        if missing or unexpected:
            missing_real = [key for key in missing if "attn_mask" not in key]
            if missing_real or unexpected:
                raise RuntimeError(
                    f"Failed to load encoder cleanly: missing={missing_real[:8]}, unexpected={unexpected[:8]}"
                )
        model.update_resolution(*self.current_img_size)
        return model

    def _extract_encoder_weights(self, checkpoint: Dict) -> Dict:
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and any("ga." in key for key in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError("Unsupported checkpoint format for NTSCC encoder")

        encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith("ga."):
                new_key = key[3:]
                if "attn_mask" in new_key:
                    continue
                encoder_dict[new_key] = value
            elif key.startswith("module.ga."):
                new_key = key[10:]
                if "attn_mask" in new_key:
                    continue
                encoder_dict[new_key] = value

        if not encoder_dict:
            raise ValueError("Checkpoint does not contain encoder weights with ga.* prefix")
        return encoder_dict

    def encode(self, img: np.ndarray, cbr: float = 1 / 16):
        self._ensure_resolution(img.shape[:2])
        x = self._preprocess(img)
        with torch.no_grad():
            latent = self.model(x)

        context = {
            "ntscc_mode": "real",
            "cbr": cbr,
            "latent_shape": list(latent.shape),
            "img_shape": list(img.shape),
        }
        return latent, context

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        height, width = img.shape[:2]
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"Image size must be divisible by 16, got {(height, width)}")
        if img.shape[2] != 3:
            raise ValueError("Expected RGB image with 3 channels")
        if img.dtype != np.float32:
            raise TypeError(f"Expected float32 image, got {img.dtype}")
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return x.to(self.device)

    def _ensure_resolution(self, img_hw: Tuple[int, int]) -> None:
        img_size = tuple(int(x) for x in img_hw)
        if img_size != self.current_img_size:
            self.model.update_resolution(*img_size)
            self.current_img_size = img_size


if __name__ == "__main__":
    ckpt_path = "checkpoints/PSNR_SNR=10_gaussian/ntscc_hyperprior_quality_4_psnr.pth"
    if not Path(ckpt_path).exists():
        print(f"Missing checkpoint: {ckpt_path}")
    else:
        wrapper = NTSCCWrapper(ckpt_path=ckpt_path)
        img = np.random.rand(128, 128, 3).astype(np.float32)
        latent, ctx = wrapper.encode(img, cbr=1 / 16)
        print(f"Latent shape: {latent.shape}")
        print(f"Context: {ctx}")
