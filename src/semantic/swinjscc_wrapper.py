import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
SWIN_ROOT = REPO_ROOT / "external" / "SwinJSCC"
if str(SWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(SWIN_ROOT))

from net.encoder import create_encoder  # noqa: E402


class SwinJSCCWrapper:
    """
    Minimal encoder wrapper for the bounded Fig.4 semantic branch.

    Current supported candidate:
    - SwinJSCC_w/o_SAandRA
    - AWGN HRimage
    - fixed snr10
    - PSNR-trained
    - C192
    """

    def __init__(self, ckpt_path: str, device: str = None):
        self.ckpt_path = str(ckpt_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "SwinJSCC_w/o_SAandRA"
        self.model_size = "base"
        self.given_snr = 10
        self.channel_dim = 192
        self.encoder = self._load_encoder()
        self.encoder.eval()

    def _encoder_kwargs(self):
        return dict(
            model=self.model_name,
            img_size=(128, 128),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320],
            depths=[2, 2, 6, 2],
            num_heads=[4, 6, 8, 10],
            C=self.channel_dim,
            window_size=8,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
        )

    def _load_encoder(self):
        ckpt_path = Path(self.ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing SwinJSCC checkpoint: {ckpt_path}")

        encoder = create_encoder(**self._encoder_kwargs()).to(self.device)
        state_dict = torch.load(ckpt_path, map_location=self.device)
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                new_key = key[len("encoder."):]
                if "attn_mask" in new_key:
                    continue
                encoder_state[new_key] = value
        if not encoder_state:
            raise ValueError("Checkpoint does not contain encoder.* keys")

        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        missing_real = [k for k in missing if "attn_mask" not in k]
        if missing_real or unexpected:
            raise RuntimeError(
                f"Failed to load SwinJSCC encoder cleanly: missing={missing_real[:8]}, unexpected={unexpected[:8]}"
            )

        encoder.update_resolution(128, 128)
        return encoder

    def encode(self, img: np.ndarray, cbr: float = 1 / 16) -> Tuple[torch.Tensor, Dict]:
        x = self._preprocess(img)
        with torch.no_grad():
            latent = self.encoder(x, self.given_snr, self.channel_dim, self.model_name)

        context = {
            "semantic_family": "swinjscc",
            "semantic_variant": "wo_saandra_psnr_c192_snr10",
            "semantic_mode": "real",
            "model_name": self.model_name,
            "model_size": self.model_size,
            "given_snr": self.given_snr,
            "channel_dim": self.channel_dim,
            "latent_shape": list(latent.shape),
            "img_shape": list(img.shape),
            "cbr_nominal": float(cbr),
        }
        return latent, context

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        assert (h, w) == (128, 128), f"Expected 128x128, got {(h, w)}"
        assert img.shape[2] == 3
        assert img.dtype == np.float32
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return x.to(self.device)
