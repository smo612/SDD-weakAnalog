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


MODEL_SPECS = {
    "wo_saandra_psnr_c192_snr10": {
        "model_name": "SwinJSCC_w/o_SAandRA",
        "model_size": "base",
        "given_snr": 10,
        "given_rate": 192,
        "channel_dim": 192,
    },
    "sa_base_psnr_c96_snr13": {
        "model_name": "SwinJSCC_w/_SA",
        "model_size": "base",
        "given_snr": 13,
        "given_rate": 96,
        "channel_dim": 96,
    },
    "ra_base_psnr_cbr192_snr10": {
        "model_name": "SwinJSCC_w/_RA",
        "model_size": "base",
        "given_snr": 10,
        "given_rate": 192,
        "channel_dim": None,
    },
    "saandra_base_psnr_cbr192_snr13": {
        "model_name": "SwinJSCC_w/_SAandRA",
        "model_size": "base",
        "given_snr": 13,
        "given_rate": 192,
        "channel_dim": None,
    },
    "saandra_large_psnr_cbr192_snr13": {
        "model_name": "SwinJSCC_w/_SAandRA",
        "model_size": "large",
        "given_snr": 13,
        "given_rate": 192,
        "channel_dim": None,
    },
}


class SwinJSCCOfficialWrapper:
    """
    Minimal official-packing SwinJSCC encoder wrapper for bounded Fig.4 probes.

    Important detail:
    SwinJSCC's native channel code pairs the flattened latent as
    flat[:half] + 1j * flat[half:], not even/odd interleaving.
    This wrapper follows that official packing.
    """

    def __init__(self, ckpt_path: str, variant: str, device: str = None):
        if variant not in MODEL_SPECS:
            raise ValueError(f"Unsupported SwinJSCC variant: {variant}")
        self.ckpt_path = str(ckpt_path)
        self.variant = variant
        self.spec = MODEL_SPECS[variant]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = self._load_encoder()
        self.encoder.eval()

    def _depths(self):
        if self.spec["model_size"] == "large":
            return [2, 2, 18, 2]
        return [2, 2, 6, 2]

    def _encoder_kwargs(self):
        return dict(
            model=self.spec["model_name"],
            img_size=(128, 128),
            patch_size=2,
            in_chans=3,
            embed_dims=[128, 192, 256, 320],
            depths=self._depths(),
            num_heads=[4, 6, 8, 10],
            C=self.spec["channel_dim"],
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
            if self.spec["model_name"] in {"SwinJSCC_w/_RA", "SwinJSCC_w/_SAandRA"}:
                feature, mask = self.encoder(
                    x,
                    self.spec["given_snr"],
                    self.spec["given_rate"],
                    self.spec["model_name"],
                )
                latent = feature * mask
                active_fraction = float(mask.detach().float().mean().item())
                mask_shape = list(mask.shape)
            else:
                latent = self.encoder(
                    x,
                    self.spec["given_snr"],
                    self.spec["given_rate"],
                    self.spec["model_name"],
                )
                active_fraction = 1.0
                mask_shape = None

        context = {
            "semantic_family": "swinjscc",
            "semantic_variant": self.variant,
            "semantic_mode": "real",
            "model_name": self.spec["model_name"],
            "model_size": self.spec["model_size"],
            "given_snr": self.spec["given_snr"],
            "given_rate": self.spec["given_rate"],
            "channel_dim": self.spec["channel_dim"],
            "latent_shape": list(latent.shape),
            "mask_shape": mask_shape,
            "packing_mode": "half_split",
            "img_shape": list(img.shape),
            "cbr_nominal": float(cbr),
            "active_fraction": active_fraction,
        }
        return latent, context

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        h, w = img.shape[:2]
        assert (h, w) == (128, 128), f"Expected 128x128, got {(h, w)}"
        assert img.shape[2] == 3
        assert img.dtype == np.float32
        x = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return x.to(self.device)


def latent_to_baseband_half_split(latent, target_power=None):
    if hasattr(latent, "cpu"):
        latent_np = latent.detach().cpu().numpy()
    else:
        latent_np = latent
    flat = latent_np.flatten().astype(np.float32)
    half = flat.size // 2
    if half * 2 != flat.size:
        raise ValueError(f"Expected even flattened latent length, got {flat.size}")
    symbols = (flat[:half] + 1j * flat[half:]).astype(np.complex64)
    original_power = float(np.mean(np.abs(symbols) ** 2))
    if target_power is not None and original_power > 1e-10:
        tx_scale = np.sqrt(target_power / original_power)
        symbols = symbols * tx_scale
    else:
        tx_scale = 1.0
    return symbols, float(tx_scale), float(original_power)
