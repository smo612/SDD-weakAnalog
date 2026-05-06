from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image


def resolve_image_path(img_name: str, kodak_dir: str = "data/kodak") -> Path:
    path = Path(img_name)
    if path.exists():
        return path
    if not img_name.endswith(".png"):
        img_name = f"{img_name}.png"
    path = Path(kodak_dir) / img_name
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return path


def load_kodak_image(
    img_name: str,
    kodak_dir: str = "data/kodak",
    protocol: str = "resize_128",
    target_size: Tuple[int, int] = (128, 128),
) -> Tuple[np.ndarray, str, Dict]:
    path = resolve_image_path(img_name, kodak_dir=kodak_dir)
    img = Image.open(path).convert("RGB")
    orig_width, orig_height = img.size

    if protocol == "resize_128":
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        crop_box = None
    elif protocol == "witt_hr_center_crop":
        crop_width = orig_width - (orig_width % 128)
        crop_height = orig_height - (orig_height % 128)
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError(
                f"Image {path} is too small for WITT HR protocol: {(orig_width, orig_height)}"
            )
        left = (orig_width - crop_width) // 2
        top = (orig_height - crop_height) // 2
        crop_box = (left, top, left + crop_width, top + crop_height)
        img = img.crop(crop_box)
    else:
        raise ValueError(f"Unsupported Kodak protocol: {protocol}")

    arr = np.asarray(img).astype(np.float32) / 255.0
    meta = {
        "protocol": protocol,
        "source_size_wh": [orig_width, orig_height],
        "final_size_hw": [int(arr.shape[0]), int(arr.shape[1])],
        "crop_box_ltrb": list(crop_box) if crop_box is not None else None,
    }
    return arr, str(path), meta
