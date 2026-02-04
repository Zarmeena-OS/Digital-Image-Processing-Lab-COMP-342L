"""
Lab 7: Color preservation — CLAHE in HSV (V channel only)
Digital Image Processing — Avoid color shift when enhancing.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from skimage.exposure import equalize_adapthist
    from skimage.color import rgb2hsv, hsv2rgb
    USE_SKIMAGE = True
except ImportError:
    USE_SKIMAGE = False
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_rgb(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return np.clip(img.astype(np.float64) / 255.0, 0, 1)


def clahe_grayscale(img: np.ndarray, clip_limit: float = 0.03, kernel_size: int = 64) -> np.ndarray:
    if USE_SKIMAGE:
        return equalize_adapthist(img, clip_limit=clip_limit, kernel_size=kernel_size)
    if not HAS_CV2:
        raise RuntimeError("Need skimage or opencv.")
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernel_size, kernel_size))
    out = clahe_obj.apply(u8)
    return out.astype(np.float64) / 255.0


def clahe_hsv_preserve_color(rgb: np.ndarray, clip_limit: float = 0.03, kernel_size: int = 64) -> np.ndarray:
    """Apply CLAHE only to V channel in HSV; H and S unchanged to preserve color."""
    if USE_SKIMAGE:
        hsv = rgb2hsv(rgb)
        hsv[..., 2] = clahe_grayscale(hsv[..., 2], clip_limit, kernel_size)
        return np.clip(hsv2rgb(hsv), 0, 1)
    if HAS_CV2:
        rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float64) / np.array([180, 255, 255])
        hsv[..., 2] = clahe_grayscale(hsv[..., 2], clip_limit, kernel_size)
        hsv_u8 = (np.clip(hsv, 0, 1) * np.array([180, 255, 255])).astype(np.uint8)
        rgb_out = cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2RGB)
        return rgb_out.astype(np.float64) / 255.0
    return rgb


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "Content", "download.jpeg"),
        os.path.join(base, "Content", "download.jpg"),
        os.path.join(base, "..", "Lab 3", "Content", "download.jpeg"),
        os.path.join(base, "..", "Lab 2", "Ali Hamza's Lab", "Content", "download.jpeg"),
    ]
    image_path = next((p for p in candidates if os.path.exists(p)), None)
    if not image_path:
        print("No image found.")
        return

    rgb = load_rgb(image_path)
    enhanced = clahe_hsv_preserve_color(rgb)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")
    axes[1].imshow(enhanced)
    axes[1].set_title("CLAHE on V (HSV) — color preserved")
    axes[1].axis("off")
    plt.tight_layout()
    out_path = os.path.join(base, "Lab7_color_preserved.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
