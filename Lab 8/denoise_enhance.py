"""
Lab 8: Denoising + enhancement pipeline (bilateral + CLAHE)
Digital Image Processing — Combined pipeline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
try:
    from skimage.exposure import equalize_adapthist
    from skimage.color import rgb2gray
    USE_SKIMAGE = True
except ImportError:
    USE_SKIMAGE = False


def load_grayscale(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = np.array(Image.open(path))
    if img.ndim == 3:
        gray = rgb2gray(img) if USE_SKIMAGE else np.dot(img[..., :3].astype(np.float64), [0.299, 0.587, 0.114]) / 255.0
    else:
        gray = img.astype(np.float64) / np.iinfo(img.dtype).max
    return np.clip(gray, 0, 1)


def bilateral_denoise(img: np.ndarray, d: int = 5, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Bilateral filter (edge-preserving denoising). Input/output float [0,1]."""
    if not HAS_CV2:
        return img
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = cv2.bilateralFilter(u8, d, sigma_color, sigma_space)
    return out.astype(np.float64) / 255.0


def clahe(img: np.ndarray, clip_limit: float = 0.03, kernel_size: int = 64) -> np.ndarray:
    if USE_SKIMAGE:
        return equalize_adapthist(img, clip_limit=clip_limit, kernel_size=kernel_size)
    if HAS_CV2:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernel_size, kernel_size))
        out = clahe_obj.apply(u8)
        return out.astype(np.float64) / 255.0
    return img


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

    gray = load_grayscale(image_path)
    denoised = bilateral_denoise(gray)
    enhanced = clahe(denoised)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(denoised, cmap="gray")
    axes[1].set_title("Bilateral denoised")
    axes[1].axis("off")
    axes[2].imshow(enhanced, cmap="gray")
    axes[2].set_title("Denoised + CLAHE")
    axes[2].axis("off")
    plt.tight_layout()
    out_path = os.path.join(base, "Lab8_denoise_enhance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
