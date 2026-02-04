"""
Lab 3: Contrast baseline — Global HE vs CLAHE
Digital Image Processing — First step of the enhancement pipeline (Labs 3–12).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from skimage.exposure import equalize_hist, equalize_adapthist
    from skimage.color import rgb2gray
    USE_SKIMAGE = True
except ImportError:
    USE_SKIMAGE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_grayscale(path: str) -> np.ndarray:
    """Load image and return grayscale float [0, 1]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = np.array(Image.open(path))
    if img.ndim == 3:
        if USE_SKIMAGE:
            gray = rgb2gray(img)
        else:
            gray = np.dot(img[..., :3].astype(np.float64), [0.299, 0.587, 0.114]) / 255.0
    else:
        gray = img.astype(np.float64) / np.iinfo(img.dtype).max
    return np.clip(gray, 0.0, 1.0)


def global_he(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization. Input/output float [0,1]."""
    if USE_SKIMAGE:
        return equalize_hist(img)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hist, _ = np.histogram(u8.flatten(), bins=256, range=(0, 256))
    cdf = hist.cumsum()
    cdf = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-8)
    out = cdf[u8]
    return out.astype(np.float64) / 255.0


def clahe(img: np.ndarray, clip_limit: float = 0.03, kernel_size: int = 64) -> np.ndarray:
    """CLAHE. Input/output float [0,1]. kernel_size ~ tile size."""
    if USE_SKIMAGE:
        return equalize_adapthist(img, clip_limit=clip_limit, kernel_size=kernel_size)
    if not HAS_CV2:
        raise RuntimeError("Need skimage or opencv for CLAHE.")
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    cv2_clip = max(1.0, min(4.0, clip_limit * 40))
    clahe_obj = cv2.createCLAHE(clipLimit=cv2_clip, tileGridSize=(kernel_size, kernel_size))
    out = clahe_obj.apply(u8)
    return out.astype(np.float64) / 255.0


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "Content", "download.jpeg"),
        os.path.join(base, "Content", "download.jpg"),
        os.path.join(base, "Content", "image.png"),
        os.path.join(base, "..", "Lab 2", "Ali Hamza's Lab", "Content", "download.jpeg"),
    ]
    image_path = None
    for p in candidates:
        if os.path.exists(p):
            image_path = p
            break
    if image_path is None:
        print("No image found. Place an image in Lab 3/Content/ (e.g. download.jpeg).")
        return

    gray = load_grayscale(image_path)
    he_img = global_he(gray)
    clahe_img = clahe(gray, clip_limit=0.03, kernel_size=64)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original (grayscale)")
    axes[0].axis("off")

    axes[1].imshow(he_img, cmap="gray")
    axes[1].set_title("Global Histogram Equalization")
    axes[1].axis("off")

    axes[2].imshow(clahe_img, cmap="gray")
    axes[2].set_title("CLAHE")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(base, "Lab3_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
