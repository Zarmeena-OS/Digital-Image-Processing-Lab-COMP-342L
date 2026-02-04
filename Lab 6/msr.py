"""
Lab 6: Multi-scale Retinex (MSR)
Digital Image Processing — MSR pipeline (weighted sum of SSR at multiple scales).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
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
    return np.clip(gray, 1e-6, 1.0)


def ssr(img: np.ndarray, sigma: float) -> np.ndarray:
    r = np.log10(img + 1e-6) - np.log10(gaussian_filter(img, sigma=sigma) + 1e-6)
    return r


def msr(img: np.ndarray, sigmas: list, weights: list = None) -> np.ndarray:
    """MSR = weighted sum of SSR at sigmas. Weights default to 1/len(sigmas)."""
    if weights is None:
        weights = [1.0 / len(sigmas)] * len(sigmas)
    out = np.zeros_like(img, dtype=np.float64)
    for s, w in zip(sigmas, weights):
        out += w * ssr(img, s)
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return np.clip(out, 0, 1)


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
    msr_img = msr(gray, sigmas=[15, 80, 250])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(msr_img, cmap="gray")
    axes[1].set_title("MSR (sigma=15,80,250)")
    axes[1].axis("off")
    plt.tight_layout()
    out_path = os.path.join(base, "Lab6_msr.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
