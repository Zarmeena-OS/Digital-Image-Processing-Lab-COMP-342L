"""
Lab 5: Single-scale Retinex (SSR)
Digital Image Processing — Illumination–reflectance decomposition.
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


def single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    """SSR: R = log(I) - log(G*I). G = Gaussian. Input/output float [0,1]."""
    blurred = gaussian_filter(img, sigma=sigma)
    r = np.log10(img + 1e-6) - np.log10(blurred + 1e-6)
    r = (r - r.min()) / (r.max() - r.min() + 1e-8)
    return np.clip(r, 0, 1)


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
    ssr_small = single_scale_retinex(gray, sigma=15)
    ssr_large = single_scale_retinex(gray, sigma=80)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(ssr_small, cmap="gray")
    axes[1].set_title("SSR (sigma=15)")
    axes[1].axis("off")
    axes[2].imshow(ssr_large, cmap="gray")
    axes[2].set_title("SSR (sigma=80)")
    axes[2].axis("off")
    plt.tight_layout()
    out_path = os.path.join(base, "Lab5_ssr.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
