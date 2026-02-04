"""
Lab 4: Gamma / log / power-law transforms
Digital Image Processing — Compare with Lab 3 contrast baseline.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from skimage.color import rgb2gray
    USE_SKIMAGE = True
except ImportError:
    USE_SKIMAGE = False


def load_grayscale(path: str) -> np.ndarray:
    """Load image, return grayscale float [0, 1]."""
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


def gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law: out = img^gamma. Input/output float [0,1]."""
    return np.power(np.clip(img, 1e-6, 1.0), gamma)


def log_transform(img: np.ndarray, c: float = 1.0) -> np.ndarray:
    """Log: out = c * log(1 + img). Input/output float [0,1]."""
    return (c * np.log1p(np.clip(img, 0, 1) * 255) / np.log1p(255)).astype(np.float64)


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
        print("No image found. Place an image in Lab 4/Content/ or Lab 3/Content/.")
        return

    gray = load_grayscale(image_path)
    gamma_05 = gamma_transform(gray, 0.5)
    gamma_15 = gamma_transform(gray, 1.5)
    log_img = log_transform(gray, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(gamma_05, cmap="gray")
    axes[0, 1].set_title("Gamma = 0.5 (brighten)")
    axes[0, 1].axis("off")
    axes[1, 0].imshow(gamma_15, cmap="gray")
    axes[1, 0].set_title("Gamma = 1.5 (darken)")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(log_img, cmap="gray")
    axes[1, 1].set_title("Log transform (c=1)")
    axes[1, 1].axis("off")
    plt.tight_layout()
    out_path = os.path.join(base, "Lab4_gamma_log.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
