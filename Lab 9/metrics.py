"""
Lab 9: Objective metrics (PSNR, SSIM) + evaluation
Digital Image Processing — Compare enhanced vs reference (or input).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_sk
    from skimage.metrics import structural_similarity as ssim_sk
    USE_SKIMAGE = True
except ImportError:
    USE_SKIMAGE = False
try:
    from skimage.color import rgb2gray
    USE_RGB2GRAY = True
except ImportError:
    USE_RGB2GRAY = False


def load_grayscale(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = np.array(Image.open(path))
    if img.ndim == 3:
        gray = rgb2gray(img) if USE_RGB2GRAY else np.dot(img[..., :3].astype(np.float64), [0.299, 0.587, 0.114]) / 255.0
    else:
        gray = img.astype(np.float64) / np.iinfo(img.dtype).max
    return np.clip(gray, 0, 1)


def psnr(ref: np.ndarray, out: np.ndarray, data_range: float = 1.0) -> float:
    if USE_SKIMAGE:
        return float(psnr_sk(ref, out, data_range=data_range))
    mse = np.mean((ref.astype(np.float64) - out.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10 * np.log10((data_range ** 2) / mse))


def ssim(ref: np.ndarray, out: np.ndarray, data_range: float = 1.0) -> float:
    if USE_SKIMAGE:
        return float(ssim_sk(ref, out, data_range=data_range))
    return 0.0


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

    ref = load_grayscale(image_path)
    # Simulate "enhanced" as slight gamma for demo (no reference GT); here we use ref vs enhanced copy
    enhanced = np.power(np.clip(ref, 1e-6, 1.0), 0.9)
    # When no GT: report self-comparison and enhanced vs ref (ref as proxy for input)
    p = psnr(ref, enhanced)
    s = ssim(ref, enhanced)
    print(f"PSNR (ref vs enhanced): {p:.2f} dB")
    print(f"SSIM (ref vs enhanced): {s:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ref, cmap="gray")
    axes[0].set_title("Reference (input)")
    axes[0].axis("off")
    axes[1].imshow(enhanced, cmap="gray")
    axes[1].set_title("Enhanced (gamma=0.9)")
    axes[1].axis("off")
    plt.suptitle(f"PSNR={p:.2f} dB  SSIM={s:.4f}")
    plt.tight_layout()
    out_path = os.path.join(base, "Lab9_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
