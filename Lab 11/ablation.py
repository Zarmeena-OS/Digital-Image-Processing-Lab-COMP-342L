"""
Lab 11: Ablation / parameter study
Digital Image Processing — Vary CLAHE clip_limit and kernel_size, report metrics.
"""

import os
import numpy as np
from PIL import Image

try:
    from skimage.exposure import equalize_adapthist
    from skimage.color import rgb2gray
    from skimage.metrics import peak_signal_noise_ratio as psnr_sk
    from skimage.metrics import structural_similarity as ssim_sk
    USE_SKIMAGE = True
except ImportError:
    USE_SKIMAGE = False
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_grayscale(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = np.array(Image.open(path))
    if img.ndim == 3:
        gray = rgb2gray(img) if USE_SKIMAGE else np.dot(img[..., :3].astype(np.float64), [0.299, 0.587, 0.114]) / 255.0
    else:
        gray = img.astype(np.float64) / np.iinfo(img.dtype).max
    return np.clip(gray, 0, 1)


def clahe(img: np.ndarray, clip_limit: float = 0.03, kernel_size: int = 64) -> np.ndarray:
    if USE_SKIMAGE:
        return equalize_adapthist(img, clip_limit=clip_limit, kernel_size=kernel_size)
    if HAS_CV2:
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cl = max(1.0, min(4.0, clip_limit * 40))
        clahe_obj = cv2.createCLAHE(clipLimit=cl, tileGridSize=(kernel_size, kernel_size))
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

    ref = load_grayscale(image_path)
    clip_limits = [0.01, 0.03, 0.06]
    kernel_sizes = [32, 64, 128]
    print("Ablation: CLAHE clip_limit x kernel_size -> PSNR(ref,out), SSIM(ref,out)")
    print("-" * 60)
    for cl in clip_limits:
        for ks in kernel_sizes:
            out = clahe(ref, clip_limit=cl, kernel_size=ks)
            if USE_SKIMAGE:
                p = psnr_sk(ref, out, data_range=1.0)
                s = ssim_sk(ref, out, data_range=1.0)
            else:
                mse = np.mean((ref - out) ** 2)
                p = 10 * np.log10(1.0 / (mse + 1e-10))
                s = 0.0
            print(f"clip={cl:.2f} kernel={ks:3d}  PSNR={p:.2f} dB  SSIM={s:.4f}")
    print("-" * 60)
    out_path = os.path.join(base, "Lab11_ablation.txt")
    with open(out_path, "w") as f:
        f.write("Ablation: CLAHE parameters\n")
        f.write("clip_limit x kernel_size -> PSNR, SSIM (ref vs enhanced)\n")
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
