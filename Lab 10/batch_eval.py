"""
Lab 10: Batch evaluation on dataset
Digital Image Processing — Run enhancement on multiple images, save results.
"""

import os
import numpy as np
from PIL import Image

try:
    from skimage.exposure import equalize_adapthist
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
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(kernel_size, kernel_size))
        out = clahe_obj.apply(u8)
        return out.astype(np.float64) / 255.0
    return img


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    content_dir = os.path.join(base, "Content")
    fallback_dirs = [
        os.path.join(base, "..", "Lab 3", "Content"),
        os.path.join(base, "..", "Lab 2", "Ali Hamza's Lab", "Content"),
    ]
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    images = []
    for d in [content_dir] + fallback_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(exts):
                    images.append(os.path.join(d, f))
    if not images:
        print("No images in Content/ or fallback dirs.")
        return

    out_dir = os.path.join(base, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        gray = load_grayscale(path)
        enhanced = clahe(gray)
        out_path = os.path.join(out_dir, f"{name}_clahe.png")
        out_u8 = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(out_u8).save(out_path)
        print(f"Saved: {out_path}")
    print(f"Batch done. {len(images)} images -> {out_dir}")


if __name__ == "__main__":
    main()
