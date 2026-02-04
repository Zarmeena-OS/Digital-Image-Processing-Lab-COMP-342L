# Lab 3 implementation plan

## Goal
Contrast-enhancement baseline for the research project (Labs 3–12): **global histogram equalization** vs **CLAHE** on grayscale images.

## Scope (this lab only)
1. Load an image (grayscale or RGB → grayscale).
2. Apply global HE and CLAHE.
3. Visualize: original | global HE | CLAHE (side-by-side).
4. Save figures for report/paper.

## Structure (all under `Lab 3/`)
- `Content/` — Test images (e.g. `download.jpeg`). Script falls back to Lab 2 image if none here.
- `clahe_vs_he.py` — Main script: load → grayscale → global HE → CLAHE → plot → save.
- `README.md` — Problem statement, roadmap, run instructions.

## Implementation steps
1. Create `Content/` directory (user adds images).
2. Implement `clahe_vs_he.py`:
   - Load image; convert to grayscale (float [0,1]).
   - Global HE: `skimage.exposure.equalize_hist` or manual.
   - CLAHE: `skimage.exposure.equalize_adapthist` or `cv2.createCLAHE`.
   - Matplotlib: 1×3 figure, save PNG.
   - Image path: try `Content/download.jpeg`, `Content/download.jpg`, then Lab 2 image.
3. Update README so all paths refer to `Lab 3/` (no subfolder).

## Run
```bash
cd "Lab 3"
python clahe_vs_he.py
```

## Output
- Figure displayed and saved as `Lab3_comparison.png` in `Lab 3/`.
