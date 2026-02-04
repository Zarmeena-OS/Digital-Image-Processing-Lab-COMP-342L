# Lab 3 — Contrast enhancement baseline (HE vs CLAHE)

Lab 3 for Digital Image Processing (COMP-342L). Implements the contrast-enhancement baseline for the research project (Labs 3–12): **global histogram equalization** vs **CLAHE** on grayscale images.

---

## Setup (project virtual environment)

From the **project root** (parent of `Lab 3/`):

```bash
# Create and activate virtual environment (if not already)
python -m venv venv
source venv/bin/activate   # macOS/Linux
# Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

If you use **Python 3.9** and the main `requirements.txt` fails (e.g. numpy 2.4 needs Python 3.11+), use:

```bash
pip install -r requirements-py39.txt
```

Required for this lab: `numpy`, `opencv-python` or `scikit-image`, `matplotlib`, `Pillow`.

---

## Run

From the **project root**:

```bash
source venv/bin/activate
cd "Lab 3"
python clahe_vs_he.py
```

Or from anywhere (with venv activated):

```bash
python "Lab 3/clahe_vs_he.py"
```

---

## Input

- Place a test image in **`Lab 3/Content/`** (e.g. `download.jpeg`, `download.jpg`, or `image.png`).
- If `Content/` is empty, the script uses the image at **`Lab 2/Ali Hamza's Lab/Content/download.jpeg`** (if it exists).

---

## Output

- **`Lab 3/Lab3_comparison.png`** — Side-by-side: **Original (grayscale)** | **Global Histogram Equalization** | **CLAHE**.
- Figure is also displayed if you run in an environment with a display (not when using `MPLBACKEND=Agg`).

---

## Files in this lab

| Path | Description |
|------|--------------|
| `Lab 3/README.md` | This file — setup, run, input/output. |
| `Lab 3/clahe_vs_he.py` | Main script: load → grayscale → global HE → CLAHE → plot → save. |
| `Lab 3/Content/` | Put test images here. |
| `Lab 3/PLAN.md` | Implementation plan for this lab. |

---

## Research context (Labs 3–12)

- **Topic:** Low-light and uneven illumination image enhancement (classical and optional hybrid pipelines).
- **Lab 3 goal:** Contrast baseline — compare global HE (often over-enhances noise, flattens local contrast) with CLAHE (limits gain, preserves local detail). This is the first block of the enhancement pipeline and the first comparison for the paper.
- **Roadmap:** Lab 4 = gamma/log → Lab 5–6 = Retinex (SSR, MSR) → Lab 7 = color preservation → Lab 8 = denoising + enhancement → Lab 9–11 = metrics, dataset, ablation → Lab 12 = paper draft.
- **References:** CLAHE in medical/low-light literature (e.g. F1000Research 2021, MDPI Algorithms 2024). Full problem statement and roadmap: main repo [README.md](../README.md) and [Lab 3/README.md](README.md).
