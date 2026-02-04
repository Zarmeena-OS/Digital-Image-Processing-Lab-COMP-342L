# Digital Image Processing Lab — COMP-342L

Repository for lab exercises and the research project (Labs 3–12) for the Digital Image Processing course.

---

## Full implementation report (course-wide)

### 1. Repository structure

| Path                 | Purpose                             |
| -------------------- | ----------------------------------- |
| `Lab 1/` – `Lab 12/` | Per-lab exercises and code          |
| `Lab Manuals/`       | Lab manuals and reference materials |
| `Lab Reports/`       | Lab reports and submissions         |
| `requirements.txt`   | Python dependencies for all labs    |

---

### 2. Dependencies

All labs use a shared environment. Install once from repo root:

```bash
pip install -r requirements.txt
```

**Main packages:** `numpy`, `opencv-python`, `scikit-image`, `scipy`, `matplotlib`, `Pillow`, `scikit-learn`, `ImageIO`, `tifffile`, and standard matplotlib stack.

---

### 3. Lab-by-lab implementation status

| Lab                    | Status      | Implemented                                       | Location                        | Run          |
| ---------------------- | ----------- | ------------------------------------------------- | ------------------------------- | ------------ |
| **Lab 1**              | Placeholder | —                                                 | `Lab 1/README.md` only          | —            |
| **Lab 2**              | Implemented | Image load/display; RGB channel display           | Ali Hamza's Lab, Zarmeena's Lab | See §4 below |
| **Lab 3**              | Implemented | Contrast baseline (global HE vs CLAHE); research roadmap | `Lab 3/`                        | See §4 below |
| **Lab 4** – **Lab 12** | Placeholder | —                                                 | README only in each folder      | —            |

---

### 4. Implemented components (details)

#### Lab 2 — Image I/O and channel display

- **Ali Hamza's Lab**
  - **Part 1.py** — Load image from `Content/download.jpeg`, print shape, display with `matplotlib`.
  - **Part 2.py** — Load image, print height/width/channels, display Red / Green / Blue channels in grayscale (side-by-side).
- **Zarmeena's Lab**
  - **Part 1.py** — Load image from `Content/download.jpg`, print shape, display with `matplotlib`.
  - **Part 4.py** — Load image, print dimensions, display R / G / B channels (side-by-side).

**Run (from repo root):**

```bash
cd "Lab 2/Ali Hamza's Lab"
python Part 1.py    # or Part 2.py
```

```bash
cd "Lab 2/Zarmeena's Lab"
python Part 1.py    # or Part 4.py
```

**Note:** Image path is relative to each lab folder (`Content/download.jpeg` or `Content/download.jpg`). Ensure the image exists in that folder.

---

#### Lab 3 — Research project start: contrast baseline (HE vs CLAHE)

- **Research project (Labs 3–12)**  
  One topic spans Labs 3–12 and leads to a short paper: **low-light and uneven illumination image enhancement** (classical and optional hybrid pipelines). Full problem statement and lab-by-lab roadmap: **[Lab 3/README.md](Lab%203/README.md)**.

- **Implemented in Lab 3** (all under `Lab 3/`, no subfolder)
  - **`Lab 3/clahe_vs_he.py`** — Contrast baseline: **global histogram equalization** vs **CLAHE** on grayscale images.
    - **Functions:** `load_grayscale`, `global_he`, `clahe`, `main`.
    - **Input:** Image in `Lab 3/Content/` (e.g. `download.jpeg`); script falls back to Lab 2 image if none.
    - **Output:** Side-by-side figure (original | global HE | CLAHE), saved as `Lab 3/Lab3_comparison.png`.
  - **`Lab 3/Content/`** — Put test images here.
  - **`Lab 3/PLAN.md`** — Implementation plan for this lab.

**Run (from repo root):**

```bash
cd "Lab 3"
python clahe_vs_he.py
```

Place an image in `Lab 3/Content/` (e.g. `download.jpeg`) or rely on fallback to Lab 2 image.

---

### 5. Research project roadmap (Labs 3 → 12)

| Lab   | Focus                                                             | Output                                 |
| ----- | ----------------------------------------------------------------- | -------------------------------------- |
| **3** | Contrast baseline (global HE vs CLAHE)                           | Code + figures                         |
| 4     | Gamma / log / power-law transforms                                | Code + comparison                      |
| 5     | Single-scale Retinex (SSR)                                        | SSR implementation                     |
| 6     | Multi-scale Retinex (MSR / MSRCR)                                 | MSR pipeline                           |
| 7     | Color preservation (e.g. HSV/Lab)                                 | Color-stable pipeline                  |
| 8     | Denoising + enhancement (e.g. bilateral + CLAHE)                  | Combined pipeline                      |
| 9     | Objective metrics (PSNR, SSIM) + subjective protocol              | Evaluation code + tables               |
| 10    | Dataset curation + batch evaluation                               | Results on dataset                     |
| 11    | Ablation / parameter study or hybrid method                       | Tables + analysis                      |
| 12    | Paper draft                                                       | Intro, method, experiments, conclusion |

Full problem statement and research angle: **[Lab 3/README.md](Lab%203/README.md)**.

---

### 6. File index (implementations)

| File                              | Lab | Description                     |
| --------------------------------- | --- | ------------------------------- |
| `Lab 2/Ali Hamza's Lab/Part 1.py` | 2   | Load and display image          |
| `Lab 2/Ali Hamza's Lab/Part 2.py` | 2   | Display R/G/B channels          |
| `Lab 2/Zarmeena's Lab/Part 1.py`  | 2   | Load and display image          |
| `Lab 2/Zarmeena's Lab/Part 4.py`  | 2   | Display R/G/B channels          |
| `Lab 3/clahe_vs_he.py`            | 3   | Global HE vs CLAHE (contrast baseline) |

---

### 7. Setup and usage

**Environment:**

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# or: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**Usage:**  
Navigate to the lab folder and run the corresponding script (see §4 and §6). All paths in scripts are relative to the folder containing the script or to `Content/` inside that folder.

---

## Summary

- **Labs with code:** Lab 2 (image I/O, RGB channels), Lab 3 (contrast baseline: HE vs CLAHE + research roadmap).
- **Labs 4–12:** Planned; see Lab 3 README for the research roadmap.
- **Single research thread:** Low-light / uneven illumination enhancement (and related classical pipelines) leading to a short paper by Lab 12.
