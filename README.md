# Digital Image Processing Lab — COMP-342L

Repository for lab exercises and the **research project (Labs 3–12)** for the Digital Image Processing course. All 12 labs are implemented; Labs 3–12 form one pipeline: **low-light and uneven illumination image enhancement**.

---

## 1. Repository structure

| Path                    | Purpose                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `Lab 1/` – `Lab 12/`    | Per-lab exercises and code                                   |
| `Lab Manuals/`          | Lab manuals and reference materials                          |
| `Lab Reports/`          | Lab reports and submissions                                  |
| `requirements.txt`      | Python dependencies (Python 3.10+ preferred)                 |
| `requirements-py39.txt` | Relaxed deps for Python 3.9                                  |
| `problem_statement.tex` | LaTeX: problem statement + full implementation (all 12 labs) |

---

## 2. Setup (virtual environment)

From the **project root**:

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you use **Python 3.9** and the main `requirements.txt` fails:

```bash
pip install -r requirements-py39.txt
```

**Main packages:** `numpy`, `opencv-python`, `scikit-image`, `scipy`, `matplotlib`, `Pillow`.

---

## 3. Lab-by-lab implementation (all 12 labs)

| Lab        | Focus                                       | Script                                                         | Output                            |
| ---------- | ------------------------------------------- | -------------------------------------------------------------- | --------------------------------- |
| **Lab 1**  | Placeholder                                 | —                                                              | —                                 |
| **Lab 2**  | Image I/O, RGB channels                     | `Lab 2/Ali Hamza's Lab/Part 1.py`, `Part 2.py`; Zarmeena's Lab | Display                           |
| **Lab 3**  | Contrast baseline (global HE vs CLAHE)      | `Lab 3/clahe_vs_he.py`                                         | `Lab3_comparison.png`             |
| **Lab 4**  | Gamma / log / power-law transforms          | `Lab 4/gamma_log_transforms.py`                                | `Lab4_gamma_log.png`              |
| **Lab 5**  | Single-scale Retinex (SSR)                  | `Lab 5/ssr.py`                                                 | `Lab5_ssr.png`                    |
| **Lab 6**  | Multi-scale Retinex (MSR)                   | `Lab 6/msr.py`                                                 | `Lab6_msr.png`                    |
| **Lab 7**  | Color preservation (CLAHE on V in HSV)      | `Lab 7/color_preservation.py`                                  | `Lab7_color_preserved.png`        |
| **Lab 8**  | Denoising + enhancement (bilateral + CLAHE) | `Lab 8/denoise_enhance.py`                                     | `Lab8_denoise_enhance.png`        |
| **Lab 9**  | Objective metrics (PSNR, SSIM)              | `Lab 9/metrics.py`                                             | `Lab9_metrics.png`                |
| **Lab 10** | Batch evaluation on dataset                 | `Lab 10/batch_eval.py`                                         | `Lab 10/outputs/<name>_clahe.png` |
| **Lab 11** | Ablation (CLAHE clip_limit, kernel_size)    | `Lab 11/ablation.py`                                           | Console + `Lab11_ablation.txt`    |
| **Lab 12** | Paper draft                                 | `Lab 12/paper_draft.md`                                        | —                                 |

---

## 4. How to run each lab

From project root (with venv activated):

```bash
# Lab 2
cd "Lab 2/Ali Hamza's Lab"
python "Part 1.py"    # or Part 2.py

# Lab 3 – 12 (research pipeline)
cd "Lab 3"  && python clahe_vs_he.py
cd "Lab 4"  && python gamma_log_transforms.py
cd "Lab 5"  && python ssr.py
cd "Lab 6"  && python msr.py
cd "Lab 7"  && python color_preservation.py
cd "Lab 8"  && python denoise_enhance.py
cd "Lab 9"  && python metrics.py
cd "Lab 10" && python batch_eval.py
cd "Lab 11" && python ablation.py
# Lab 12: paper_draft.md — no script
```

**Input images:** Place images in `Lab 3/Content/` (or the lab’s `Content/`). Scripts fall back to `Lab 2/Ali Hamza's Lab/Content/download.jpeg` if needed.

---

## 5. Research project (Labs 3–12)

- **Problem:** Low-light and uneven illumination image enhancement using classical (and optional hybrid) pipelines.
- **Deliverable:** Short paper (intro, related work, method, experiments, conclusion). Draft: `Lab 12/paper_draft.md`. LaTeX: `problem_statement.tex`.
- **Pipeline:** Contrast (HE, CLAHE) → gamma/log → Retinex (SSR, MSR) → color preservation (HSV) → denoising + enhancement → metrics (PSNR, SSIM) → batch evaluation → ablation → paper.

Full problem statement and lab-by-lab description: **`problem_statement.tex`** (and PDF after `pdflatex problem_statement.tex`).

---

## 6. File index (implementations)

| File                              | Lab | Description            |
| --------------------------------- | --- | ---------------------- |
| `Lab 2/Ali Hamza's Lab/Part 1.py` | 2   | Load and display image |
| `Lab 2/Ali Hamza's Lab/Part 2.py` | 2   | Display R/G/B channels |
| `Lab 2/Zarmeena's Lab/Part 1.py`  | 2   | Load and display image |
| `Lab 2/Zarmeena's Lab/Part 4.py`  | 2   | Display R/G/B channels |
| `Lab 3/clahe_vs_he.py`            | 3   | Global HE vs CLAHE     |
| `Lab 4/gamma_log_transforms.py`   | 4   | Gamma, log transforms  |
| `Lab 5/ssr.py`                    | 5   | Single-scale Retinex   |
| `Lab 6/msr.py`                    | 6   | Multi-scale Retinex    |
| `Lab 7/color_preservation.py`     | 7   | CLAHE on V (HSV)       |
| `Lab 8/denoise_enhance.py`        | 8   | Bilateral + CLAHE      |
| `Lab 9/metrics.py`                | 9   | PSNR, SSIM             |
| `Lab 10/batch_eval.py`            | 10  | Batch CLAHE            |
| `Lab 11/ablation.py`              | 11  | Parameter study        |
| `Lab 12/paper_draft.md`           | 12  | Paper draft            |

---

## 7. LaTeX document

**`problem_statement.tex`** — Full problem statement and implementation report for all 12 labs. Build with:

```bash
pdflatex problem_statement.tex
```

---

## 8. Summary

- **Labs 1–2:** Course basics (image I/O, channels).
- **Labs 3–12:** Full enhancement pipeline implemented; each lab has a script, README, and (where applicable) output figures. Lab 12 provides the paper draft; `problem_statement.tex` is the formal problem statement and implementation report for the whole project.
