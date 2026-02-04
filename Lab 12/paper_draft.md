# Low-Light and Uneven Illumination Image Enhancement: A Classical Pipeline

**Course:** Digital Image Processing (COMP-342L)  
**Project:** Labs 3–12 — Full implementation and report  
**Author:** [Your Name]

---

## Abstract

This report presents a complete classical pipeline for **low-light and uneven illumination image enhancement**, implemented across Labs 3–12. We design, implement, and evaluate histogram-based methods (global HE, CLAHE), gamma and log transforms, Single- and Multi-Scale Retinex (SSR, MSR), color-preserving enhancement in HSV, and a denoising-plus-enhancement stage (bilateral filter + CLAHE). We report objective metrics (PSNR, SSIM), batch evaluation on a dataset, and an ablation study over CLAHE parameters. The pipeline is modular, reproducible, and suitable as a baseline for future learning-based or hybrid extensions.

**Keywords:** Image enhancement, low-light, CLAHE, Retinex, histogram equalization, color preservation, PSNR, SSIM.

---

## 1. Introduction

### 1.1 Motivation

Images captured under low light or uneven illumination often suffer from:

- **Low contrast** — Dark and bright regions are compressed; detail is lost.
- **Noise** — Sensor noise is more visible in dark areas.
- **Color distortion** — Global adjustments can shift hue and saturation.

Applications include medical imaging (X-ray, microscopy), surveillance, consumer photography, and document scanning. Classical (non–deep learning) methods remain important for interpretability, low resource use, and as baselines.

### 1.2 Objective

We set out to:

1. **Design** a practical enhancement pipeline using classical building blocks.
2. **Implement** each stage in Python (Labs 3–11) with clear inputs/outputs.
3. **Evaluate** with objective metrics (PSNR, SSIM), batch runs, and parameter ablation.
4. **Document** the problem, method, and results in this report (Lab 12).

### 1.3 Scope

The pipeline covers: **contrast baseline (HE vs CLAHE)** → **gamma/log transforms** → **Retinex (SSR, MSR)** → **color preservation (HSV)** → **denoising + enhancement** → **metrics and ablation**. All code and figures are in the course repository; each lab has a README and runnable script.

---

## 2. Related Work

- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Used widely in medical imaging. F1000Research (2021) showed CLAHE on digitized X-ray films with PSNR ~31–32 dB and high diagnostic agreement. MDPI Algorithms (2024) studied CLAHE with U-Net/Mask R-CNN for spine X-ray segmentation. CLAHE limits contrast gain via a clip limit and uses local tiles, preserving detail and controlling noise better than global HE.

- **Retinex theory:** Decomposes an image into reflectance and illumination. Single-Scale Retinex (SSR) and Multi-Scale Retinex (MSR) improve visibility under uneven lighting; MSR combines multiple Gaussian scales (e.g. σ = 15, 80, 250) for a more balanced result.

- **Color preservation:** Applying enhancement only to the luminance (e.g. V in HSV) and leaving hue and saturation unchanged avoids color shift; we use this in Lab 7.

- **Denoising + enhancement:** Edge-preserving filters (e.g. bilateral) before contrast enhancement reduce noise amplification; we combine bilateral filter with CLAHE in Lab 8.

---

## 3. Method — Pipeline Overview

The pipeline is implemented in **Labs 3–11**; each lab produces figures and/or metrics used in this report.

| Lab | Component | What we did | Output |
|-----|------------|-------------|--------|
| **3** | Contrast baseline | Global histogram equalization vs CLAHE on grayscale | Side-by-side: original, global HE, CLAHE |
| **4** | Gamma / log | Power-law (γ = 0.5, 1.5) and log transform | Comparison of tone-mapping effects |
| **5** | Single-Scale Retinex | R = log(I) − log(G∗I) with Gaussian G; σ = 15, 80 | SSR at two scales |
| **6** | Multi-Scale Retinex | Weighted sum of SSR at σ = 15, 80, 250 | Single MSR result |
| **7** | Color preservation | CLAHE applied only to V channel in HSV; H, S unchanged | Color-preserved enhancement |
| **8** | Denoising + enhancement | Bilateral filter (edge-preserving) then CLAHE | Pipeline: original → denoised → enhanced |
| **9** | Metrics | PSNR and SSIM between reference and enhanced image | Numerical results + figure |
| **10** | Batch evaluation | Run CLAHE on all images in Content/; save outputs | One enhanced image per input |
| **11** | Ablation | Vary CLAHE clip_limit (0.01, 0.03, 0.06) and kernel_size (32, 64, 128) | PSNR/SSIM table |
| **12** | Report | This document | — |

### 3.1 Key Algorithms (short)

- **Global HE:** CDF-based mapping so output histogram is approximately uniform.
- **CLAHE:** Divide image into tiles; equalize each with a clip limit; interpolate to avoid block artifacts.
- **Gamma:** *out* = *I*^γ; γ < 1 brightens, γ > 1 darkens.
- **Log:** *out* ∝ log(1 + *I*); expands dark regions.
- **SSR:** *R* = log(*I*) − log(G∗*I*); *G* = Gaussian blur.
- **MSR:** *R* = Σ wₖ · SSR(*I*, σₖ) with weights wₖ.
- **Bilateral filter:** Smoothing that preserves edges (spatial + range kernel).

---

## 4. Experiments

### 4.1 Setup

- **Environment:** Python 3.9+; `numpy`, `opencv-python`, `scikit-image`, `scipy`, `matplotlib`, `Pillow` (see `requirements.txt` / `requirements-py39.txt` in the repo).
- **Data:** Test images in `Lab 3/Content/` (and fallback to `Lab 2/Ali Hamza's Lab/Content/download.jpeg` when no image is placed in a lab’s Content folder).
- **Reproducibility:** From the project root, each lab is run as: `cd "Lab N" && python <script>.py`.

### 4.2 Metrics

- **PSNR (Peak Signal-to-Noise Ratio):** In dB; higher is better (reference vs enhanced).
- **SSIM (Structural Similarity Index):** In [0, 1]; higher is better.

We use these in Lab 9 (single comparison) and Lab 11 (ablation over CLAHE parameters).

### 4.3 Results (summary)

- **Lab 3:** CLAHE yields more controlled local contrast than global HE; global HE can over-enhance noise.
- **Lab 4:** Gamma 0.5 brightens; gamma 1.5 darkens; log expands dark regions.
- **Lab 5–6:** SSR at small σ captures fine detail; MSR combines scales for a more natural look.
- **Lab 7:** CLAHE on V in HSV preserves color while improving luminance.
- **Lab 8:** Bilateral + CLAHE reduces noise and improves contrast without strong artifacts.
- **Lab 9:** Example PSNR/SSIM (e.g. ref vs gamma-enhanced) show the metrics in use.
- **Lab 10:** Batch run produces one enhanced image per input in `Lab 10/outputs/`.
- **Lab 11:** Ablation shows that CLAHE clip_limit and kernel_size affect PSNR/SSIM; e.g. lower clip_limit can give higher PSNR but different perceived contrast.

All figures are saved in the corresponding lab folders (e.g. `Lab3_comparison.png`, `Lab4_gamma_log.png`, …, `Lab9_metrics.png`). The professor can regenerate them by running the scripts from the repository.

---

## 5. Discussion

- **Global HE vs CLAHE:** Global HE often over-enhances noise and flattens local contrast; CLAHE’s clip limit and local tiles give more controlled enhancement and are preferred in medical and low-light literature.
- **Gamma and log:** Simple and fast; useful for tone mapping; choice of γ (or log constant) is application-dependent.
- **Retinex:** SSR/MSR improve visibility under uneven illumination; MSR is more robust across scales. Both assume a multiplicative illumination model.
- **Color preservation:** Working in HSV and enhancing only V avoids the color shifts that can occur when applying grayscale methods to RGB channels separately.
- **Denoising before enhancement:** Bilateral filtering before CLAHE helps avoid amplifying noise; the order (denoise → enhance) is important.

---

## 6. Conclusion

We implemented a **full classical pipeline** for low-light and uneven illumination image enhancement across Labs 3–12: contrast baseline (global HE, CLAHE), gamma/log transforms, Single- and Multi-Scale Retinex, color-preserving CLAHE in HSV, denoising plus enhancement (bilateral + CLAHE), and evaluation with PSNR/SSIM, batch processing, and parameter ablation. The pipeline is modular and reproducible; all code and outputs are in the course repository. This work can serve as a baseline for future extensions (e.g. learning-based or hybrid methods) and for comparison in reporting.

---

## 7. Repository and How to Run

- **Repository:** All labs, scripts, and this report are in the course repo (e.g. `Digital-Image-Processing-Lab-COMP-342L`).
- **Run any lab:** From the project root, with the virtual environment activated:
  - `cd "Lab 3"  && python clahe_vs_he.py`
  - `cd "Lab 4"  && python gamma_log_transforms.py`
  - … (same pattern for Labs 5–11; see main `README.md` for the full list).
- **Input:** Place an image in `Lab 3/Content/` (e.g. `download.jpeg`); scripts fall back to Lab 2’s image if needed.
- **Outputs:** Each lab writes figures to its folder (e.g. `Lab3_comparison.png`); Lab 10 writes to `Lab 10/outputs/`; Lab 11 prints a table and writes `Lab11_ablation.txt`.

A **LaTeX problem statement** (`problem_statement.tex` in the repo root) summarizes the same pipeline and all 12 labs; it can be compiled with `pdflatex problem_statement.tex` for a PDF version.

---

## References

1. F1000Research (2021). Enhancement of digitized X-ray films using Contrast-Limited Adaptive Histogram Equalization (CLAHE).
2. MDPI Algorithms (2024). Impact of image enhancement using CLAHE, anisotropic diffusion, and histogram equalization on spine X-ray segmentation with U-Net, Mask R-CNN, and transfer learning.
3. He et al., Single image haze removal using dark channel prior, IEEE TPAMI 2010 (related: dehazing and atmospheric model).
4. Jobson et al., A multiscale retinex for bridging the gap between color images and the human observation of scenes, IEEE TIP 1997 (Retinex theory and MSR).
