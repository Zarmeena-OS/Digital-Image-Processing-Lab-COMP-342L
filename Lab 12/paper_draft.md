# Paper draft — Low-light and uneven illumination image enhancement

**Title:** Low-light and uneven illumination image enhancement: a classical pipeline (Labs 3–12).

**Authors:** [Your name]  
**Course:** Digital Image Processing (COMP-342L).

---

## 1. Introduction

Low-light and unevenly lit images suffer from low contrast, noise, and color distortion. This report describes a classical enhancement pipeline implemented across Labs 3–12: contrast enhancement (global HE, CLAHE), gamma/log transforms, Retinex (SSR, MSR), color preservation in HSV, denoising (bilateral) plus CLAHE, and evaluation with PSNR/SSIM and batch/ablation studies.

**Goal:** Design, implement, and evaluate a practical pipeline for low-light/uneven illumination enhancement and compare methods in terms of contrast, noise, and color fidelity.

---

## 2. Related work

- **CLAHE:** Contrast Limited Adaptive Histogram Equalization; used in medical imaging (F1000Research 2021, MDPI Algorithms 2024). Limits contrast gain and preserves local detail vs global HE.
- **Retinex:** Illumination–reflectance decomposition (SSR, MSR) for uneven lighting.
- **Color preservation:** Applying enhancement only to the V channel in HSV avoids color shift.

---

## 3. Method (pipeline)

1. **Lab 3 — Contrast baseline:** Global HE vs CLAHE on grayscale.
2. **Lab 4 — Gamma/log:** Power-law and log transforms for tone mapping.
3. **Lab 5 — SSR:** Single-scale Retinex R = log(I) − log(G∗I).
4. **Lab 6 — MSR:** Multi-scale Retinex (weighted sum of SSR at σ = 15, 80, 250).
5. **Lab 7 — Color preservation:** CLAHE on V in HSV; H, S unchanged.
6. **Lab 8 — Denoising + enhancement:** Bilateral filter then CLAHE.
7. **Lab 9 — Metrics:** PSNR and SSIM (ref vs enhanced).
8. **Lab 10 — Batch:** Run CLAHE on dataset; save outputs.
9. **Lab 11 — Ablation:** Vary CLAHE clip_limit and kernel_size; report PSNR/SSIM.
10. **Lab 12 — Paper:** This draft.

---

## 4. Experiments

- **Datasets:** Images in Lab 3/Content (and fallbacks). Batch evaluation (Lab 10) on all images in Content/.
- **Metrics:** PSNR (dB), SSIM (Lab 9). Ablation over clip_limit and kernel_size (Lab 11).
- **Visual results:** Figures saved in each lab (e.g. Lab3_comparison.png, Lab4_gamma_log.png, …).

---

## 5. Discussion

- Global HE often over-enhances noise; CLAHE gives more controlled local contrast.
- Gamma < 1 brightens; gamma > 1 darkens. Log expands dark regions.
- SSR/MSR improve visibility under uneven illumination; MSR combines multiple scales.
- Color preservation (HSV V-only) avoids hue/saturation shift.
- Bilateral + CLAHE reduces noise while enhancing contrast.

---

## 6. Conclusion

A full classical pipeline for low-light/uneven illumination enhancement was implemented across 12 labs: contrast (HE, CLAHE), gamma/log, Retinex (SSR, MSR), color preservation, denoising + enhancement, and evaluation (PSNR, SSIM, batch, ablation). The pipeline is modular and can be extended (e.g. learning-based modules) for future work.

---

## References

- F1000Research (2021). Enhancement of digitized X-ray films using CLAHE.
- MDPI Algorithms (2024). Impact of image enhancement using CLAHE on spine X-ray segmentation.
- He et al., Single image haze removal using dark channel prior (TPAMI 2010) — related dehazing.
