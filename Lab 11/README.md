# Lab 11 — Ablation / parameter study

Vary CLAHE clip_limit and kernel_size; report PSNR and SSIM vs reference.

## Run

```bash
cd "Lab 11"
python ablation.py
```

## Output

- Console: table of (clip_limit, kernel_size) → PSNR, SSIM.
- **`Lab 11/Lab11_ablation.txt`** — short summary.

## Files

- `ablation.py` — main script; `Content/` — optional images.
