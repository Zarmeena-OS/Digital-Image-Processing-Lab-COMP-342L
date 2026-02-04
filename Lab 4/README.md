# Lab 4 — Gamma / log / power-law transforms

Part of the enhancement pipeline (Labs 3–12). Applies **gamma** and **log** transforms for comparison with Lab 3 contrast baseline.

## Run

From project root (with venv activated):

```bash
cd "Lab 4"
python gamma_log_transforms.py
```

## Input

Image in `Lab 4/Content/` or `Lab 3/Content/` or Lab 2 image (fallback).

## Output

- **`Lab 4/Lab4_gamma_log.png`** — 2×2: Original | Gamma=0.5 | Gamma=1.5 | Log (c=1).

## Files

- `gamma_log_transforms.py` — main script.
- `Content/` — optional test images.
