# DP-OP-Print

Can membership leakage in on-policy self-distillation (OPSD) be predicted *ex ante* from the initial policy θ0? What do the training mechanics, in particular OPSD's element-wise JSD clip, imply for the OP-Print membership attack (arXiv 2610.31616) and for DP defenses?

This repository holds the experimental design, which is pre-registration style and not yet run, together with small verification scripts.

**Main line:**
- **H0.** Replicate OP-Print on Qwen3-1.7B (LoRA r = 64, lr 5e-6, clip 0.05).
- **H1 / H1′.** Test whether a θ0 concentration statistic predicts per-sample leakage beyond θ0 leverage and loss, and evaluate reference-calibrated attacks.
- **H3.** Test the clip mechanism: the G-Max − G gap as a function of clip strength.

DP comparisons (H2), the top-k teacher (H4) and the numerical training ablation (H5) are conditional.

## Documents

| File | Description |
|---|---|
| `docs/op-print-design-v1.2.md` / `.pdf` | **Current design (English).** |
| `docs/archive/op-print-design-v1.1.md` / `.pdf` | v1.1 (English): numerically stable Δ, noise-floor ε, H5 |
| `docs/archive/op-print-design-v1.md` / `.pdf` | v1 (Chinese): first critique folded in |
| `docs/archive/op-print-design-v0.md` / `.pdf` | v0 (Chinese): initial design |

## Scripts

`scripts/toy_clip_grad.py` is an fp64 numpy toy. It checks three things:

- the gradient of OPSD's element-wise clipped KL, g_j = q_j·P_U − p_j·1[j∈U], against finite differences;
- the worked example (clip ∈ {0, 0.05});
- log-space KL formulas (Barron et al., arXiv 2610.31642) against a 50-digit reference, in fp64 and fp32.

To run it:

```
python3 scripts/toy_clip_grad.py   # needs numpy only; expected output in scripts/toy_clip_grad.expected_output.txt
```

## Notes

Third-party papers and the OPSD code (github.com/siyan-zhao/OPSD) are not included. See the references in the design document.
