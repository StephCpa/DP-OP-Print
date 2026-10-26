# DP-OP-Print

Can membership leakage in on-policy self-distillation (OPSD) be predicted *ex ante* from the initial policy θ0? What do the training mechanics, in particular OPSD's element-wise JSD clip, imply for the OP-Print membership attack (arXiv 2610.31616) and for DP defenses?

This repository holds three things:

- the experimental design (pre-registration style, not yet run on GPUs);
- small verification scripts;
- a round of pre-GPU desk research that tests the design: a code audit, numerics, clip mechanics, design statistics and a toy OPSD dry run.

**Main line:**
- **H0.** Replicate OP-Print on Qwen3-1.7B (LoRA r = 64, lr 5e-6, clip 0.05, effective batch 32).
- **H1 / H1′.** Test whether a θ0 concentration statistic predicts per-sample leakage beyond θ0 leverage and loss, against the mechanistic competitor r_dir. Evaluate reference-calibrated attacks.
- **H3.** Test the clip mechanism: the G-Max − G gap as a function of clip strength, using sign-agnostic attacks.

DP comparisons (H2), the top-k teacher (H4) and the numerical training ablation (H5) are conditional.

## Documents

| File | Description |
|---|---|
| `docs/op-print-design-v1.3.md` | **Current design.** It folds in the desk research below. |
| `docs/results/research-report-2026-10-26.md` | **Desk research report.** Findings F1–F13 and the design changes they imply. |
| `docs/verification/opsd-code-audit.md` | OPSD source audit: loss temperature, thinking-mode asymmetry, padding, step budget, clip semantics. |
| `docs/archive/op-print-design-v1.2.md` / `.pdf` | v1.2 (English) |
| `docs/archive/op-print-design-v1.1.md` / `.pdf` | v1.1 (English): numerically stable Δ, noise-floor ε, H5 |
| `docs/archive/op-print-design-v1.md` / `.pdf` | v1 (Chinese): first critique folded in |
| `docs/archive/op-print-design-v0.md` / `.pdf` | v0 (Chinese): initial design |

## Scripts and experiments

All of these run on CPU. Each experiment directory commits its outputs (`*.out.txt`, `*.json`) next to the script.

| Path | What it checks | Runtime |
|---|---|---|
| `scripts/toy_clip_grad.py` | v1.2 Appendix A: fp64 clip-gradient algebra, the worked example, and KL formulas at V = 50 | seconds |
| `experiments/numerics/large_vocab_numerics.py` | OPSD's loss path and KL formulas at V = 151,936 in bf16/fp32/fp64, T ∈ {1, 1.1}, and the logit-quantisation floor with a synthetic LM head | ~2 min, ~6 GB RAM |
| `experiments/clip_mechanism/clip_mechanism.py` | Clamped logits are pushed down; negative-loss optima of the clipped loss; one-step direction of the clamped token; r_dir vs m_U | ~2 min |
| `experiments/design_sim/design_sim.py` | Reliability-gate feasibility and post-H0 bound; validity of H1 tests (sign-flip placebo vs HC3); H3 MDE; chunk-1 spike | ~3 min |
| `scripts/render_pdf.py` | Renders the Markdown documents to PDF (python-markdown, then Node Playwright/Chromium) | seconds |
| `experiments/toy_opsd/` | Toy OPSD simulator and an end-to-end dry run of H0/H1/H1′/H3 and the §1.0 diagnostic. `run_all.sh`, then `analyze.py`. Results are in `results.md`. | ~25 min |

Requirements: `numpy`, `scipy`, `torch` (CPU is fine), `scikit-learn`, `mpmath`.

```
python3 scripts/toy_clip_grad.py
python3 experiments/numerics/large_vocab_numerics.py
python3 experiments/clip_mechanism/clip_mechanism.py
python3 experiments/design_sim/design_sim.py
bash experiments/toy_opsd/run_all.sh && python3 experiments/toy_opsd/analyze.py
```

## Notes

- Third-party papers and the OPSD code (github.com/siyan-zhao/OPSD) are not included. See the references in the design document.
- The desk research ran without GPU access, and without access to huggingface.co or arxiv.org. No Qwen3 weights were used, and OP-Print could not be re-read. The report labels which statements this affects.
