# H0 per-token scores and prompts (run of 2026-10-27)

Exported from the H0 run directories on the a6000 server by `export_h0.py` (offline, CPU only). Nothing was re-scored: every number is copied from the scorer's stored outputs (`score/tok_XXXX.pt`, `score/records.jsonl`).

| Arm | Run dir (`~/dp-op-print/runs/`) | Teacher template | Student template |
|---|---|---|---|
| primary ("main") | `h0_primary_split0A` | thinking (`teacher_thinking=True`) | non-thinking (`student_thinking=False`) |
| control | `h0_control_split0A` | non-thinking | non-thinking |

Both arms: Qwen3-1.7B + LoRA, split 0A, 1000 samples (500 members / 500 non-members), Openthoughts_math_30k_opsd.

## Files

| File | Content |
|---|---|
| `h0_{arm}_T1.1.npz` | Input for `experiments/gpu_checks/gmax_rescore.py` (keys below) |
| `h0_{arm}_prompts.jsonl` | One line per sample: `id`, `member`, `student_prompt`, `teacher_prompt` (exact rendered strings, chat template applied, pads stripped), their token ids, rollout indices `k`, and the stored first-token KL `delta1[{theta,theta0,dtrain}_T{1.0,1.1}]` for every rollout. Input for `first_token_check.py --prompts` |
| `h0_{arm}_delta1_theta_T1.0.json` | `{id: [Δ₁ per rollout]}`, theta, T = 1.0 (the temperature `first_token_check.py` compares against). Input for `--pipeline` |
| `export_report.json` | Shapes, K histogram, truncation, round-trip check, Δ₁ spread across rollouts |
| `gmax_rescore_output.json` | Output of `gmax_rescore.py h0_primary_T1.1.npz h0_control_T1.1.npz --boot 1000` |
| `export_h0.py` | The export script |

### `.npz` keys

- `delta` float16 `[1000, 8, 1024]`: per-token Δ_t = KL(p(·|teacher ctx, y<t) ‖ p(·|student ctx, y<t)) under θ (adapter on both sides), **T = 1.1**, log1p form. NaN past each rollout's length.
- `delta0` float16, same shape: the same under θ0 (adapter disabled), on the same rollouts (CRN).
- `dtheta` float16, same shape: the training objective D_θ (θ0 teacher context vs θ student context), as `gmax_rescore.py` expects for f_rm. **Note:** this is *not* `delta − delta0`; compute that from the two arrays if needed.
- `member` bool `[1000]`: 1 = member (in the training split).
- `idx` int32 `[1000]`: dataset row index; `Lmax` = 1024 (completion cap; a rollout of length 1024 is truncated).
- K = 8: rollouts k = 0..7. 200 samples per arm have 32 rollouts in the run; only k < 8 is exported, matching the H0 summary (`summarize_h0.py` uses k < 8).

## Conventions

- **Lower score ⇒ member.** AUC = P(score_member < score_non-member); AUC > 0.5 is the OP-Print direction, and values below 0.5 are reported, never folded.
- Scoring: bf16 trunk, LM head in fp32 on the final hidden state, temperatures 1.0 and 1.1 (exported: 1.1), 64-token chunks for G / G-Max.
- Prompts in the scorer were built with `SelfDistillationDataCollator(..., prompt_padding_side="left")`. Pads were then stripped via the attention mask, and each sequence was `prompt + completion`, right-padded only after the completion. So **scoring has no padding inside the context**. A batch-size-1 recomputation reproduces the stored Δ₁: 22.90 vs 22.89 nats, primary, θ, T1.1, 20 samples.
- **Training padding:** both arms were **trained** with `prompt_padding_side=right` (the upstream default, confirmed in the training log). The first-token pad-position issue is therefore present in the **training loss** of both arms, even though the scoring is clean.

## Reproduction check

`python3 experiments/gpu_checks/gmax_rescore.py data/h0/2026-10-27/h0_primary_T1.1.npz data/h0/2026-10-27/h0_control_T1.1.npz` gives G-Max 0.490 (primary) / 0.458 (control) and G-Max from segment 2 0.460 / 0.466, matching the operator's numbers. θ0 G-Max (computed with the script's `chunk_stats`) is 0.481 / 0.459.
