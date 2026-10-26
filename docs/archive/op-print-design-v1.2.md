# Ex-ante prediction of membership leakage in on-policy self-distillation (OPSD): experimental design v1.2 (2026-10-26)

**Basis.** v1.2 builds on v1.1. It adds three things:

- Alex's review of v1.1.
- An fp64 toy verification (Appendix A).
- A re-read of OPSD's README and OP-Print's Table 2.

**Markers.**

- **[unverified]**, **[inference]** and **[estimate]** label statements not taken directly from a primary source.
- **[assumption]** labels modelling assumptions.
- **[design choice]** labels pre-registered thresholds that are ours.

**Earlier versions.** v0 and v1 (in Chinese) and v1.1 (in English) are archived unchanged.

## Changelog v1.1 → v1.2

1. **The clip mechanism is now central (§0.1).**
   - Under OPSD's element-wise clip, the gradient is g_j = q_j·P_U − p_j·1[j∈U].
   - At a strong-correction site, the teacher's preferred token is clamped and receives no direct upward gradient. In the worked example its probability *falls*.
   - **The algebra, the example, and both v1.1 formulas are confirmed in fp64** (Appendix A).
   - Consequence: under clip = 0.05, the high-Δ sites that G-Max reads are not trained directly, so any imprint there must arise indirectly.
2. **H3 is rewritten.**
   - Primary outcome: the G-Max − G AUC gap as a function of clip strength, tested two-sided.
   - Sign-agnostic attacks are added.
   - Matched utility may be unreachable at clip = 0.01, and the sign may flip there.
   - Existing weak evidence from OP-Print's Table 2 is noted.
3. **H1 changes.**
   - New mechanistic competitor: trainable mass m_U(x).
   - **One** primary c (the top-10% chunk share).
   - Pre-registered predictions at clip = 0 and clip = 0.05.
   - The placebo is now a within-pair sign flip (16 configurations per sample).
4. **New first diagnostic (§1.0).** Log the token at each chunk's argmax-Δ position, under θ and θ0: style vs math tokens.
5. **f_rm** is now based on the θ0-teacher residual. H0 runs at lr 5e-6 regardless; the LR grid is a fallback only.
6. **Noise floor.**
   - New N2 test: a full-fp32 reference forward pass on about 100 samples, with relative error reported separately for in- and out-samples. ε is calibrated from N2.
   - N1 is kept as a lower bound.
   - The reviewer's prior (≈ 3e-4 nats/token) is added as [estimate].
   - The toy shows that direct fp32 KL is unreliable at small δ.
7. **H5 is conditional** on three triggers. Drift is measured as a per-layer ratio ‖W_base drift‖_F / ‖(α/r)BA‖_F.
8. **H2 is conditional** on a utility-only pilot. The adapter parameter count was **verified from the HF config**: 69.7M at r = 64, 8.7M at r = 8.
9. **Scope.** The minimal paper is H0 → H1/H1′ → H3. H2, H4 and H5 are conditional. Compute has been re-estimated (§3).
10. **§7.** The reviewer's concessions are recorded, and the FPC common-factor claim is now labelled **[assumption]**.
11. **New facts from the OPSD README.**
    - The clip was introduced because style tokens ("wait", "think") show 6–15× higher KL than math tokens.
    - The README's non-thinking result tables use `--jsd_token_clip 1e-6` (1.7B, 4B) and `1e-7` (8B).
    - The tinker-style loss has "no clipped implemented".
12. **Repository.** A repo-ready folder `dp-op-print-repo/` holds the docs, the archive, and the toy script. It has not been pushed.

The v1 → v1.1 and v0 → v1 changelogs are in the archived v1.1 document.

## 0. Facts pinned from the paper and code

- **Signal.** Δ_t = KL(p_θ(·|p_t(q,r), y<t) ‖ p_θ(·|p_s(q), y<t)).
  - Both sides use the attacked model θ, so the teacher side runs **with the adapter on**.
  - Rollouts are sampled as y ~ p_θ(·|p_s).
- **Gray-box scores.** G averages Δ over chunks, and G-Max takes the maximum over 64-token chunks. Both are averaged over K full rollouts (default K = 32).
- **Training loss (OPSD, β = 0).** The summand is f_k = p_k(log p_k − log q_k), where:
  - p is the teacher distribution: θ0 (adapter disabled) under p_t;
  - q is the student distribution: θ under p_s.
  - `jsd.clamp(max=clip)` acts **element-wise on f_k before the vocabulary sum**. clip = 0 disables it.
  - No `.float()` upcast appears before `log_softmax` [inference: likely bf16 logits].
- **OPSD README (read).**
  - The clip is described as "per-token point-wise KL clipping".
  - Motivation: style tokens "can exhibit 6–15× higher KL divergence than math-related tokens, and dominates the training signal".
  - "The loss can be negative due to positive KL summand being capped."
  - The non-thinking result tables use clip 1e-6 or 1e-7.
  - The tinker-style loss has no clip.
- **OP-Print configuration.** Table 3 uses clip 0.05 (0.06 for 8B), matching the official scripts.
- **OP-Print reported results (AUC).**

| Objective | Score | 1.7B | 4B | 8B |
|---|---|---|---|---|
| JSD, clip 0.05 (Table 1) | G | 0.579 | 0.580 | 0.556 |
| | G-Max | 0.594 | 0.611 | 0.587 |
| Tinker-style (Table 2) | G | 0.571 | 0.563 | 0.548 |
| | G-Max | 0.583 | 0.571 | 0.579 |

### 0.1 What the element-wise clip does (verified in Appendix A)

Let U be the set of unclamped vocabulary entries (f_k ≤ clip) and P_U = Σ_{k∈U} p_k. The gradient of Σ_k min(f_k, clip) with respect to the student logits is:

**g_j = q_j·P_U − p_j·1[j∈U]**

This is identical to v1.1's form 1[j∈U]·(q_j − p_j) − q_j(1 − P_U) + q_j·1[j∉U]; they agree to 6e-18 in the toy.

**Worked example.** The student is uniform over 20 tokens (q = 0.05 each). The teacher puts p_a = 0.9 on token a.

- a's summand is 2.60, far above 0.05, so a is clamped and P_U = 0.1.
- g_a = +0.005, and Σ_i q_i g_i = 0, so under gradient descent **log q_a decreases**. Over 20 SGD steps q_a falls from 0.050 to 0.048.
- With clip = 0, g_a = q_a − p_a = −0.85, and q_a rises to 0.80 over the same 20 steps.
- **The reviewer's algebra is correct.**

**Consequences [inference].**

- "Uncertain student, confident teacher" is exactly OP-Print's correction site. Under clip = 0.05 these entries get no direct upward push, and at high student entropy they can move the wrong way.
- Any membership imprint at the max-Δ chunks must therefore arise **indirectly**, through one of two routes:
  1. spillover from learning on unclamped entries;
  2. the adapter changing the p_t side. Δ_θ runs the teacher context through θ, while training used θ0 there.
- G-Max still beats G marginally in OP-Print, so one of these routes must be active.
- At OPSD's README clips of 1e-6 or 1e-7, almost every positive summand is clamped. Training is then driven mostly by negative summands, i.e. by entries where q > p.

## 1. Pre-H0 steps

### 1.0 First diagnostic (cheap; runs inside scoring)

1. For every chunk, log:
   - the argmax-Δ position;
   - the sampled token y_t at that position, and the teacher's top-1 token;
   - the fraction of Δ_t at that position that lies in entries clamped at 0.05.
2. Do this under θ (the attack) and under θ0 (CRN rollouts).
3. Classify tokens with a lexicon fixed before looking at the data **[design choice]**:
   - **style**: wait, hmm, but, so, let, actually, alternatively, think, okay, …;
   - **math**: digits, operators, LaTeX, variables;
   - **other**.
4. Report the class shares for in- vs out-samples, and the top-50 argmax tokens.

**Reading.** If the maxima fall mostly on style tokens, OP-Print's "correction sites" are a **style imprint**, not a math-correction imprint. That would reframe both OP-Print and this study.

### 1.1 Learning rate and f_rm

- **H0 runs at lr 5e-6** (the paper and official setting: LoRA r = 64, α = 128, 3 epochs), **whatever value f_rm takes.**
- **f_rm definition.** f_rm = 1 − mean_{x∈train} D_θ(x) / D_θ0(x), where D_θ = KL(p_θ0(·|p_t) ‖ p_θ(·|p_s)). This is the θ0-teacher residual, i.e. the quantity training actually minimizes, measured on CRN rollouts.
  - Also report the *clipped* residual Σ min(f, clip). Under clip, training minimizes that rather than the full KL, so f_rm can stay small even when training is working.
  - f_rm is reported as a diagnostic only.
- **Fallback.** Only if H0 fails, run the grid {2e-5, 5e-5, 1e-4} and select by f_rm ∈ [0.2, 0.5] with a pass@1 drop of at most 2 points **[design choice]**. Never select on attack metrics.
- **Clip-rate logging.** Log how often max-grad-norm = 0.1 clips, and the median gradient norm.

### 1.2 bf16 merge/unmerge and base drift

**Code (read).** At every vLLM sync the trainer runs `merge_adapter()`, then `load_weights`, then `unmerge_adapter()`, in bf16.

**[inference]** Two consequences:

- rollouts come from rounded merged weights;
- the base weights can drift. The base *is* the fixed teacher.

**Drift metric.**

- Per layer: ρ_ℓ = ‖W_base,ℓ(end) − W_base,ℓ(start)‖_F / ‖(α/r)·B_ℓA_ℓ‖_F. Report the median and the maximum over layers.
- Measuring L∞ is pointless: it is almost surely non-zero.
- **PEFT checkpoints save only adapters**, so the in-memory base must be dumped explicitly at the end of training.

**If ρ is non-negligible:**

- H1′ arm 3 is no longer "the training objective". Training used the drifted base, while scoring uses the public one.
- Score arm 3 additionally with the drifted base.
- Switch to vLLM native LoRA serving.

**Scoring** always uses the unmerged adapter, or a merge done in fp32.

### 1.3 Full-parameter control (optional; only if H0 fails)

Qwen3-0.6B full-parameter training needs about 12 GB with 8-bit Adam, or about 15–16 GB with fp32 Adam **[estimate; v1.1 §1.3 breakdown]**. 1.7B full-parameter does not fit: its fp32 AdamW states alone take ≈ 13.6 GB.

### 1.4 Numerically stable Δ (from v1.1, unchanged except as noted)

**Pipeline.**

- bf16 trunk with an **fp32 LM-head projection**.
- Barron et al. Eq. 9 in fp32. Per position:
  - δ = z_S − z_T
  - d_k = δ_k − Σ_i p_i δ_i
  - s_k = d_k − log1p(Σ_i p_i expm1(d_i))
  - Δ_t = −Σ_k p_k s_k
- **Direction.** z_T comes from the privileged context and z_S from the plain context, so this is KL(p‖q) = OP-Print's Δ. No reversal is needed.
- **Cross-check.** log1p(Σ_i p_i[expm1(d_i) − d_i]).
- 64-token position chunks, ≈ 39 MB per side.

**Toy verification (Appendix A, V = 50).**

- In fp64, both log-space forms match a 50-digit reference to ≤ 6e-10 relative error at every perturbation scale. The direct form p(log p − log q) is off by 2.4e-2 at σ = 1e-6.
- **In fp32, the direct form has relative error 0.63 at logit-perturbation σ = 0.01, versus 3.4e-5 for the Barron form.**
- The direct form is therefore unusable for small Δ even in fp32. The log-space form is required, not optional.
- The series branch for |d| < 1e-3 proposed in v1.1 is unnecessary in fp64. It is untested in fp32.
- Caveat: the toy uses V = 50, not 152k.

**Barron et al.'s magnitudes** (∼1e-3 in bf16, ∼1e-8 in fp32) are residual *gradients* in a toy distillation, not KL floors.

**Qwen3-1.7B facts (verified from the HF config).**

- hidden 2048, intermediate 6144, 28 layers, 16 query heads / 8 KV heads, head_dim 128, vocabulary 151,936, tied embeddings.
- KV cache = 114,688 B/token in bf16.

### 1.5 Noise floor and calibrated ε

**Reviewer's prior [estimate].**

- With q ∝ p·e^δ, KL ≈ ½·Var_p(δ).
- bf16 logit spacing of 0.0625 gives a per-side rounding std of ≈ 0.018 (0.0625/√12).
- That puts the floor at roughly 3e-4 nats/token or less, and only at high-entropy positions.
- The fp32 LM head removes most of it. The floor is probably small next to member Δ̄; G-Max's max-of-noise is the exception.

**Tests.**

- **N0 / N0′ (arithmetic).** Identical logits, then logits with small perturbations. Compare the standard routine and the fixed routine against fp64.
- **N1 (lower bound).** True Δ = 0: the plain context is fed on both sides, with masked left-padding to the p_t length.
  - Masked padding contributes exact zeros, so **N1 misses the accumulation error from attending over a real 8k-token r**. It is therefore only a lower bound.
- **N2 (primary).**
  - Run a **fully fp32 forward pass** (θ in fp32, adapter merged in fp32) on about 100 samples (50 in, 50 out) from the pilot model, as the reference.
  - 1.7B in fp32 is ≈ 6.8 GB of weights, plus fp32 KV for 8k tokens (≈ 1.8 GB) **[estimate]**. It fits in 16 GB.
  - Compute the production pipeline's Δ on the same rollouts. Report the per-sample relative error |Δ̄_prod − Δ̄_ref| / Δ̄_ref **separately for in- and out-samples**, for both the chunk mean (G) and the chunk max (G-Max).
  - Report the log-ratio compression v_prod − v_ref directly. N1 cannot measure this, because its true Δ is zero.
- **ε (calibrated from N2) [design choice].** ε_G and ε_GMax are each the 99th percentile of the absolute N2 error of the corresponding aggregate, floored at the N1 value. Winsorization at the 1/99 percentiles is kept.
- **Gate.** If the median relative N2 error for in-samples exceeds 5% **[design choice]**, fix the pipeline before running H0. This is also H5 trigger (ii).

## 2. Hypotheses

**Minimal paper.** H0 → H1/H1′ → H3, with H3 serving as the clip-mechanism test. H2, H4 and H5 are conditional.

### H0: replication (lr 5e-6, LoRA r = 64, clip 0.05)

- **Primary test.** G-Max AUC minus the AUC of the best baseline, by paired bootstrap. The best baseline is selected on a disjoint half of the samples.
- **G vs G-Max.** Paired DeLong per model, plus the spread across models.
- **Scoring diagnostic.** Re-score with the standard bf16 routine vs the fixed routine. A gap of |ΔAUC| ≥ 0.02 is H5 trigger (iii).
- **Kill.** The bootstrap CI includes 0. Then run the fallbacks in §1.1 and §1.3.

### H1 (main line): does θ0 concentration predict leakage?

**Primary c (pre-registered, only one).**

- c(x) = the top-10% chunk share of Δ_θ0: the sum of the largest ⌈0.1·n⌉ chunk Δs divided by the total, averaged over K θ0 rollouts.
- Chosen over max/mean because it is less sensitive to noise maxima and to the number of chunks.
- max/mean is reported as secondary only.

**Mechanistic competitor (new).**

- m_U(x) = mean_t Σ_{k∈U_t} f_tk under θ0, using the clip of the target run. This is the trainable mass: θ0's Δ in unclamped entries.
- At clip = 0 it reduces to Δ̄_θ0. The share version m_U/Δ̄_θ0 is secondary.

**Outcome.** v(x) = log((Δ̄_out + ε) / (Δ̄_in + ε)), with ε from N2, multiplicative out-calibration and winsorization. The difference is a secondary outcome.

**Covariates d.** θ0 pass@k, Δ̄_θ0, log T, θ0 loss, and θ0 last-layer leverage (transferring 2602.15919 to LLMs is [unverified]).

**Pre-registered predictions (sparse-imprint story).**

- **clip = 0:** c has a positive incremental effect over d. Since m_U = Δ̄_θ0 ⊂ d at clip = 0, "c beats m_U" here means c adds to d.
- **clip = 0.05:** m_U's incremental R² over d exceeds c's, i.e. trainable mass beats raw concentration. The reason is that concentrated mass sits in clamped entries and is not trained directly.
- If c beats m_U at clip = 0.05 as well, the imprint is not mediated by direct training at the high-Δ entries. That is consistent with the indirect routes in §0.1.

**Placebo null (corrected).**

- Under the complement design, the exact null is a **within-pair sign flip**: for each sample, swap in/out independently within each of the 4 (split, complement) pairs. That gives 16 configurations per sample, drawn independently across samples, with 1000 joint draws.
- "Any 4 of 8" relabelling would mix between-pair model variance into the null, and the true v never contains that variance.
- Requirements:
  - c's coefficient must exceed the 95th percentile of the placebo distribution;
  - c must not predict v_placebo.

**Kill.** c adds nothing over θ0 leverage + loss (+ the rest of d).

### H1′: reference-calibrated attacks (CRN)

1. S_θ − S_θ0.
2. ρ(x) = ΣΔ_θ / ΣΔ_θ0.
3. The θ0-teacher arm D_θ = KL(p_θ0(·|p_t) ‖ p_θ(·|p_s)), which is the training objective (subject to the drift caveat in §1.2), plus a clipped variant.

**New: the difference Δ_θ − D_θ.** This isolates how much of the imprint comes from the adapter changing the **p_t side**, which is indirect route 2 in §0.1.

### H3: clip-mechanism test (primary outcome changed)

**Doses.** clip ∈ {0, 0.2, 0.05 (paper), 0.01}. OPSD's README setting of 1e-6 is an optional extra dose.

**Primary outcome.** gap(clip) = AUC(G-Max) − AUC(G), as a function of clip strength, **tested two-sided**:

- paired bootstrap of gap(0) − gap(0.05);
- Spearman trend across doses.

**Predictions.**

- The sparse-imprint story predicts that **the gap widens as clip → 0**, because the strong sites are then trained directly.
- v1.1's prediction ("the G-Max advantage falls monotonically with clip") does not follow and is withdrawn.

**Existing weak evidence.** OP-Print's tinker-style objective is likely unclipped [inference from the OPSD README]. Its gaps are +0.012 / +0.008 / +0.031 (1.7B / 4B / 8B), versus +0.015 / +0.031 / +0.031 under clip 0.05. That is no wider, but the objective differs, each is a single run, and no CIs are given. **Inconclusive.**

**Sign-agnostic attacks (required).**

- A one-sided G-Max falling to 0.5 could be a **sign flip**, not a working defense.
- **|score − median_out|** after out-calibration.
- A **learned aggregator on signed features**: the sorted chunk vector of Δ_θ − Δ_θ0 under CRN, fed to logistic regression trained on 2 models and tested on 6.
- Report AUC below 0.5 with its CI. Do not fold it into max(AUC, 1 − AUC).

**clip = 0.01.** Unclamped and negative summands dominate the loss. **Matched utility may be unreachable**, and the high-Δ imprint can flip sign. At this dose, report the utility–AUC frontier rather than a matched comparison.

**Scale.**

- 8 models at clip ∈ {0, 0.05}. This also supports H1's two-clip predictions.
- 2 models (one split + complement) at clip ∈ {0.2, 0.01}, for AUC-level gaps only.
- DuoLearn is a qualitative comparator.

### Conditional hypotheses

**H2: DP comparison.** Conditional on a feasibility pilot.

- The adapter has **69,730,304 parameters at r = 64** (verified from the HF config: all 7 linear projections × 28 layers) and 8,716,288 at r = 8.
- DP-SGD noise has per-coordinate std σC/B, and B ≤ 500 records.
- **Utility-only pilot first:** ε = 8, δ = 1e-5, r = 8, full batch.
  - If pass@1 does not move (less than half of the non-private gain **[design choice]**), H2 needs a separate larger-N setup or a later paper.
  - Otherwise, run the v1.1 design: arm A is DP-SGD; arm B is DP-OPD-004 / DP-IER-E, **pending Alex's confirmation that its privacy unit is (q, r)**, else B′.

**H4: top-k teacher.** Conditional on H3's outcome: `--top_k_loss` ∈ {0, 20, 100}.

**H5: training-side numerical ablation (standard vs fixed loss × merge vs native LoRA).** Run only if at least one trigger trips:

- (i) median drift ρ_ℓ > 0.01, or maximum ρ_ℓ > 0.05 **[design choice]**;
- (ii) N2 median relative error for in-samples > 5% of Δ̄_in;
- (iii) |ΔAUC| ≥ 0.02 between standard and fixed scoring in H0.

The fixed loss uses the verified clip-aware gradient g_j = q_j P_U − p_j 1[j∈U], computed with Barron's u_j = q_j − p_j. Softmax Collapse is a secondary concern (high-confidence, low-Δ tokens).

## 3. Setup and compute

**Data and splits.**

- Pool: OpenThoughts math, N+ = 1000.
- 4 random splits + complements, so each sample is exactly 4 in / 4 out.
- Split-half reliability is computed over pair-halves (each sample 2 in / 2 out). Gate: raw r ≥ 0.3, which is ≈ 0.46 after Spearman–Brown.

**FPC.** The complement design fixes N/N+ = 0.5, the worst case in 2605.25819.

- For H1 (relative ranking), we treat FPC as roughly a common factor across samples. **[assumption]** This is untested: per-sample FPC bias could correlate with c or with length.
- FPC correction is applied to every absolute TPR ("Decomposed FPC" details only partly read **[unverified]**).
- Pooled ROCs are out-calibrated.

**Scoring.** K = 8 full rollouts, with a K = 32 check on 200 samples. "First 64 tokens" is an ablation only.

**Compute [estimate; per-unit figures from v1.1, to be calibrated on the pilot model].** One model takes about 1–2 GPU·h to train and about 1.25 h to score (1000 samples × K = 8).

| Item | GPU·h |
|---|---|
| Pre-H0: drift dump, N0/N1, N2 (100 samples in fp32), §1.0 diagnostic (inside scoring) | 3–5 |
| H0: 1 model at 5e-6 + scoring + standard-vs-fixed re-score | 3–5 |
| H1/H1′ at clip 0.05: 7 further models + scoring + θ0-teacher arm | 20–30 |
| H3: 8 models at clip 0 (also used for H1) + 4 models at clip {0.2, 0.01} + scoring | 30–45 |
| **Minimal paper total** | **≈ 56–85 GPU·h ≈ 2.3–3.5 GPU-days** |

- This is above the reviewer's 1.5–2 days because H1's two-clip prediction needs 8 models at clip 0.
- If that is too much, use 4 models (2 pairs) at clip 0, with lower reliability.
- Conditional extras: H5 8–12 GPU·h; H2 pilot ≈ 3–5 GPU·h; H4 ≈ 6–10 GPU·h.

**LT-IQR-style ceiling.** In-sample loss traces plus CRN checkpoint Δ. Only 3 free points per sample.

## 4. Prior work (verification status, unchanged from v1.1 except Barron)

- **2602.15919 (Dorseuil, Atif, Cappé).** Single trained model, loss + leverage. Not ex ante. Classification and regression only. ✓
- **LT-IQR (USENIX Sec '25; arXiv 2411.05743).** Loss-trace IQR, image classification. ✓
- **2605.25819 (Jälkö et al.).** FPC inflation, per-sample FPR miscalibration, and fixes. ✓
- **2505.18773 (Hayes et al.).** Per-sample instability; length correlation. ✓
- **2609.36331 (Rita, Zhang, Tramèr).** Per-example calibration is the common bottleneck; images only. ✓
- **2505.11837.** The earlier "tension" claim is withdrawn: it concerns a different quantity.
- **DuoLearn (2502.19726).**
- **2610.31642 (Barron et al.).** Softmax Cancellation and Collapse. All experiments are toy or grokking models, and the authors themselves call the regimes "uncommon in standard large-scale training". Transfer is [unverified]. The fp32 KL-value result in Appendix A is ours, not theirs.

## 5. Execution order and stopping rules

1. **Bring up OPSD.** Then:
   - drift dump;
   - N0/N1/N2 and ε;
   - train the pilot model at 5e-6;
   - run the §1.0 argmax diagnostic.
2. **H0.** Kill if the CI includes 0. Otherwise:
   - check H5 triggers (ii) and (iii);
   - only if H0 fails, run the LR-grid and 0.6B fallbacks.
3. **Remaining 7 models at clip 0.05.** Apply the reliability gate. If raw r < 0.3, switch to risk stratification.
4. **H1, H1′ and the LT-IQR ceiling.**
5. **H3:** 8 models at clip 0, then the 0.2 / 0.01 cells. Then H1's two-clip comparison.
6. **Conditional work:**
   - H5 if a trigger tripped;
   - the H2 pilot, then H2 only if feasible and the privacy unit is confirmed;
   - H4.

## 6. Risks

- **Risk stratification may be the main deliverable.** Bin samples into quintiles of c and test monotonicity of the pooled TPR@1%FPR.
- **Style-imprint reframing.** §1.0 may show that OP-Print's signal is stylistic.
- **Clip-induced sign effects** at small clips.
- **Numerical compression** of the log-ratio. The fp32 direct KL is shown unreliable (Appendix A), so the log-space form is mandatory.
- **H2 likely infeasible** at N = 500.
- **Weak base effect:** AUC ≈ 0.6.
- **Unverified FPC common-factor assumption.**
- **The LoRA inference**, now strongly supported.

## 7. Disagreements with the v1 critique: resolution

1. **5e-6 for LoRA.** *Conceded by the reviewer.* The official scripts use LoRA r = 64 at 5e-6, and the earlier premise came from v0's erroneous full-parameter claim.
2. **K / 64-token truncation.** *Conceded by the reviewer.*
3. **Complements vs FPC.** The reviewer acknowledged giving conflicting advice and found the resolution reasonable. "FPC is roughly a common factor" is now labelled **[assumption]**.
4. **z-scoring vs the ratio outcome.** *Reviewer agrees.* Multiplicative out-calibration is kept.
5. **LT-IQR "free".** *Reviewer agrees* it is not: 3 epochs give 3 points per sample.
6. **0.6B full-parameter control optional.** Not contested.

**v1.1 review.** I found no claims that were wrong. The clip algebra and the worked example are confirmed exactly (Appendix A). The reviewer's remark that "(q_j − p_j)" was missing from the v1.1 PDF is **correct, and the cause is ours, not text extraction**. The v1.1 markdown has the term, but the renderer parsed `[j∈U](q_j − p_j)` as a Markdown link and dropped it. v1.2 escapes such patterns. Both forms are verified equal.

## Appendix A: fp64 toy verification (`scripts/toy_clip_grad.py`)

**Method.**

- Reproduces OPSD's `generalized_jsd_loss` at β = 0 for a single position in numpy fp64: f = p·(log p − log_softmax(z_S)), then element-wise `min(f, clip)` (clip = 0 means off), then a vocabulary sum.
- Compares the analytic gradient against central finite differences, used as the autograd reference.
- Checks KL implementations against a 50-digit `decimal` reference.

**Output (verbatim).**

```
summands p(log p - log q): [ 2.601335 -0.011849]
clip=0.0: a clamped=False P_U=1.0000 g_a=-0.850000 g_a-sum(q g)=-0.850000 dlog q_a/step=+0.850000 max|analytic-FD|=5.70e-10 max|v1.1-analytic|=0.00e+00
   after 20 SGD steps (lr 0.5): q_a 0.0500 -> 0.7969
clip=0.05: a clamped=True P_U=0.1000 g_a=+0.005000 g_a-sum(q g)=+0.005000 dlog q_a/step=-0.005000 max|analytic-FD|=4.51e-11 max|v1.1-analytic|=6.07e-18
   after 20 SGD steps (lr 0.5): q_a 0.0500 -> 0.0476
random (200 cases, V=50): max|analytic-FD grad| = 9.61e-10
max relative KL error vs 50-digit reference, by logit-perturbation sigma:
  sigma=1e-06: direct p(logp-logq)=2.4e-02; -sum p s (Barron)=5.8e-10; log1p(sum p(expm1 d-d))=1.4e-10; fp32 direct=1.2e+07; fp32 Barron=3.7e-01
  sigma=0.0001: direct p(logp-logq)=5.4e-06; -sum p s (Barron)=5.0e-12; log1p(sum p(expm1 d-d))=1.2e-12; fp32 direct=1.9e+03; fp32 Barron=2.7e-03
  sigma=0.01: direct p(logp-logq)=1.6e-09; -sum p s (Barron)=5.5e-14; log1p(sum p(expm1 d-d))=8.9e-15; fp32 direct=6.3e-01; fp32 Barron=3.4e-05
  sigma=0.5: direct p(logp-logq)=1.1e-13; -sum p s (Barron)=2.1e-15; log1p(sum p(expm1 d-d))=1.4e-15; fp32 direct=8.4e-05; fp32 Barron=1.1e-06
```

**Reading.**

1. With clip = 0.05, token a is clamped, P_U = 0.1, g_a = +0.005, Σ q g = 0, so q_a falls. With clip = 0, g_a = −0.85 and q_a rises. **The reviewer's algebra and example are confirmed.**
2. The analytic gradient q_j P_U − p_j 1[j∈U] matches finite differences to ≤ 1e-9 on 200 random cases at both clips. v1.1's clip-aware form is identical to within 6e-18.
3. In fp64, both log-space KL forms (Barron −Σ p s, and log1p(Σ p(expm1 d − d))) are accurate to ≤ 6e-10 relative error at every σ. The direct fp64 form degrades to 2.4e-2 at σ = 1e-6.
4. **In fp32, the direct form is unusable for small perturbations** (relative error 0.63 at σ = 0.01), while the Barron form gives 3.4e-5.
5. **Limits.** V = 50; no bf16 (numpy lacks it); the log1p form was not run in fp32. σ is the std of the logit perturbation, not a calibrated Δ scale for Qwen3.

## References

- **[OP-Print]** Q. Li, X. Wang. arXiv 2610.31616 (2026).
- **[OPSD]** github.com/siyan-zhao/OPSD (code, scripts and README; read 2026-10-26).
- **[Barron+26]** J. T. Barron, D. Gnanapragasam, M. Tschannen. "A Fix for Softmax and Cross-Entropy". arXiv 2610.31642.
- **[DLC26]** Dorseuil, Atif, Cappé. arXiv 2602.15919.
- **[LT-IQR]** Pollock, Shilov, Dodd, de Montjoye. USENIX Security '25; arXiv 2411.05743.
- **[JPRH26]** Jälkö, Pradhan, Räisä, Honkela. arXiv 2605.25819.
- **[Hayes+25]** arXiv 2505.18773 (NeurIPS 2025).
- **[RZT26]** Rita, Zhang, Tramèr. arXiv 2609.36331 (AISec '26).
- **[KD-MIA]** arXiv 2505.11837.
- **[DuoLearn]** arXiv 2502.19726.
- **[Steinke+23]** Steinke, Nasr, Jagielski. NeurIPS 2023 (one-run auditing; not re-read).
- **[Qwen3-1.7B config]** huggingface.co/Qwen/Qwen3-1.7B/config.json (read 2026-10-26).
