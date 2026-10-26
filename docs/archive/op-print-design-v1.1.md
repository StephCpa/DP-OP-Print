# Ex-ante prediction of membership leakage in on-policy self-distillation (OPSD): experimental design v1.1 (2026-10-26)

**Basis of this version**

- **Read in full:**
    - OP-Print (arXiv 2610.31616).
    - The official OPSD code (`siyan-zhao/OPSD`): `opsd_trainer.py`, `opsd_train.py`, `scripts/run_opsd_{1b,4b,8b}.sh` and the README.
    - arXiv 2505.11837.
    - Barron, Gnanapragasam and Tschannen, "A Fix for Softmax and Cross-Entropy" (arXiv 2610.31642): §§1, 3, 7 and 8 read closely.
- **Read at abstract and key-passage level:** 2602.15919, 2605.25819, 2505.18773, 2609.36331, and LT-IQR (arXiv 2411.05743).
- **Markers:**
    - **[unverified]**, **[inference]** and **[estimate]** mark statements that do not come from reading a primary source.
    - **[design choice]** marks pre-registered thresholds that are ours, not taken from the literature.
- v0 (`op-print-design.md`, Chinese) and v1 (`op-print-design-v1.md`, Chinese) are left unchanged.

## Changelog

### v1 → v1.1

1. **Language.** The document is now fully in English. Content is otherwise unchanged except for the items below.
2. **Numerically stable Δ (new §1.4).** Per-token KL is now computed in log space following Barron et al. (Eq. 9), in fp32, with an fp32 LM-head projection.
    - The KL direction was checked against OP-Print's Δ and against OPSD's β = 0 loss. Both are KL(teacher-context ‖ student-context), so no reversal is needed.
3. **ε for the log-ratio outcome.** The ad-hoc ε (1st percentile of Δ̄) is replaced by an ε calibrated to the *measured* noise floor of the exact scoring pipeline, using true-Δ = 0 runs. This is a new pre-H0 sanity check (§1.5). Winsorization is kept.
4. **New training-side ablation (H5).** The standard OPSD loss is compared with a fixed log-space loss.
    - This is crossed with the weight-synchronization path (bf16 merge/unmerge vs native LoRA serving), which separates the two drift channels.
    - Softmax Collapse is noted as a secondary concern.
5. **New risk.** H0/H1 can fail for purely numerical reasons.
6. A references list has been added (§9).

### v0 → v1 (carried over)

1. **Correction of a factual error in v0.** v0 stated that OP-Print used full-parameter training. That was my inference, and it is very likely wrong.
    - OP-Print Appendix A states that all runs use the fixed teacher. In the OPSD code, `fixed_teacher=True` raises an error unless PEFT is enabled.
    - OP-Print Table 3 matches the official OPSD scripts item by item:
     - β = 0, λ = 1, T = 1.1, top-p 0.95, top-k 20;
     - max grad norm 0.1, lr 5e-6;
     - token clip 0.05, or 0.06 for 8B only.
    - Those scripts use **LoRA (r = 64, α = 128, all linear projections) at lr 5e-6**.
    - OP-Print is therefore very likely LoRA r = 64 at 5e-6 for 3 epochs (the scripts use 30 epochs). This is **[strong inference]**: the paper never mentions LoRA.
2. **Pre-H0 fixes:**
    - LR is selected by the fraction of θ0's Δ removed on the training set, never by attack metrics.
    - The clip rate of the max-grad-norm is logged.
    - bf16 merge/unmerge is handled explicitly.
    - The full-parameter control moves to Qwen3-0.6B, with a memory budget.
3. **H1 rewritten:**
    - log-ratio primary outcome with an ε/winsorization rule;
    - placebo-membership permutation null;
    - log T and θ0 leverage added as covariates;
    - 4 splits + complements;
    - explicit reliability threshold;
    - out-calibration before pooling;
    - FPC bias handling;
    - kill criterion "c adds nothing over θ0 leverage + loss".
4. **H0 statistics.** Paired bootstrap on the AUC difference, and paired DeLong for G vs G-Max.
5. **H2 made comparative.** Matched ε and matched utility, with δ = 1e-5. DP-OPD-004 / DP-IER-E is a conditional arm.
6. **H3** is a dose axis on the existing `jsd_token_clip`.
7. **Minor changes:**
    - K disambiguated.
    - KL computed in chunks.
    - H1′ uses common random numbers (CRN) plus a θ0-teacher arm.
    - The "tension" claim about 2505.11837 is withdrawn.
    - An LT-IQR-style ceiling is added.
    - Risk stratification is promoted.
8. **New code findings:**
    - OPSD ships `--top_k_loss`, so H4 needs no new code.
    - The token clip is element-wise over vocabulary entries (§0).

## 0. Facts pinned from the paper and the code

- **Signal.** Δ_t = KL(p_θ(·|p_t(q,r), y<t) ‖ p_θ(·|p_s(q), y<t)).
    - Both sides use the attacked model θ.
    - Rollouts are drawn as y ~ p_θ(·|p_s(q)).
    - p_t is the privileged (teacher) context containing the reference solution r. p_s is the plain (student) context.
- **Gray-box scores.**
    - G = −(1/K)Σ_k (1/T_k)Σ_t Δ_t.
    - G-Max = −(1/K)Σ_k max_t Δ_t.
    - Δ is computed over 64-token chunks.
- **Meaning of K.** K is the number of full rollouts (default 32; Fig. 5a ablates the rollout count and the chunk size).
    - Chunks are 64-token segments *inside* each full rollout (up to 1024 tokens).
    - Algorithm 1 describes only the black-box attack.
- **Training loss (OPSD, β = 0).** `F.kl_div(student_logp, teacher_logp, log_target=True)` is the summand p_k(log p_k − log q_k), i.e. KL(teacher ‖ student). This is the same direction as Δ.
    - The teacher is the adapter-disabled base plus the privileged context.
    - No `.float()` upcast occurs before `log_softmax`, so under `--torch_dtype bfloat16` the loss is very likely computed on bf16 logits. **[inference]** This depends on the `transformers` version.
- **`jsd_token_clip`.**
    - `jsd` has shape `[B,T,V]`. `clamp(max=clip)` is applied element-wise *before* summing over the vocabulary; the sum is then divided by the token count.
    - clip = 0 disables it.
    - The README confirms it is point-wise rather than a cap on the vocabulary-summed loss, and that the loss can be negative.
- **Reported metrics** are AUC, Acc and TPR@5%FPR only.
    - G-Max AUC: 0.594 (1.7B), 0.611 (4B), 0.587 (8B).

## 1. Pre-H0 fixes and sanity checks

### 1.1 Learning-rate selection

- **Removal metric.** f_rm = 1 − mean_{x∈train} Δ̄_θ(x) / Δ̄_θ0(x), computed on fixed CRN rollouts (§3).
- **Grid.** LR ∈ {5e-6 (official anchor), 2e-5, 5e-5, 1e-4}, with r = 64 and 3 epochs.
- **Selection rule [design choice].** Take the smallest LR such that:
    - f_rm ∈ [0.2, 0.5], and
    - θ0's pass@1 drops by at most 2 points.
- **No attack metric is consulted at any point.**
- **Clip logging.** Log the fraction of steps where the pre-clip norm exceeds 0.1, and the median norm.
    - If clipping is active close to 100% of the time, training is effectively normalized-gradient descent with LR as the only step-size knob. Report this.

### 1.2 bf16 merge/unmerge

**What the code does (read):** at each vLLM sync, it calls `merge_adapter()` → `load_weights` → `unmerge_adapter()`.

**[inference]** This has two consequences:

1. Rollouts are generated by bf16-rounded merged weights, so they differ slightly from the training policy.
2. The in-place bf16 add/subtract is not exactly invertible, so base weights can accumulate drift. Because the fixed teacher *is* the adapter-disabled base, **the teacher itself can drift**.

**Mitigations:**

- Score with the adapter *unmerged* (HF forward pass, bf16 base + adapter), or merge in fp32.
- Hash and compute the L∞ distance of the base weights before and after training.
- If drift is non-zero, serve the adapter through vLLM's native LoRA (`enable_lora`) without merging.
- Report the rollout-policy vs training-policy mismatch.

### 1.3 Full-parameter control

- **1.7B full-parameter training does not fit in 16 GB.** AdamW's fp32 m and v alone take 1.7B × 8 B ≈ 13.6 GB.
- **Qwen3-0.6B full-parameter instead [estimate]:**

| Component | Memory |
|---|---|
| bf16 weights | 1.2 GB |
| bf16 gradients | 1.2 GB |
| fp32 AdamW m, v | 4.8 GB (8-bit Adam ≈ 1.2 GB) |
| Frozen teacher copy (OPSD's fixed teacher needs PEFT, so full-parameter training needs an explicit copy) | 1.2 GB |
| Activations (checkpointed) + chunked logits | 2–3 GB |
| vLLM at a 0.3 share | ≈ 4.8 GB |
| **Total** | **fp32 Adam ≈ 15–16 GB (too tight); 8-bit Adam ≈ 12 GB (feasible). To be measured.** |

- **Position.** OP-Print's target is very likely LoRA, so this control is optional. It is used only if H0 fails, to test whether LoRA capacity is the cause.

### 1.4 Numerically stable Δ computation (new; Barron et al., 2610.31642)

**What the paper says (verified wording).**

- *Softmax Cancellation* "occurs when student logits are close to teacher logits, and produces non-zero gradients even when the student and teacher logits are equal" (abstract).
- Standard implementations compute p = softmax(z_T) and q = softmax(z_S) separately and subtract them. This loses precision as z_S → z_T, and the gradient "plateau[s] at a non-zero noise floor" (Fig. 1b).
- The quoted magnitudes are **residual gradients at z_S = z_T in a teacher-initialized toy distillation** (a + b mod 53, T = 4, Adam ε = 1e-8; §7, Fig. 7): "∼10⁻¹⁷" in fp64, "∼10⁻⁸" in fp32 (a brief KL spike) and "∼10⁻³" in bf16 (drives the student away from the teacher).
- **These are gradient residuals, not KL-value floors.** The paper reports no noise floor for the KL *value*. Carrying the concern over to the scored Δ is **our inference**, which is why the floor is measured directly (§1.5).

**Why it matters here [inference].**

- For members, training drives p_θ(·|p_s) toward p_θ(·|p_t). At most positions both contexts therefore sit in the near-equal regime, where any additive numerical floor is largest *relative* to the true Δ.
- A floor inflates Δ̄_in proportionally more than Δ̄_out, which compresses log(Δ̄_out/Δ̄_in) toward 0. **H0/H1 could therefore fail for numerical rather than substantive reasons.**
- A second, probably larger source is that **bf16 logits are quantized**.
    - bf16 has 8 significand bits, so the spacing is 2⁻⁴ = 0.0625 for |z| ∈ [8, 16) and 0.125 for |z| ∈ [16, 32).
    - Logit differences between the two contexts are therefore resolved only to about 0.06–0.1 at typical magnitudes. **[estimate]** Qwen3's logit magnitudes are unverified.
    - The log-space formula cannot recover information lost before it.

**Procedure:**

1. Compute the final hidden states in bf16, then apply **the LM-head projection in fp32** (upcast the hidden state and the weight). Do the whole KL computation in fp32.
2. Per position, let z_T be the logits under the privileged context p_t and z_S the logits under the plain context p_s. Then:
    - p = softmax(z_T);
    - δ = z_S − z_T;
    - d_k = δ_k − Σ_i p_i δ_i;
    - s_k = d_k − log1p(Σ_i p_i expm1(d_i)) = log(q_k/p_k);
    - **Δ_t = KL(p‖q) = −Σ_k p_k s_k.**
3. **Direction check.** OP-Print defines Δ = KL(p_θ(·|p_t) ‖ p_θ(·|p_s)) = KL(p‖q) with p from the teacher context and q from the student context. This is exactly Barron's convention (p = softmax(z_T), s = log(q/p)). **No reversal is needed.**
    - The same holds for OPSD's β = 0 training loss and for the θ0-teacher arm of H1′ (where z_T comes from θ0 under p_t).
    - If a reverse-KL variant (β = 1) is ever scored, swap the roles: δ′ = z_T − z_S, centred under q, giving KL(q‖p) = −Σ_k q_k s′_k.
4. **Equivalent non-negative form (our algebra; to be checked against fp64 in §1.5).** Since Σ_i p_i d_i = 0, the definition reduces to KL(p‖q) = log1p(Σ_i p_i expm1(d_i)) = log1p(Σ_i p_i [expm1(d_i) − d_i]).
    - Every term in the last expression is ≥ 0, so there is no sign cancellation.
    - Use a series for |d_i| < 1e-3.
    - Keep −Σ p s as the primary implementation and this form as a cross-check.
5. **Chunking.** Process 64-token position chunks: 64 × 151,936 × 4 B ≈ 39 MB per side, vs ≈ 0.62 GB per side for a full 1024-token sequence.
    - Two reductions over the vocabulary are needed (Σ p δ, then Σ p expm1(d)).
    - Vocabulary chunking with a two-pass reduction is available if memory is tight.
    - Privileged-prompt KV caches are reused across the K rollouts (≈ 115 KB/token for Qwen3-1.7B, i.e. ≈ 0.9 GB for an 8k-token r **[estimate]**).

### 1.5 Noise-floor sanity check and calibrated ε (new; replaces the 1st-percentile ε)

This runs before H0, on one pilot model.

**Three tests, all with true Δ = 0:**

- **N0 (arithmetic).** Feed identical logit tensors (z_S = z_T) to both the standard (bf16 `F.kl_div`, as in OPSD) and the fixed routine.
    - Expected: the fixed routine gives exactly 0.
    - The standard routine may also give 0 when both sides take the same code path, so N0 alone is not informative.
- **N0′ (accuracy).** Use z_S = z_T + η with η ~ N(0, σ²) for σ ∈ {1e-6, …, 1e-1}. Compare each routine and precision against an fp64 reference KL, and record the absolute error vs σ.
- **N1 (end-to-end pipeline floor).** Run the *exact* scoring pipeline (same batching, chunking, KV reuse, fp32 head) with the privileged context replaced by the plain context, left-padded with attention-masked tokens to the real p_t length and with adjusted `position_ids`.
    - The true Δ is mathematically 0, but both forward passes follow the real, length-dependent numerical paths.
    - Record the per-sample aggregates exactly as the attack computes them:
    - the mean over chunks (for G);
    - **the max over chunks (for G-Max). The max of noise is biased upward, so it needs its own floor.**

**Calibrated ε [design choice]:**

- ε_G = 99th percentile of the N1 per-sample mean.
- ε_GMax = 99th percentile of the N1 per-sample max.
- Floor the ε at these values; never set them below. Winsorization at the 1/99 percentiles of v is kept.

**Gate [design choice]:**

- If the N1 floor is not at least 10× below the median Δ̄_in of the pilot model, the numerics are suspect. Fix the pipeline (fp32 head, deterministic kernels, identical batch shapes) before H0.
- Report the floor for both the standard and the fixed pipeline. Their difference is itself a result.

## 2. Hypotheses

### H0: replication

**Setup.** Qwen3-1.7B, LoRA r = 64, the LR from §1.1, the fixed Δ pipeline from §1.4.

**Tests:**

- G-Max AUC minus the best baseline's AUC, by **paired bootstrap**: resample samples, 2000 replicates. The "best baseline" is selected on a disjoint half of the samples, to avoid selection bias.
- G vs G-Max by **paired DeLong** on each model, plus the spread across the 8 models. OP-Print's "G-Max is better" is currently neither significant nor marginal.

**Diagnostic.** Also report H0 with the standard bf16 scoring routine. If AUC rises materially with the fixed routine, numerical noise was masking signal.

### H1 (main line): does θ0 concentration predict leakage?

**Hypothesis.** A concentration statistic c(x) computed under θ0 predicts the per-sample leakage v(x). For c we use the top-10% chunk share of Δ_θ0 and max/mean.

**Primary outcome.** v(x) = log((Δ̄_out + ε) / (Δ̄_in + ε)), with ε calibrated in §1.5. The secondary outcome is Δ̄_out − Δ̄_in. The G-Max version is a secondary analysis.

**Calibration before pooling:**

- Divide each model's Δ̄ by that model's median Δ̄ over its out-samples (multiplicative out-calibration). This keeps values positive, so the ratio stays meaningful.
- Per-model z-scores are used only for the difference outcome and for pooled ROCs. z-scoring would break the ratio.

**Covariates d:**

- θ0 pass@k;
- θ0's Δ̄;
- log T (mean θ0 rollout length);
- θ0's loss;
- θ0 last-layer leverage. This is the 2602.15919 method transplanted to θ0 hidden states; its transfer is **[unverified]**.

**Placebo-membership null:**

- For each sample, randomly relabel 4 of its 8 models as "in" (counts fixed), recompute v_placebo and refit. Repeat 200 times.
- c's coefficient on v must exceed the 95th percentile of the placebo distribution.
- c must not predict v_placebo.

**Kill.** Given d, c's partial-coefficient CI includes 0, or its incremental R² does not exceed the placebo. In words: **c adds nothing over θ0 leverage + loss.**

### H1′: reference-calibrated attacks

Use **CRN**: Δ_θ and Δ_θ0 are computed on the same rollouts and the same chunks, with the §1.4 pipeline. Three scores:

1. S_θ − S_θ0.
2. The remaining correction ratio ρ(x) = ΣΔ_θ / ΣΔ_θ0, the per-sample analogue of ρ_S.
3. **The θ0-teacher arm**, KL(p_θ0(·|p_t) ‖ p_θ(·|p_s)), which is the training objective itself. A variant applies the clip.

**Cost and novelty:**

- All three scores need only the public base model and r.
- None appears in OP-Print.
- The extra cost is one θ0 prefill.

### H2: comparative DP

**Comparison.** Two mechanisms at matched ε (δ = 1e-5 ≪ 1/N = 2e-3) and matched utility (held-out pass@1 within 1 point). The outcome is the OP-Print advantage, measured as TPR@1%FPR and AUC − 0.5 on the out-calibrated pooled ROC.

**Arm A: DP-SGD.**

- Per-record clipping on (q, r): all rollout gradients of q are summed into one before clipping.
- Poisson subsampling, PRV accountant.

**Arm B: DP-OPD-004 / DP-IER-E.**

- **Alex must confirm that its privacy unit is the distillation record (q, r)**, not the teacher's training data.
- Until confirmed, this arm is out of scope and is replaced by **arm B′**: DP-SGD with the token clip on, vs arm A with the clip off.

**Prediction (falsifiable).** B or B′ has a significantly lower OP-Print advantage than A, or there is no difference. Both directions are pre-registered.

"ε̂ ≪ ε" is no longer a conclusion, because it is the default outcome.

### H3: dose response

**Dose axis.** `jsd_token_clip` ∈ {0 (off), 0.2, 0.05 (paper), 0.01}.

**Prediction.** Clipping acts element-wise at high-discrepancy entries, so the G-Max advantage should fall monotonically with dose.

**Adaptive attacks (required):**

- G (sum);
- the top-q chunk mean;
- a learned aggregator: logistic regression on the chunk-Δ vector, trained on 2 models and tested on 6.

**Reporting and baseline.** Report utility at each dose. DuoLearn (token-level empirical defense) is the comparator.

### H4: top-k teacher (lowest priority)

OPSD's `--top_k_loss` ∈ {0, 20, 100} vs TT-OPD-style unbiased top-k, compared at matched utility.

### H5: training-side numerical ablation (new)

**Motivation:**

- Barron et al. Fig. 7: with standard soft-target cross-entropy, a student initialized at the teacher is driven away from it in bf16 ("amplifies rounding error into parameter drift in bf16 and a brief KL spike in fp32"). The fixed loss stays at KL = 0.
- The mechanism: Adam normalizes any residual gradient that exceeds its ε into steps of size η.
- In OPSD there are two distinct drift channels **[inference]**:
    1. **Adapter drift.** The cancellation pseudo-gradient moves the LoRA adapter, i.e. the student, at positions where the true Δ ≈ 0. These are disproportionately member positions late in training.
    2. **Teacher drift.** Base-weight drift from bf16 merge/unmerge (§1.2) moves the adapter-disabled base, which is the fixed teacher.
- Both blur the in/out contrast that OP-Print exploits. The direction of their effect on AUC is not obvious, so the ablation could *raise* measured leakage.
- **Caveat [inference].** Barron's regime has a true gradient of exactly 0. In OPSD the aggregate gradient is dominated by high-Δ tokens and globally clipped at 0.1, so the effect size could be negligible. That is what H5 tests.

**Design (2 × 2):**

| Factor | Levels |
|---|---|
| Loss | (a) standard OPSD bf16 `F.kl_div`; (b) **fixed**: a custom autograd function with forward −Σ p s from §1.4, and a backward that uses Barron's log-space u_k = p_k expm1(s_k) = q_k − p_k with the zero-sum correction g_k = u_k − q_k Σ_i u_i |
| Weight sync | bf16 merge/unmerge vs vLLM native LoRA |

- **Clip-aware gradient (our derivation).** With the element-wise clip and unclipped set U, the gradient is g_j = 1[j∈U](q_j − p_j) − q_j(1 − P_U) + q_j·1[j∉U], where P_U = Σ_{k∈U} p_k.
    - Compute (q_j − p_j) via u_j, and 1 − P_U as the sum of the clipped p_k.
    - This must be checked against autograd in fp64 before use.
- **Scale.** Start with one split + complement pair per cell (2 models × 4 cells), on identical splits. Run all 8 splits only if a cell differs by more than the pre-registered threshold (|ΔAUC| ≥ 0.02, paired bootstrap CI excluding 0 **[design choice]**).
- **Outcomes:**
    - OP-Print AUC, TPR@1%FPR and TPR@5%FPR, scored with the fixed pipeline in every cell;
    - base-weight L∞ drift;
    - held-out teacher–student KL on out-samples;
    - pass@1.

**Secondary: Softmax Collapse.**

- This affects high-confidence tokens: a margin above the "precision cliff" (≈ 5.5 in bf16, 16.6 in fp32) zeroes the target gradient (Barron et al. §2). In soft-target terms, p or q rounds toward one-hot.
- Such tokens are common in LLM rollouts but mostly have low Δ. **[inference]** They mainly affect G's mean and the noise floor, not G-Max's high-discrepancy positions.
- fp32 scoring plus log-space evaluation also mitigates them. No dedicated arm.

## 3. Setup and compute

**Pool and design:**

- OpenThoughts math subset, N+ = 1000.
- **4 random splits + their complements = 8 models**, each trained on 500 samples. Every sample is exactly 4 in / 4 out.

**Split-half reliability:**

- Each half takes 2 (split, complement) pairs, so each half is exactly 2 in / 2 out. There are 3 such partitions; average over them.
- **Threshold:** raw split-half r ≥ 0.3, which corresponds to ≈ 0.46 for the 8-model estimate after Spearman–Brown.

**FPC bias (2605.25819, verified):**

- Drawing N = 0.5·N+ from a fixed superset gives FPC = 1 − N/N+ = 0.5. In/out variance is underestimated and vulnerability is overestimated.
- **The complement design pins N/N+ at exactly 0.5**, which is the worst case for this bias, so two critique points conflict here.
- **Resolution:** keep the complement design for H1, which concerns relative ranking, where FPC is approximately a common factor. Apply the paper's FPC correction wherever absolute TPR is reported. The details of its "Decomposed FPC" were only partially read **[unverified]**.
- Concatenated ROCs give inconsistent per-sample FPRs, so report out-calibrated results and note this.

**Scoring:**

- **K = 8 full rollouts.** Fig. 5a (4B) shows G-Max AUC ≈ 0.02 lower at K = 8 than at K = 32.
- Check rank stability with K = 32 on a 200-sample subset.
- "First 64 tokens only" is a cheap ablation. It is **not** the method: the paper's max is over the whole rollout.

**Compute [rough estimate; calibrate on one model]:**

| Item | Estimate |
|---|---|
| Training, 8 models | ≈ 8–16 GPU·h |
| Scoring (≈ 64k rollouts + two prefills) | ≈ 10 h; the fp32 head adds an unmeasured overhead |
| H1′ θ0 pass | ≈ +3 h |
| LR grid | ≈ 4–8 h |
| H5 (4 cells × 2 models + scoring) | ≈ 8–12 h |

**LT-IQR-style training-time ceiling:**

- Log each in-sample's Δ loss every time it is trained on, plus CRN Δ at `save_steps` checkpoints.
- Use the IQR ranking as the ceiling for information available during training, and compare it with c (θ0, pre-training).
- **Caveat:** with 3 epochs, only 3 points per sample come for free. Checkpoint evaluations cost extra.

## 4. Prior work (verification status)

- **2602.15919** (Dorseuil, Atif, Cappé), verified.
    - Per-sample risk from a *single trained* model via loss plus last-layer leverage. **Not ex ante.**
    - Experiments cover classification and regression only (CIFAR, BloodMNIST, SST-2 BERT, tabular data). No generative LLMs.
    - Applying leverage to θ0 is our adaptation **[unverified transfer]**.
- **LT-IQR** (Pollock et al., USENIX Security '25; arXiv 2411.05743), verified.
    - IQR of per-sample, per-epoch loss traces, with no extra models. 92% precision@1% on CIFAR.
    - Image classification only **[abstract]**.
- **2605.25819** (Jälkö et al.), verified:
    1. Shadow sets drawn from a fixed superset inflate vulnerability (FPC).
    2. Concatenated TPR is miscalibrated across per-sample FPRs.
    - Fixes are proposed for both.
- **2505.18773** (Hayes et al., NeurIPS '25), verified.
    - Many per-sample decisions are coin-flip-unstable across training seeds.
    - Vulnerability correlates with sample length (§6.1, App. F).
- **2609.36331** (Rita, Zhang, Tramèr, AISec '26), verified.
    - Per-example calibration without reference models is "the common bottleneck" for one-round attacks and one-run DP auditing. Neighbour-based calibration is proposed.
    - Image tasks only. Relevant if H2 uses one-run auditing.
- **2505.11837.** The v0 claim of a "tension" with OP-Print is **withdrawn**. Its teacher–ground-truth KL on fixed data is a different quantity from OP-Print's teacher–student Δ on student rollouts.
- **DuoLearn (2502.19726).** H3 comparator.
- **2610.31642** (Barron et al.), verified: §1.4, §1.5 and H5.
    - All of its experiments are small grokking or toy-distillation models. Its conclusion itself says the regimes probed "are uncommon in standard large-scale training".
    - Transfer to LLM-scale OPSD is **[unverified]**, which is why the effect is measured, not assumed.

## 5. Execution order and stopping rules

1. Bring up the OPSD code. Then:
    - bf16 base-drift check (§1.2);
    - **noise-floor check N0/N0′/N1 and ε calibration (§1.5)**;
    - LR grid (§1.1; f_rm and pass@1 only).
2. **H0** on one model. **Kill:** the paired-bootstrap AUC-difference CI includes 0.
    - If H0 fails, first run the standard vs fixed scoring diagnostic.
    - Optionally run the 0.6B full-parameter control.
3. Train 8 models and apply the reliability gate. If **raw r < 0.3**, drop per-sample H1 and switch to risk stratification (§6).
4. H1 and H1′ in parallel, then the LT-IQR ceiling.
5. Then, in this order:
    - H3 (dose axis; no DP; cheap);
    - H5 (numerical ablation; can share models with H3's clip = 0.05 cell);
    - H2 (pending Alex's confirmation of arm B's privacy unit);
    - H4.

## 6. Risks

- **Risk stratification may be the main deliverable,** not merely a fallback, given 2505.18773's per-sample instability.
    - Bin samples into quintiles of c and test monotonicity of the pooled TPR@1%FPR.
    - This is more robust than per-sample ranking and still delivers an ex-ante risk indicator.
- **Numerical failure mode (new).** bf16 logit quantization, the softmax-cancellation floor and the inflated G-Max noise maximum can compress the in/out log-ratio. H0/H1 could fail for numerical rather than substantive reasons. §1.4–1.5 and the H0 diagnostic address this.
- The effect is weak (AUC ≈ 0.6).
- If the LoRA-configuration inference is wrong, replication may fail.
- bf16 drift may contaminate the teacher.
- Long r makes prefill costly.
- The sampling temperature must stay at T = 1.1.
- If arm B's privacy unit is not the distillation record, H2 reduces to A vs B′.

## 7. Where I disagree with the critique (unchanged from v1)

1. **"LoRA at 5e-6 barely moves the adapter."** The premise largely fails. The official OPSD configuration is LoRA r = 64 at 5e-6; OP-Print very likely used it and measured AUC 0.59 there. The Δ-removal LR procedure is still adopted because it is sound, with 5e-6 kept as the anchor.
2. **K and 64-token truncation.** K counts rollouts, and chunks live inside full rollouts, so truncation is an ablation only.
3. **The complement design conflicts with 2605.25819's FPC point.** Resolution in §3.
4. **Per-model z-scoring is incompatible with a ratio outcome.** Multiplicative out-calibration is used instead.
5. **The 0.6B full-parameter control is optional** (reason in §1.3).
6. **LT-IQR is not free in OPSD.** 3 epochs give 3 points per sample.

All other points were adopted.

## 8. Notes on the v1.1 additions (precision of claims)

- **Barron's ∼10⁻³ (bf16) and ∼10⁻⁸ (fp32) are residual gradient magnitudes at z_S = z_T in one toy setting**, not KL-value floors and not LLM measurements. Our ε comes only from the N1 measurement.
- The KL direction matches without reversal (§1.4, step 3).
- The log1p-form KL and the clip-aware gradient are our derivations, to be validated against fp64.

## 9. References

- **[OP-Print]** Q. Li, X. Wang. arXiv 2610.31616 (2026).
- **[OPSD code]** github.com/siyan-zhao/OPSD (`opsd_trainer.py`, `opsd_train.py`, `scripts/`, README; read 2026-10-26).
- **[Barron+26]** J. T. Barron, D. Gnanapragasam, M. Tschannen. "A Fix for Softmax and Cross-Entropy". arXiv 2610.31642 (2026).
- **[DLC26]** Dorseuil, Atif, Cappé. arXiv 2602.15919 (2026).
- **[LT-IQR]** Pollock, Shilov, Dodd, de Montjoye. USENIX Security '25, pp. 5525–5544; arXiv 2411.05743.
- **[JPRH26]** Jälkö, Pradhan, Räisä, Honkela. arXiv 2605.25819 (2026).
- **[Hayes+25]** Hayes et al. arXiv 2505.18773 (NeurIPS 2025).
- **[RZT26]** Rita, Zhang, Tramèr. arXiv 2609.36331 (AISec '26).
- **[KD-MIA]** arXiv 2505.11837.
- **[DuoLearn]** arXiv 2502.19726.
- **[Steinke+23]** Steinke, Nasr, Jagielski. "Privacy Auditing with One (1) Training Run", NeurIPS 2023 (one-run auditing; not re-read for this version).
