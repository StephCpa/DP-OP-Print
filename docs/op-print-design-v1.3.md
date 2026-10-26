# Ex-ante prediction of membership leakage in on-policy self-distillation (OPSD): experimental design v1.3 (2026-10-26)

**Basis.** v1.3 builds on v1.2 and folds in a round of pre-GPU desk research:

- an OPSD source audit (`docs/verification/opsd-code-audit.md`);
- numerics at V = 151,936;
- an analysis of the clip mechanism;
- simulations of the design statistics;
- a toy OPSD dry run of the full analysis.

All of these are reported in `docs/results/research-report-2026-10-26.md`, which this document cites as R1–R5 and F1–F13. Nothing in that round used Qwen3 weights or a GPU. OP-Print itself could not be re-read, because `arxiv.org` and `huggingface.co` were unreachable.

**Markers.**

- **[verified]**: read in code or a primary source.
- **[unverified]**, **[inference]**, **[estimate]**: not taken directly from a primary source.
- **[assumption]**: a modelling assumption.
- **[design choice]**: a pre-registered threshold that is ours.

**Earlier versions.** v0 and v1 (in Chinese), v1.1 and v1.2 (in English) are archived unchanged in `docs/archive/`.

## Changelog v1.2 → v1.3

1. **New code facts (§0, from R1).**
   - The OPSD loss runs at the **sampling temperature T = 1.1**.
   - The 1.7B/4B/8B scripts pair a **thinking-mode teacher with a non-thinking student**, and the teacher prompt adds "explore/backtrack" instructions.
   - **t = 1 is read at a pad position** in training for all but the longest prompt in a micro-batch.
   - The training budget is **48 steps with 3 rollouts per member**.
   - The clip = 0 semantics depend on the entry point.
2. **§0.1 strengthened (R3).**
   - A clamped logit is always pushed *down*: g_a = q_a·P_U ≥ 0.
   - The clipped loss has negative-loss optima that abandon teacher tokens with p ≳ clip.
   - The one-step direction of q_a has a closed form: it rises for a sharp student and falls for a diffuse one.
3. **§1.0 diagnostic extended.**
   - Log Δ₁ and the argmax-chunk index.
   - Report G/G-Max without t = 1 and without chunk 1.
   - Add a **no-r teacher context** p_t∖r, to separate template- and instruction-induced Δ from r-induced Δ.
   - Score Δ at both T = 1 and T = 1.1.
4. **§1.1.** Pin effective batch 32, `vllm_sync_frequency` and the thinking flags. Add a θ0 **rollout-noise measurement**: two independent K = 8 rollout sets per sample.
5. **§1.4.** The fp32 **log1p form becomes primary**, because it is 2–8× more accurate than Barron's form at V = 151,936. Sequences are built without mid-sequence pads.
6. **§1.5.** The bf16 floor follows Σ_k p_k(1−p_k)s_k²/12, which is logged per position. N1 is not a reliable lower bound, because bitwise-identical passes give exactly zero. N2 stays primary.
7. **H1 competitor.** m_U is negative at correction sites (R3), so it is replaced by **r_dir**, the direct first-order full-KL reduction under the clipped gradient. m_U and m_U⁺ are secondary.
8. **H1 inference.** The primary test becomes **OLS + HC3 with 16 assignment-pattern fixed effects and a cubic spline in d**. The v1.2 sign-flip placebo is anti-conservative under heterogeneous leakage (10.7–13.7% at nominal 5%, R4-S2), so it is demoted to a sharp-null sanity check.
9. **Gate.**
   - A **staged reliability gate** after 2 split/complement pairs (4 models) replaces "train 8, then test". Continue only if the Spearman–Brown projection 2r_11/(1 + r_11) ≥ 0.3.
   - A variance-based post-H0 bound was derived (R4-S1) but is too loose to gate: 0.44–0.96 in the toy where r ≈ 0 (R5).
10. **H3.**
    - Sign effects are expected at clip 0.05 already, so sign-agnostic attacks are primary at every dose. The **learned aggregator** is the primary sign-agnostic attack. |score − median_out| is demoted, because it detects dispersion, not shifts, and was uninformative in every toy arm.
    - **New co-primary outcome:** the AUC level of the strongest pre-registered attack, and utility, by clip dose.
    - The dose trend is restricted to doses with detectable leakage.
    - MDEs are stated: 0.007 for 8 vs 8 models, 0.016 for 2 vs 2.
    - The clip 1e-6 dose is dropped unless run in fp32, because bf16 clamp status is unstable there.
    - New secondary arm: `--teacher_thinking False` at clip ∈ {0, 0.05}, 2 models each.
11. **Toy dry run (R5).** The full analysis ran end to end on a synthetic OPSD system (80 models). In the toy:
    - leakage falls sharply under any clip;
    - gap(0) − gap(0.05) = +0.038 [0.030, 0.046];
    - at clip ≥ 0.01 the template spike turns G-Max into a chunk-1 attack;
    - the θ0-teacher arm is the strongest score and flips sign at clip 1e-6;
    - f_rm targets were met without leakage;
    - leakage was homogeneous (r ≈ 0), and the staged gate stops correctly.

    These drive items 9 and 10, the f_rm caveat (§1.1), and H1 multiplicity (one primary test per clip).
12. **Compute (§3)** is re-estimated for the additions: about +10–14 GPU·h.

The v1.1 → v1.2 changelog is in the archived v1.2 document.

## 0. Facts pinned from the paper and code

**Signal.** Δ_t = KL(p_θ(·|p_t(q,r), y<t) ‖ p_θ(·|p_s(q), y<t)).

- Both sides use the attacked model θ, so the teacher side runs **with the adapter on**.
- Rollouts are sampled as y ~ p_θ(·|p_s) at T = 1.1, top-p 0.95, top-k 20.
- Whether OP-Print's Δ uses T = 1 or T = 1.1 is **[unverified]**, so both are scored.

**Gray-box scores.** G averages Δ over chunks, and G-Max takes the maximum over 64-token chunks. Both are averaged over K full rollouts (paper default K = 32).

**Training loss (OPSD, β = 0) [verified].** The summand is f_k = p_k(log p_k − log q_k), where:

- p = softmax(z_T / T) is the teacher, θ0 (adapter disabled, no grad) under p_t;
- q = softmax(z_S / T) is the student, θ under p_s;
- **T = 1.1** is the same temperature as sampling.

`jsd.clamp(max=clip)` acts element-wise on f before the vocabulary sum, and the loss is the mean over generated tokens.

- `opsd_train.py` maps clip ≤ 0 to `None` (off). The trainer itself only checks `is not None`, so **the H3 harness passes `None` for clip off**. Assertion: at clip off, the logged loss equals the unclipped KL.
- There is no `.float()` before `log_softmax`. The logit dtype follows the `transformers` version **[inference: bf16]**.

**Prompts [verified].**

- **Student:** `Problem: {problem}` followed by the `\boxed{}` instruction, with `enable_thinking=False`.
- **Teacher:** the same problem, "Here is a reference solution …" with r between markers, a transition prompt (re-derive the solution, "exploring approaches and backtracking"), and the `\boxed{}` instruction, with `enable_thinking=True`. True is the default and no main script overrides it. Only the `*_nonthink.sh` scripts set both flags to False.
- **Template [unverified here; Qwen3 template as recalled].** The student's generation follows an empty `<think>\n\n</think>\n\n`. The teacher's context ends at `assistant\n`.

**Padding [verified; effect is inference].** Prompts are right-padded to the micro-batch maximum and concatenated with the generation. No `position_ids` are passed, and logits are sliced at the batch-level prompt length. For shorter prompts, the logit for the first generated token is therefore read at a pad position.

**Budget [verified].** One rollout per prompt per step. Effective batch is 32 prompts in all scripts. At N = 500 and 3 epochs that is **48 optimiser steps and 3 training rollouts per member**. `lmbda` is unused.

**Data [verified].** `siyanzhao/Openthoughts_math_30k_opsd`, fields `problem` and `solution`, `--max_length 20000`.

**OPSD README [verified].**

- The clip is "per-token point-wise KL clipping", motivated by style tokens with "6–15× higher KL".
- The loss can be negative.
- The non-thinking tables use clip 1e-6 (1.7B, 4B) or 1e-7 (8B).
- The tinker-style loss is a sampled-token policy gradient with no clip **[verified in code]**.

**OP-Print configuration.** Table 3 uses clip 0.05 (0.06 for 8B), matching the official scripts.

**OP-Print reported results (AUC).**

| Objective | Score | 1.7B | 4B | 8B |
|---|---|---|---|---|
| JSD, clip 0.05 (Table 1) | G | 0.579 | 0.580 | 0.556 |
| | G-Max | 0.594 | 0.611 | 0.587 |
| Tinker-style (Table 2) | G | 0.571 | 0.563 | 0.548 |
| | G-Max | 0.583 | 0.571 | 0.579 |

### 0.1 What the element-wise clip does (Appendix A; R3)

Let U = {k : f_k ≤ clip} and P_U = Σ_{k∈U} p_k. The gradient with respect to z_S/T is:

**g_j = q_j·P_U − p_j·1[j∈U]**

This was verified in fp64 (Appendix A).

**M1: clamped logits go down [verified algebra, fp64].** For every clamped a, g_a = q_a·P_U ≥ 0. Gradient descent never raises a clamped logit.

**M2: the clipped objective has optima away from the teacher.** With free logits:

- inf L = min_C [ |C|·clip + (1−P_C)·log(1−P_C) ];
- this is < 0 = L(q = p) as soon as one teacher token has p_k − p_k²/2 > clip, so q = p is only a local minimum;
- worked example (student uniform over 20 tokens, p_a = 0.9, clip 0.05): long gradient descent ends at L = −0.180 with **q_a = 9e-4**, i.e. the teacher's 0.9 token is abandoned, and the full KL rises from 2.38 to 5.95;
- 300 random positions: gradient descent ends with the full KL above its starting value in 34 / 67 / 87 / 98% of cases at clip 0.2 / 0.05 / 0.01 / 1e-6.

**M5: the one-step direction of a clamped token.** d log q_a ∝ P_U(Σ q_i² − q_a) − Σ_U p_i q_i. q_a rises when the student is sharp and falls when it is diffuse.

**r_dir.** The direct first-order change of the full KL is −r_dir, with r_dir = ⟨q − p, g⟩. It is negative at 80–82% of synthetic correction sites at clip 0.05 (T = 1 and 1.1).

**Temperature.** At T = 1.1 the clamped set shrinks a lot for diffuse positions: the median clamped KL share at clip 0.05 falls from 105% at T = 1 to 0%. Correction sites stay clamped.

**Consequences [inference].**

- "Uncertain student, confident teacher" is OP-Print's correction site. Under clip 0.05 these entries get a direct *downward* push. If training ran long enough, the direct route would move the student *away* from the teacher there.
- Over the real budget (48 small steps) the direct effect is small. Any membership imprint at the max-Δ chunks therefore arises mainly **indirectly**:
  1. spillover from learning on unclamped entries;
  2. the adapter changing the p_t side, since Δ_θ runs the teacher context through θ.
- G-Max still beats G marginally in OP-Print, so one of these routes must be active.
- At the README's clips of 1e-6/1e-7, nearly every under-predicted entry is clamped. The objective then rewards concentrating the student on a set of teacher mass 1/e, a sharpening pressure.
- The thinking-template token at t = 1 (§0) is clamped at every clip and gets no direct training. Without the clip, forward KL pushes the student to emit it. In the toy (R5), an unclipped student learns to emit THINK first in 98.5% of rollouts, and in ≤ 0.2% under any clip. One plausible reading of why the authors found clipping "stabilises training" **[inference]**.

## 1. Pre-H0 steps

### 1.0 First diagnostic (cheap; runs inside scoring)

1. **For every chunk, log:**
   - the argmax-Δ position and the **chunk index of the argmax chunk**;
   - the sampled token y_t at that position, and the teacher's top-1 token;
   - the fraction of Δ_t at that position that lies in entries clamped at 0.05, at T = 1.1.
2. **Log Δ₁** (t = 1) separately. Report G and G-Max with and without t = 1, and with and without chunk 1.
3. **Run under θ (the attack) and under θ0 (CRN rollouts).**
4. **No-r teacher context p_t∖r (new).** Same template, thinking flag and instructions, with the solution block removed. Under θ0, log:
   - Δ^instr = KL(p_θ0(·|p_t∖r) ‖ p_θ0(·|p_s)) per position;
   - KL(p_θ0(·|p_t) ‖ p_θ0(·|p_t∖r)).
   This costs one extra θ0 prefill per sample.
5. **Classify tokens** with a lexicon fixed before looking at the data **[design choice]**:
   - **style**: wait, hmm, but, so, let, actually, alternatively, think, okay, …;
   - **math**: digits, operators, LaTeX, variables;
   - **template**: `<think>`, `</think>`, role tokens;
   - **other**.
6. **Report** the class shares for in- vs out-samples, the top-50 argmax tokens, the share of argmax chunks that are chunk 1, and the share of Δ explained by Δ^instr.

**Reading.**

- If the maxima fall mostly on style tokens that keep their Δ without r, OP-Print's "correction sites" are a **style or template imprint**, not a reference-correction imprint.
- If chunk 1 dominates the argmax, G-Max is effectively a first-chunk attack.

### 1.1 Learning rate, budget and f_rm

- **H0 runs at lr 5e-6** (LoRA r = 64, α = 128, 3 epochs), **effective batch 32 prompts × 1 rollout** via gradient accumulation, **whatever value f_rm takes**.
  - Script defaults are kept: student non-thinking, teacher thinking.
  - `vllm_sync_frequency` is pinned and logged **[unverified default from TRL's `GOLDConfig`]**.
- **f_rm definition.** f_rm = 1 − mean_{x∈train} D_θ(x)/D_θ0(x), with D_θ = KL(p_θ0(·|p_t) ‖ p_θ(·|p_s)), on CRN rollouts. The clipped residual Σ min(f, clip) at T = 1.1 is also reported. f_rm is a diagnostic only.
- **Fallback (only if H0 fails).** Run the grid {2e-5, 5e-5, 1e-4}, selected by f_rm ∈ [0.2, 0.5] with a pass@1 drop of at most 2 points **[design choice]**. Never select on attack metrics.
  - **Caveat from R5:** in the toy, f_rm can be met by learning the shared template, style and instruction components alone, with no membership signal at all. If the fallback runs, it also reports **f_rm on the no-r context** (generic learning) and **f_rm^r = f_rm − f_rm(no-r)**.
- **Clip-rate logging.** Log how often max-grad-norm = 0.1 clips, and the median gradient norm.
- **θ0 rollout noise (new).** Draw two independent K = 8 rollout sets per sample under θ0 and compute σ_roll² = ½·Var(log Δ̄_set1 − log Δ̄_set2), together with σ_a² from the between-sample variance. This runs inside the θ0 pass.
  - It informs the K = 8 vs K = 32 choice and the variance decomposition.
  - It is **not** used as a gate: the variance-based bound is too loose (R4-S1, R5).

### 1.2 bf16 merge/unmerge and base drift

Unchanged from v1.2.

- Per-layer ρ_ℓ = ‖W_base,ℓ(end) − W_base,ℓ(start)‖_F / ‖(α/r)·B_ℓA_ℓ‖_F.
- Dump the in-memory base at the end of training.
- If drift is non-negligible, score arm 3 with the drifted base as well and switch to vLLM native LoRA.
- Scoring always uses the unmerged adapter, or an fp32 merge.

### 1.3 Full-parameter control (optional; only if H0 fails)

Unchanged. Qwen3-0.6B needs ≈ 12 GB with 8-bit Adam **[estimate]**.

### 1.4 Numerically stable Δ

- **Pipeline.** bf16 trunk with an **fp32 LM-head projection**. Per position, in fp32:
  - δ = z_S − z_T;
  - d_k = δ_k − Σ_i p_i δ_i;
  - **Δ_t = log1p(Σ_i p_i [expm1(d_i) − d_i])**. This is primary: at V = 151,936 in fp32 it is 2–8× more accurate than Barron's form (R2-P2).
  - Barron's −Σ_k p_k s_k, with s_k = d_k − log1p(Σ_i p_i expm1(d_i)), is the cross-check.
- **Temperatures.** Both T = 1 and T = 1.1, from the same logits.
- **Direction.** z_T comes from the privileged context, so this is KL(p‖q) = OP-Print's Δ.
- **Sequence construction (new).** No mid-sequence pads: per-sample sequences, or left padding with explicit `position_ids`.
- **Chunks.** 64-token position chunks, ≈ 39 MB per side.
- **Qwen3-1.7B facts [verified from the HF config in v1.2; arithmetic re-checked].** Hidden 2048, intermediate 6144, 28 layers, 16 query heads / 8 KV heads, head_dim 128, vocabulary 151,936, tied embeddings. KV cache 114,688 B/token in bf16.

### 1.5 Noise floor and calibrated ε

**Analytic floor (new, logged per position).** For bf16 logits:

ε̂_t = Σ_k p_k(1−p_k)·s_k²/12, with s_k the bf16 spacing at z_k.

On synthetic logits this matches the measured bias: ≈ 1e-4 / 4e-4 / 1e-3 nats/token for top logits near 8 / 16 / 24 (R2-P3). The reviewer's 3e-4 prior is the [8, 16) case. The fp32 head lowers the floor ≈ 10×. That covers final-hidden-state rounding only.

**Tests.**

- **N0 / N0′.** Arithmetic, as in v1.2.
- **N1.** Kept, but **not** treated as a lower bound unless its two passes are shown to differ bitwise. Identical logits round identically and give exactly zero (R2-P3).
- **N2 (primary).** Unchanged: a full fp32 forward pass on ≈ 100 samples (50 in, 50 out). Report the relative error separately for in- and out-samples, for G and G-Max, and report v_prod − v_ref.

**ε.** ε_G and ε_GMax are the 99th percentile of the absolute N2 error of each aggregate, floored at the median per-sample ε̂ **[design choice]**. Winsorization at 1/99 is kept.

**Gate.** If the median relative N2 error for in-samples exceeds 5%, fix the pipeline before H0. This is also H5 trigger (ii).

## 2. Hypotheses

**Minimal paper.** H0 → H1/H1′ → H3. H2, H4 and H5 are conditional.

### H0: replication (lr 5e-6, LoRA r = 64, clip 0.05, effective batch 32)

- **Primary test.** G-Max AUC minus the best baseline's AUC, by paired bootstrap. The best baseline is selected on a disjoint half of the samples.
- **G vs G-Max.** Paired DeLong per model, plus the spread across models.
  - Under a chunk model calibrated to the paper (R4-S3), a 0.015 gap is ≈ 2.7 SD within one model *if* G and G-Max are highly correlated.
  - Report the observed score correlation.
- **Also report** G/G-Max without chunk 1 (§1.0).
- **Scoring diagnostic.** Standard bf16 vs fixed routine. |ΔAUC| ≥ 0.02 is H5 trigger (iii).
- **Kill.** The bootstrap CI includes 0. Then run the §1.1 / §1.3 fallbacks.

### H1 (main line): does θ0 concentration predict leakage?

**Primary c (pre-registered, only one).** c(x) = the top-10% chunk share of Δ_θ0, averaged over K θ0 rollouts. max/mean is secondary.

**Mechanistic competitor (changed).**

- **r_dir(x) = mean_t ⟨q_t − p_t, g_t⟩** under θ0 at T = 1.1 and the target run's clip, with g = q·P_U − p·1_U. It is the direct first-order reduction of the full θ0-teacher KL that a clipped step would make, under an isotropic-NTK **[assumption]**.
  - At clip 0, r_dir = mean ‖q − p‖².
  - It is computed from logits already in memory in the θ0 pass.
- **Secondary:** v1.2's m_U = Σ_U f (negative at clamped correction sites, R3) and m_U⁺ = Σ_{k∈U} max(f_k, 0).

**Outcome.** v(x) = log((Δ̄_out + ε)/(Δ̄_in + ε)) after multiplicative out-calibration and winsorization. The difference is secondary.

**Covariates d.** θ0 pass@k, log Δ̄_θ0, log T, θ0 loss, θ0 last-layer leverage (transfer [unverified]), **and log Δ̄^instr_θ0** (no-r context).

**Primary inference (changed).**

- Fit v ~ c + spline₃(d) + pattern FE: a cubic spline with knots at the quartiles of each continuous d, and 16 assignment-pattern fixed effects that absorb model offsets.
- Report c's coefficient with **HC3** SE, one-sided at α = 0.05, and its incremental R².
- Freedman–Lane permutation is a robustness check.
- Calibration: in simulation this test held 4.0–7.0% under all nulls considered, where v1.2's sign-flip placebo gave 10.7–13.7% (R4-S2).

**Multiplicity.** There is one primary test per clip: c at clip 0 and c at clip 0.05, Holm-corrected. All competitor and secondary-outcome tests are secondary. In the toy, ≈ 30 secondary tests produced chance |t| ≈ 2–3 at zero reliability (R5).

**Sanity checks.** The within-pair sign-flip placebo (exact for the sharp null) is kept. c must not predict v_placebo, and the sharp-null rejection rate is reported.

**Pre-registered predictions (sparse-imprint story).**

- **clip = 0:** c has a positive incremental effect over d.
- **clip = 0.05:** r_dir's incremental R² over d exceeds c's. Concentrated mass sits in clamped entries, which are pushed away from the teacher directly (M1), so leakage must come from trainable mass.
- If c beats r_dir at clip 0.05 as well, the imprint is not mediated by direct training at the high-Δ entries. That is consistent with the indirect routes in §0.1.

**Power.** With N = 1000, the MDE for the partial correlation is ≈ 0.08 (one-sided 5%, 80% power). Measurement error in v attenuates the observed partial r by √(reliability). At the gate's 8-model reliability of 0.46, the true partial r must therefore be ≳ 0.12.

**Kill.** c adds nothing over d.

### H1′: reference-calibrated attacks (CRN)

1. S_θ − S_θ0.
2. ρ(x) = ΣΔ_θ / ΣΔ_θ0.
3. The θ0-teacher arm D_θ (the training objective, subject to the drift caveat in §1.2), plus a clipped variant at T = 1.1.
4. Δ_θ − D_θ, which isolates the p_t-side change (indirect route 2).
5. **New:** S computed on Δ − Δ^instr. Is the membership signal in the r-specific part?

### H3: clip-mechanism test

**Doses.** clip ∈ {0 (passed as `None`), 0.2, 0.05 (paper), 0.01}. 1e-6 is dropped unless the loss runs in fp32: in bf16, 350–700 entries per position flip clamp status (R2-P1).

**Primary outcome.** gap(clip) = AUC(G-Max) − AUC(G), tested two-sided by a paired bootstrap of gap(0) − gap(0.05) on the same splits and samples. The dose trend is secondary (see **Trend** below).

**MDE (R4-S3, calibrated chunk model).** 0.007 for 8 vs 8 models, 0.016 for 2 vs 2. The 0.2 / 0.01 cells detect only gap changes of ≳ 0.016.

**Predictions.**

- The sparse-imprint story predicts that **the gap widens as clip → 0**, because the strong sites are then trained directly.
- R3 adds that at clip 0.05 the direct effect at those sites points *away* from the teacher. A G-Max that is no better than G, or worse, at clip 0.05 is therefore admissible without implying no leakage.
- **Toy (R5, not evidence about Qwen3).**
  - The imprint sits at correction sites when unclipped (answer positions: log out/in = +0.174).
  - The gap widens as clip → 0: gap(0) − gap(0.05) = +0.038 [0.030, 0.046]. Without the template it is +0.023 [0.018, 0.029].
  - With the default thinking teacher, G-Max at clip ≥ 0.01 is dominated by chunk 1 (97–98% of argmax chunks) and is no better than G (gap −0.006 to −0.015). Without chunk 1 the gap is positive again (+0.011 at clip 0.05).
  - The pre-registered contrast is therefore computed **both with and without chunk 1**. The "without chunk 1" version is primary if §1.0 finds chunk 1 to be the argmax in > 50% of rollouts **[design choice]**.

**Co-primary outcome (new): leakage level by dose.** Report the AUC (with CI) of the learned aggregator and of D_θ at each dose, together with pass@1 (utility). In the toy, any clip ≤ 0.2 cut the clip-off AUC from 0.58–0.61 to 0.51–0.53 (R5). Whether the clip acts as an empirical defense at matched utility is directly relevant to the DP comparison.

**Trend.** The Spearman trend across doses is computed only over doses whose G or G-Max AUC CI excludes 0.5. It is also reported for gap/(AUC_G − 0.5). In the toy the unrestricted trend was misleading (−0.3), because clip 1e-6 had no leakage.

**Sign-agnostic attacks (primary at every dose).**

- **Primary:** a learned aggregator on the sorted CRN chunk vector of Δ_θ − Δ_θ0, trained on 2 models and tested on 6. In the toy it was ≥ G-Max in every clipped arm.
- **Single scores:** two-sided AUC CIs. AUC < 0.5 is reported, never folded into max(AUC, 1 − AUC). The toy's D_θ at clip 1e-6 gave 0.480 [0.469, 0.489].
- **Secondary:** |score − median_out|. It detects dispersion differences only, and was uninformative in every toy arm.

**Template arm (new, secondary).** `--teacher_thinking False` at clip ∈ {0, 0.05}, one split + complement each. It tests whether the clip-0 vs clip-0.05 contrast is driven by the template token.

**Existing weak evidence.** OP-Print's tinker-style objective is unclipped **[now verified in code]**, but it is a different objective. Its gaps (+0.012 / +0.008 / +0.031) are no wider than under clip 0.05. **Inconclusive.**

**Scale.**

- 8 models at clip ∈ {0, 0.05}, which also supports H1's two-clip predictions.
- 2 models at each of {0.2, 0.01}.
- 2 + 2 models in the template arm.

### Conditional hypotheses

**H2: DP comparison.** Unchanged from v1.2.

- A utility-only pilot (ε = 8, δ = 1e-5, r = 8, full batch) comes first.
- Arm B needs Alex's confirmation that its privacy unit is (q, r).
- The adapter has 69,730,304 parameters at r = 64 and 8,716,288 at r = 8 [verified arithmetic].

**H4: top-k teacher.** Conditional on H3: `--top_k_loss` ∈ {0, 20, 100}. It renormalises both distributions over the teacher's top-k [verified].

**H5: training-side numerical ablation.** Triggers as in v1.2. Added motivation from R2-P1:

- bf16 gradient error is 3–8% at T = 1 and 6–22% at T = 1.1. Dividing bf16 logits by T rounds them twice.
- At near-equal positions the bf16 gradient is noise.

The fixed loss divides by T in fp32 and uses g_j = q_j P_U − p_j 1[j∈U] with u_j = q_j − p_j in log space.

## 3. Setup and compute

**Data and splits.**

- Pool: `siyanzhao/Openthoughts_math_30k_opsd`, N+ = 1000.
- 4 random splits + complements, so each sample is exactly 4 in / 4 out. The same splits are used in every arm.
- Split-half reliability over pair-halves. Gate: raw r ≥ 0.3, plus the post-H0 bound (§5).

**FPC.** The complement design fixes N/N+ = 0.5. "FPC is roughly a common factor" remains an **[assumption]**. Pattern fixed effects in H1 absorb model-level offsets but not per-sample FPC bias.

**Scoring.** K = 8 full rollouts, with a K = 32 check on 200 samples.

**Compute [estimate; per-unit figures from v1.1].**

| Item | GPU·h |
|---|---|
| Pre-H0: drift dump, N0/N1, N2, θ0 pass with no-r prefill and a second rollout set (rollout noise), §1.0 diagnostic | 5–8 |
| H0: 1 model + scoring + standard-vs-fixed re-score | 3–5 |
| H1/H1′ at clip 0.05: 7 further models + scoring + θ0-teacher arm | 20–30 |
| H3: 8 models at clip 0 + 4 models at {0.2, 0.01} + scoring | 30–45 |
| H3 template arm: 4 models + scoring | 7–10 |
| **Minimal paper total** | **≈ 65–98 GPU·h ≈ 2.7–4.1 GPU-days** |

Conditional extras: H5 8–12 GPU·h; H2 pilot ≈ 3–5; H4 ≈ 6–10.

## 4. Prior work

Status is unchanged from v1.2 unless marked.

- **2602.15919 (Dorseuil, Atif, Cappé)** ✓.
- **LT-IQR** ✓.
- **2605.25819 (Jälkö et al.)** ✓.
- **2505.18773 (Hayes et al.)** ✓.
- **2609.36331 (Rita, Zhang, Tramèr)** ✓.
- **2505.11837** ("tension" claim withdrawn).
- **DuoLearn (2502.19726).**
- **2610.31642 (Barron et al.)**. Transfer [unverified]. The fp32 results at V = 151,936 in R2 are ours.
- **New, unread [titles from a web search only; content unverified]:**
  - "One Symptom, Three Levers: A Critical Review of On-Policy Self-Distillation" (arXiv 2608.25936);
  - "What Does Privileged Information Add to On-Policy Self-Distillation?" (arXiv 2609.20612);
  - "DAPD: Dual-Anchored Policy Distillation" (arXiv 2608.01735; search snippets mention a "privilege illusion" from teacher–student information asymmetry).

  These bear on §0's asymmetries and must be read before H0.

## 5. Execution order and stopping rules

1. **Bring up OPSD.** Then:
   - drift dump;
   - N0/N1/N2, ε̂ and ε;
   - the θ0 pass (no-r context, two rollout sets);
   - train the pilot model at 5e-6, effective batch 32;
   - the §1.0 diagnostic.
2. **H0.** Kill if the CI includes 0. Otherwise:
   - check H5 triggers (ii) and (iii);
   - report the variance-based reliability bound (R4-S1) as descriptive only. It is too loose to gate (R5).
   - Only if H0 fails, run the LR-grid and 0.6B fallbacks.
3. **Remaining models at clip 0.05, staged.**
   - First complete 2 split/complement pairs (the H0 model, its complement, and one more pair). Compute r_11 across the two pairs.
   - **Continue to 8 models only if 2r_11/(1 + r_11) ≥ 0.3.** Otherwise switch per-sample H1 to risk stratification. H1′ and H3 continue with 4 models per arm: H3 compares 4 vs 4, with MDE ≈ 0.01, scaled from R4-S3.
   - After 8 models, recompute the full split-half gate as before.
4. **H1, H1′ and the LT-IQR ceiling.**
5. **H3:** 8 models at clip 0 (4 if the staged gate stopped H1), then the 0.2 / 0.01 cells and the template arm. Then H1's two-clip comparison, if H1 is still running.
6. **Conditional work:** H5 if a trigger tripped; the H2 pilot, then H2 if feasible and the privacy unit is confirmed; H4.

## 6. Risks

- **Risk stratification may be the main deliverable.** The staged gate tells after 4 models.
- **Homogeneous leakage.** H0 can pass at AUC ≈ 0.6 while per-sample leakage has no reliable variance. In the toy, r ≈ 0 in every arm despite AUC 0.58–0.61 (R5).
- **Style or template imprint.** §1.0 and the no-r context may show that OP-Print's signal is driven by the instructions or the template (R1, F2).
- **No leakage at the replication point.** The toy reached f_rm targets without any membership signal (R5). H0 can fail for that reason alone.
- **Clip-induced anti-teacher dynamics** at correction sites (R3, and the toy's f_rm < 0 under clipping, R5). Sign effects can appear at clip 0.05.
- **Numerical compression.** Bounded by ε̂ and N2. A problem only if member Δ̄ ≲ 1e-3 nats/token with the fp32 head.
- **H2 likely infeasible** at N = 500.
- **Weak base effect:** AUC ≈ 0.6.
- **Unverified FPC common-factor assumption.**
- **Unread OPSD critiques** (§4).

## 7. Disagreements with earlier critiques: resolution

Items 1–6 are unchanged from v1.2 §7.

**Correction to our own v1.2.**

- The v1.2 placebo argument ("the exact null is a within-pair sign flip") is right for the sharp null and wrong for H1's composite null. The test is replaced (changelog item 8).
- v1.2's m_U is not a mass (changelog item 7).

## Appendix A: fp64 toy verification

Unchanged; see `docs/archive/op-print-design-v1.2.md`, Appendix A, and `scripts/toy_clip_grad.py`. Its stated limits (V = 50, no bf16, log1p not run in fp32) are lifted by R2.

## Appendix B: desk research

`docs/results/research-report-2026-10-26.md` (R1–R5, F1–F13). Scripts and outputs are in `experiments/`.

## References

- **[OP-Print]** Q. Li, X. Wang. arXiv 2610.31616 (2026).
- **[OPSD]** github.com/siyan-zhao/OPSD: `opsd_trainer.py`, `opsd_train.py`, `data_collator.py`, `scripts/run_opsd_{1b,4b,8b,8b_nonthink}.sh`, README (read 2026-10-26).
- **[Barron+26]** J. T. Barron, D. Gnanapragasam, M. Tschannen. "A Fix for Softmax and Cross-Entropy". arXiv 2610.31642.
- **[DLC26]** Dorseuil, Atif, Cappé. arXiv 2602.15919.
- **[LT-IQR]** Pollock, Shilov, Dodd, de Montjoye. USENIX Security '25; arXiv 2411.05743.
- **[JPRH26]** Jälkö, Pradhan, Räisä, Honkela. arXiv 2605.25819.
- **[Hayes+25]** arXiv 2505.18773 (NeurIPS 2025).
- **[RZT26]** Rita, Zhang, Tramèr. arXiv 2609.36331 (AISec '26).
- **[KD-MIA]** arXiv 2505.11837.
- **[DuoLearn]** arXiv 2502.19726.
- **[Steinke+23]** Steinke, Nasr, Jagielski. NeurIPS 2023.
- **[Qwen3-1.7B config]** huggingface.co/Qwen/Qwen3-1.7B/config.json (read for v1.2).
- **[OPSD critiques, unread]** arXiv 2608.25936; arXiv 2609.20612; arXiv 2608.01735.
