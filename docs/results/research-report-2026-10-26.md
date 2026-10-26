# Pre-H0 desk research on design v1.2: code audit, numerics, clip mechanism, design statistics, toy dry run (2026-10-26)

## Scope and constraints

**What this is.** This round tests the claims and the analysis plan of `docs/op-print-design-v1.2.md` with everything that can be done **before GPU time**. Every number below is reproducible from the scripts named in each section; outputs are committed next to the scripts.

**What it is not.** Nothing here is a Qwen3 measurement:

- There is no GPU in this environment.
- `huggingface.co`, `arxiv.org` and `alphaxiv.org` were unreachable from both the container and the fetch tool, so:
  - no Qwen3 weights, tokenizer or dataset could be loaded;
  - OP-Print (arXiv 2610.31616) and the other cited papers could not be re-read.
- The OPSD source on GitHub *was* readable.

Statements about Qwen3 logit scales, OP-Print's own scoring settings and the Qwen3 chat template therefore remain **[unverified]**, and the logit-level experiments sweep the unknown scales instead of measuring them.

**Markers** follow the design: **[verified]**, **[inference]**, **[unverified]**, **[assumption]**.

## Summary of findings and what they change

| # | Finding | Evidence | Design change (v1.3) |
|---|---|---|---|
| F1 | OPSD computes its loss at the **sampling temperature 1.1** | Code (R1) | U, P_U, m_U/r_dir and the clipped D_θ are computed at T = 1.1; Δ is scored at both T = 1 and T = 1.1 |
| F2 | The 1.7B/4B/8B scripts run a **thinking-mode teacher against a non-thinking student**. The teacher prompt also adds explicit "explore and backtrack" instructions. | Code (R1) | §1.0 adds Δ₁, the argmax-chunk index and a **no-r teacher context**. `--teacher_thinking False` becomes a secondary arm. |
| F3 | Right-padded prompts are concatenated with the generation, so in training **t = 1 is read at a pad position** for all but the longest prompt in each micro-batch | Code (R1), [inference] about the effect | Our scorer uses no mid-sequence pads; G/G-Max are also reported without t = 1 / chunk 1 |
| F4 | The training budget is **48 optimiser steps, with 3 on-policy rollouts per member** (effective batch 32) | Code (R1) | §1.1 pins effective batch 32 and `vllm_sync_frequency` |
| F5 | **Every clamped logit is pushed down** (g_a = q_a·P_U ≥ 0). The clipped loss has negative-loss optima that *abandon* teacher tokens with p ≳ clip, so q = p is only a local minimum. | Algebra + fp64 checks (R3) | §0.1 rewritten. H3 sign effects are expected at clip 0.05 already, not only at 0.01. |
| F6 | v1.2's competitor **m_U = Σ_U f is negative at essentially every correction site** and does not track trainability (Spearman ≈ 0.05–0.21 with the direct first-order KL change at clip 0.05) | R3 | m_U is replaced by **r_dir**, the first-order full-KL reduction under the clipped gradient. m_U is kept as secondary. |
| F7 | The **fp32 log1p form** is 2–8× more accurate than Barron's form at V = 151,936. The fp32 direct form is unusable below KL ≈ 1e-3. | R2 | log1p becomes the primary Δ implementation; Barron becomes the cross-check |
| F8 | The bf16-logit rounding floor follows **Σ_k p_k(1−p_k)·s_k²/12**: ≈ 1e-4 / 4e-4 / 1e-3 nats/token for top logits near 8 / 16 / 24. It is **exactly 0 when the two passes give bitwise-identical logits.** The fp32 head cuts it ~10×. | R2 | The reviewer's 3e-4 prior is confirmed for top logits in [8, 16). N1 is not even a reliable lower bound. N2 stays primary. A per-position analytic floor is added as a diagnostic. |
| F9 | The bf16 OPSD gradient has 3–8% relative error at T = 1 and 6–22% at T = 1.1 (dividing bf16 logits by T rounds them twice). At near-equal positions it is noise. At clip 1e-6, 350–700 vocabulary entries per position change clamp status. | R2 | H5 motivation strengthened. Clip 1e-6 in bf16 is numerically ill-defined, so the optional 1e-6 dose is dropped unless run in fp32. |
| F10 | The **v1.2 sign-flip placebo is anti-conservative for H1's null** when leakage is heterogeneous: 10.7–13.7% rejections at nominal 5%. HC3 OLS holds the level (4.0–7.0%). | R4 (S2) | H1's primary inference is OLS + HC3 with 16 assignment-pattern fixed effects and a spline in d. The sign-flip placebo stays as a sharp-null sanity check. |
| F11 | Gate feasibility depends on the difficulty share f_a of score variance. A variance-based post-H0 bound on split-half r is valid but **too loose to be useful**: 0.44–0.96 in the toy where the true r ≈ 0. A **staged gate after 2 split/complement pairs** stops correctly in all 9 toy arms. | R4 (S1), R5 | Staged gate: continue to 8 models only if 2r_11/(1 + r_11) ≥ 0.3 |
| F12 | Under a chunk model calibrated to OP-Print's 1.7B AUCs, H3's MDE for a gap difference is **0.007 with 8 vs 8 models and 0.016 with 2 vs 2** | R4 (S3) | The 0.2/0.01 cells are labelled "AUC-level only". The 8-model clip-0 arm stays. |
| F13 | **Toy dry run.** Clip-off leakage drops sharply under any clip (AUC 0.576 → 0.52–0.53). gap(0) − gap(0.05) = +0.038 [0.030, 0.046]. The template spike makes G-Max a chunk-1 attack at clip ≥ 0.01. \|score − median_out\| is uninformative. The θ0-teacher arm is the strongest score and flips sign at clip 1e-6. Leakage is homogeneous, so r ≈ 0. | R5 | H3: AUC-by-dose as a co-primary outcome, a restricted trend test, and the learned aggregator replacing \|·\| as the primary sign-agnostic attack. H1: one primary test per clip. |

---

## R1. OPSD source audit

Full text: `docs/verification/opsd-code-audit.md`.

**Confirmed.** All v1.2 code facts were confirmed against the source (clip semantics, β = 0 direction, no-grad fixed teacher, merge/unmerge without an fp32 base copy, script hyper-parameters, README statements). The "tinker-style loss is unclipped" item moves from [inference] to [verified]: it is a sampled-token policy gradient with no clamp. All of v1.2's arithmetic was recomputed and is correct.

**New, and design-relevant:** F1–F4 above, plus:

- **Clip semantics depend on the entry point.** `opsd_train.py` maps `jsd_token_clip ≤ 0` to `None`, but the trainer itself only checks `is not None`. Calling the trainer directly with 0.0 would therefore clamp *all* positive summands. The H3 harness must pass `None` for the clip-off arm.
- **Dataset and fields.** The data is `siyanzhao/Openthoughts_math_30k_opsd`, with fields `problem` and `solution`. `--max_length 20000` means r is not truncated.

**Why F2 matters for the attack [inference].** With `enable_thinking=False`, the student's generation starts after an empty think block. The teacher's context ends at `assistant\n` and is followed directly by the student's tokens. At t = 1 the teacher therefore expects `<think>`, a token the student essentially never emits.

That gives a large Δ₁ in every sample. It is clamped at any clip ≥ 1e-6, so it is untrained. It can make G-Max select chunk 1 regardless of membership.

The instruction text ("re-derive … exploring approaches and backtracking") and the thinking mode both raise teacher probability on reflective tokens. That is a content-independent explanation for the README's "style tokens 6–15× higher KL".

Neither component is sample-specific, so both can be learned generically. They add Δ variance that carries no membership information. R5 reproduces this mechanism in a toy.

## R2. Numerics at the Qwen3 vocabulary

**Script and outputs.** `experiments/numerics/large_vocab_numerics.py` (outputs `.json` and `.out.txt`).

**Setup.** Synthetic logits at V = 151,936, with a bulk of sd 2.5 and candidate tokens lifted to a top-logit level ∈ {8, 16, 24}. The level is swept because Qwen3's is unknown. P3/P4 use a synthetic V × 2048 LM head stored in bf16.

### P1. OPSD's loss path, low precision vs an fp64 reference on the same rounded inputs

Median relative error of the gradient w.r.t. student logits:

| regime (KL scale) | bf16, T = 1 | bf16, T = 1.1 | fp32 |
|---|---|---|---|
| correction site (≈ 0.2) | 2.8e-2 | 7.7e-2 | ≤ 6e-6 |
| correction site, clip 0.05 | 7.8e-2 | 1.7e-1 | ≤ 1.1e-5 |
| diffuse sd 0.5 (≈ 0.1) | 2.5e-2 | 6.2e-2 | ≤ 1e-5 |
| diffuse sd 0.1 (≈ 5e-3) | 8.2e-2 | 2.2e-1 | ≤ 1.2e-5 |
| near-equal (≈ 1e-5) | 44× (noise) | 27× (noise) | 6–9e-4 |

**Absolute errors.** The bf16 gradient error is ≈ 1e-3 per position at T = 1 and ≈ 3e-3 at T = 1.1. Informative positions have gradient norms of 1–4e-2.

**Why T = 1.1 is worse.** `logits / T` on bf16 logits rounds them a second time.

**Clip 1e-6 in bf16.** 350–700 vocabulary entries per position (25–30 at near-equal positions) have a different clamp status from fp64. At clip 0.05 no flips were observed. On the teacher's top token, bf16 and fp64 give the same gradient sign in only 46–61% of near-equal positions.

### P2. KL value formulas in fp32 at V = 151,936

Maximum relative error vs an fp64 log1p reference. The reference was checked against 40-digit mpmath at 2.4e-11.

| logit perturbation σ (KL) | fp32 direct | fp32 Barron −Σ p s | fp32 log1p(Σ p(expm1 d − d)) |
|---|---|---|---|
| 1e-6 (4e-13) | 2.1e6 | 0.31 | 0.066 |
| 1e-4 (3e-9) | 2.3e2 | 4.0e-3 | 5.1e-4 |
| 1e-2 (3e-5) | 0.18 | 6.3e-5 | 3.1e-5 |
| 0.1 (2.5e-3) | 5.5e-3 | 5.8e-5 | 2.8e-5 |
| 0.5 (6.8e-2) | 1.9e-4 | 5.6e-5 | 3.2e-5 |

The log1p form, which Appendix A did not run in fp32, is the most accurate at every scale. Both log-space forms are adequate for KL ≳ 1e-5.

### P3/P4. Logit-quantisation floor

**Setup.** Hidden-state pairs with controlled true Δ, scored by three pipelines and compared with the fp64 head on the unrounded hidden state:

- the HF default (bf16 head, so bf16 logits);
- OPSD's bf16 `F.kl_div` on bf16 logits;
- an fp32 head on bf16-rounded hidden states.

**Results.**

- **bf16 logits.** The mean error at true Δ > 0 is of the order of the analytic prior Σ_k p_k(1−p_k)s_k²/12, where s_k is the bf16 spacing at z_k. That is ≈ 1.3e-4 (top logit ≈ 8), ≈ 4e-4 (≈ 16) and ≈ 9e-4 (≈ 24).
  - The reviewer's ≈ 3e-4 is right for top logits in [8, 16) and about 3× too low if Qwen3's top logits sit in [16, 32).
  - Positive bias is clearest when true Δ is small. At larger Δ the per-position error is dominated by a zero-mean cross term.
- **OPSD's bf16 `F.kl_div`** errs by ±1e-3 with no consistent sign.
- **fp32 head** errs by ≈ 1–4e-5 at small Δ and ≈ 1e-4 at Δ ≈ 0.02, about 10× lower. This includes only the *final* hidden-state rounding, not accumulated trunk error, so N2 is still required.
- **Bitwise-identical logits.** At true Δ = 0 with identical hidden states, every pipeline gives **exactly zero**: identical inputs round identically.
  - Any N1-style test whose two passes are bitwise identical therefore measures nothing.
  - The floor only appears once the two contexts' logits differ by a fraction of the spacing, which is always the case in OPSD.
  - This supports v1.2's choice of N2 (an fp32 reference forward pass) as primary.
- **Aggregated to 64-token chunks (P4).** The relative bias of G's chunk mean is:
  - +44–55% (bf16 head) vs +5–10% (fp32 head) at Δ̄ ≈ 3e-4;
  - +5–36% vs 0.4–2% at Δ̄ ≈ 2e-3;
  - ≤ 6% vs ≤ 0.4% at Δ̄ ≈ 2e-2.
  G-Max's bias is similar.

**Implication.** With the fp32 head, the floor matters only if member Δ̄ falls below ≈ 1e-3 nats/token. A cheap per-position diagnostic, ε̂_t = Σ_k p_k(1−p_k)s_k²/12, can be logged by the scorer.

## R3. What the element-wise clip does (beyond §0.1)

**Script and outputs.** `experiments/clip_mechanism/clip_mechanism.py` (fp64; outputs `.json` and `.out.txt`).

**M1: clamped logits are pushed down.** For any clamped entry a (f_a > clip), g_a = q_a·P_U ≥ 0, so gradient descent never raises a clamped logit. Confirmed on 300 random positions at each of 4 clips: the minimum of g over clamped entries is 0.

**M2: the clipped objective has negative-loss optima away from the teacher.** With free logits:

- inf L = min_C [ |C|·clip + h(P_C) ], with h(P) = (1−P)·log(1−P), which has its minimum −1/e at P = 1 − 1/e;
- C is the "abandoned" set: q → 0 on C and q ∝ p elsewhere;
- one token with p_k − p_k²/2 > clip already gives L < 0 = L(q = p).

So q = p is only a local minimum.

- **Worked example (v1.2 §0.1).** Gradient descent from the uniform student reaches L = −0.1802 (the bound is −0.1803), with **q_a = 9e-4: the teacher's 0.9-probability token is abandoned.** The full KL rises from 2.38 to 5.95. Started near q = p, gradient descent stays at q = p.
- **300 random positions (V = 50).** Long gradient descent never beats the convex lower bound. It ends with L < 0 in 18 / 72 / 96 / 100% of runs and **with the full KL above its starting value in 34 / 67 / 87 / 98%**, at clip 0.2 / 0.05 / 0.01 / 1e-6.

**M3: one clipped step can raise the full KL.** The first-order change of the full KL under one clipped step is −r, with r = ⟨q − p, g⟩. r < 0 means that step *raises* the full KL. In the worked example r = −0.0045 at every clip ≤ 0.2.

**M5: the closed form for the clamped token's direction.** For clamped a:

d log q_a ∝ P_U·(Σ_i q_i² − q_a) − Σ_{i∈U} p_i q_i

It matches direct computation in 288/288 cases. q_a **rises when the student is sharp** (entropy ≤ 2 nats over V = 1000) and **falls when it is diffuse** (≥ 4 nats). That is v1.2's "can move the wrong way at high student entropy", made precise. Its magnitude is small either way: the logit itself is always pushed down (M1).

**M6: synthetic ensemble at V = 151,936.**

| T | clip | regime | share of positions with r < 0 | share with m_U < 0 |
|---|---|---|---|---|
| 1.0 | 0.05 | correction site | 80% | 100% |
| 1.1 | 0.05 | correction site | 82% | 100% |
| 1.0 | 0.05 | diffuse | 38% | 52% |
| 1.1 | 0.05 | diffuse | 20% | 28% |
| 1.1 | 0.01 | diffuse | 68% | 62% |
| 1.1 | 1e-6 | any | 30–100% | 100% |

- **Temperature changes the clamped set substantially.** For diffuse positions at clip 0.05, the median clamped share of KL is 105% at T = 1 and **0% at T = 1.1**.
- **m_U tracks r poorly.** Spearman rank correlations with r at clip 0.05 are m_U +0.05 / +0.21 (T = 1 / 1.1), m_U⁺ (positive unclamped mass) +0.22 / +0.36, and the clipped loss +0.40 / +0.45. At smaller clips they turn negative.

**Implications.**

1. **H1's mechanistic competitor.** v1.2's m_U is not a mass: it is negative wherever the teacher's preferred token is clamped. The proposed replacement is r_dir(x) = mean_t ⟨q_t − p_t, g_t⟩ at T = 1.1. It is the first-order reduction of the *full* θ0-teacher KL that one clipped step would make directly, under an isotropic-NTK **[assumption]**. m_U and m_U⁺ are kept as secondary.
2. **H3 sign effects.** At clip 0.05, the direct effect at the high-Δ sites that G-Max reads points *away* from the teacher. Members can therefore end up with *higher* Δ at those sites. Sign-agnostic attacks are needed at clip 0.05, not only at 0.01.
3. **OPSD's own clip of 1e-6.** The README's 1e-6/1e-7 setting optimises an objective whose minimisers concentrate the student on a set of teacher mass 1/e. A sharpening, entropy-minimisation-like pressure **[inference]**. This bears on interpreting the "non-thinking" results. It is not needed for the minimal paper.

## R4. Design statistics

**Script and outputs.** `experiments/design_sim/design_sim.py` (outputs `.json` and `.out.txt`).

**Model.** The per-model log score is l_m(x) = a(x) + g_m − s_m·b(x)·I_m(x) + e_m(x), on the 4 splits + complements design (N+ = 1000).

### S1. Reliability gate

Closed forms: AUC = Φ(μ_b / √(2σ_a² + τ_b² + 2σ_e²)) and r_raw = τ_b² / (τ_b² + σ_e²). Monte Carlo agrees: r = 0.723 vs 0.723 and 0.203 vs 0.204, with AUC 0.594 reproduced.

At the paper's G-Max AUC of 0.594, the raw split-half r (8-model Spearman–Brown in brackets) is:

| difficulty share f_a | CV(leakage) 0.5 | 1 | 2 |
|---|---|---|---|
| 0.5 | 0.06 [0.10] | 0.20 [0.34] | 0.62 [0.77] |
| 0.8 | 0.13 [0.23] | 0.44 [0.61] | infeasible |
| 0.9 | 0.25 [0.40] | 0.72 [0.84] | infeasible |
| 0.95 | 0.44 [0.61] | infeasible | infeasible |

**Reading.** The gate (raw r ≥ 0.3) passes only if sample difficulty dominates the in-vs-out score variance, or leakage is strongly heterogeneous.

**Post-H0 bound (new).** Four inputs, all available before the 7 further models are trained:

- σ_tot² = Var(ℓ | in) + Var(ℓ | out), on H0's model (under normality this equals (μ̂_b/Φ⁻¹(AUC))², with μ̂_b the mean out − in difference of log Δ̄);
- σ_a² from θ0's between-sample variance of log Δ̄, net of rollout noise. This assumes training leaves the difficulty variance unchanged **[assumption]**;
- σ_roll², the rollout part of σ_e, from two independent K = 8 rollout sets per sample;
- R = σ_tot² − 2σ_a².

Then r_raw ≤ (R − 2σ_roll²)/(R − σ_roll²), because σ_e ≥ σ_roll.

**This bound is valid but loose, so it is not used as the gate.** When rollout noise is a small part of σ_e it barely binds: 0.88 against a true 0.20 in the table above. In the toy (R5) it gave 0.44–0.96 where the true r ≈ 0.

**v1.3 therefore uses a staged gate.** After 2 split/complement pairs (4 models), compute the split-half r_11 across the pairs (1 in / 1 out per half) and continue only if the Spearman–Brown projection 2r_11/(1 + r_11) ≥ 0.3. That is the same quantity the 8-model gate tests, at half the cost.

### S2. H1 inference

Rejection rates at nominal 5%, one-sided, 300 replications each (MC SE ≈ 0.013):

| scenario (H1 null unless stated) | v1.2 sign-flip placebo | v1.1 any-4-of-8 | OLS HC3 | HC3 + pattern FE + spline(d) | Freedman–Lane |
|---|---|---|---|---|---|
| sharp null (no leakage) | 0.047 | 0.050 | 0.043 | 0.047 | 0.050 |
| homogeneous leakage | 0.057 | 0.060 | 0.067 | 0.063 | 0.063 |
| heterogeneous leakage (CV 1.5) | **0.107** | **0.127** | 0.040 | 0.040 | 0.043 |
| + model offsets/scales | **0.127** | **0.140** | 0.070 | 0.070 | 0.077 |
| + heteroscedastic noise | **0.083** | **0.090** | 0.047 | 0.043 | 0.063 |
| + leakage nonlinear in d | **0.137** | **0.147** | 0.047 | 0.047 | 0.070 |
| alternative β_c = 0.15 | 0.883 | 0.893 | 0.770 | 0.753 | 0.797 |

**Why the placebo fails.** The within-pair sign flip is exact for the *sharp* null "membership has no effect". H1's null is different: "c adds nothing to d", with leakage present. Under sign flips, a sample's leakage b(x) enters only as ±b(x), with variance contribution b²/4. The real v carries the leakage heterogeneity that is unrelated to c. When CV(b) ≳ 0.6, the placebo distribution is too narrow.

**Recommendation.** H1's primary test is OLS with HC3 standard errors, 16 assignment-pattern fixed effects (which absorb model-level offsets exactly) and a pre-registered cubic spline in d. Freedman–Lane is a robustness check. The sign-flip placebo is kept only as the "c does not predict v_placebo" sanity check.

### S3. G vs G-Max and the H3 gap

**Setup.** A chunk-level model: 16 chunks, K = 8, leakage on a sample's top-2 chunks ("sparse") or on all chunks ("diffuse"). It is calibrated to OP-Print 1.7B: the sparse arm gives G 0.583 and G-Max 0.603, against the paper's 0.579 and 0.594.

**Gap and its noise.**

- Sparse imprint: gap +0.0195. Diffuse imprint, matched on G: gap −0.0073.
- SD of the gap is 0.0055 for one model and 0.0017 for the mean of 8 models.
- Within one model, OP-Print's 0.015 gap would then be ≈ 2.7 SD. That holds only if G and G-Max are as correlated as in this model. Paired DeLong on the real scores decides.

**H3 MDE (80% power, two-sided 5%).** 0.0068 with 8 vs 8 models and 0.0156 with 2 vs 2. The 0.2/0.01 cells can detect only gap changes of ≳ 0.016.

**Chunk-1 spike.** A membership-independent spike in chunk 1 lowers G-Max's AUC (0.602 → 0.594) only once it is ≈ 10× the median chunk Δ. Dropping chunk 1 restores it (0.601). Whether Qwen3's Δ₁ is that large is an empirical question for §1.0.

## R5. Toy OPSD dry run

**Scripts.** `experiments/toy_opsd/`:

- `toy_opsd.py` (simulator), `run_all.sh` (all arms), `analyze.py` → `results.md`/`.json`;
- `imprint_by_class.py`, `gate_bound.py`, `staged_gate.py`;
- raw per-model scores in `runs/` and `runs_lowcap/`.

**What it is.** A small, fully specified system with OPSD's *mechanics*. It is **not** a model of Qwen3. Its purpose is (a) to execute the pre-registered analysis end to end before GPU time, and (b) to check that the mechanisms in R1/R3 survive in a shared-parameter network trained on-policy.

### Setup

**Base model and world.**

- θ0 is an MLP language model (V = 64, 128-token sequences) pre-trained on a synthetic process with three components.
- **Correction sites.** After a trigger token comes a sample-specific answer. Answers are partly predictable from the question (64-dim features, κ = 1). With r the teacher gives the correct answer probability 0.88. The student gets 0.27 (chance 0.125).
- **Instruction asymmetry.** A "reflective" flag raises style-token mass from 0.03 to 0.12.
- **Template asymmetry.** A "thinking" flag makes the first token THINK, with probability 0.98.
- Per-token θ0 KL between the contexts: 6.0 nats at t = 0, 1.18 at answer positions, 0.09 elsewhere.

**OPSD training.**

- LoRA rank 16 on every linear layer (α = 2r).
- On-policy rollouts at T = 1.1, top-k 20, top-p 0.95.
- OPSD loss at T = 1.1 with an element-wise clamp.
- Adam, max grad norm 0.1, 32 prompts × 1 rollout per step, 3 epochs = 48 steps.

**Design and scoring.**

- 4 splits + complements (8 models per arm), with the same splits in every arm.
- K = 8 scoring rollouts, 16-token chunks (8 per rollout).

**Calibration and arms.**

- **lr = 3e-3** was chosen so that the clip-off arm matches OP-Print's AUC level. This uses the attack AUC, which the real study forbids; it is a toy-only calibration.
- Arms:
  - clip ∈ {off, 0.2, 0.05, 0.01, 1e-6};
  - no-template ("nothink") at {off, 0.05};
  - lr 3e-4 at {off, 0.05}. This is the smallest lr meeting the design's f_rm ∈ [0.2, 0.5] rule at clip 0.05;
  - a low-capacity configuration (rank 4, 16-dim questions) at clip off.

### Results

**AUC by arm** (mean of 8 models; bootstrap 95% CI):

| arm | f_rm (train) | G | G-Max | G-Max w/o chunk 1 | learned aggregator | H1′ S_θ − S_θ0 | H1′ D_θ (θ0 teacher) | \|G − med_out\| |
|---|---|---|---|---|---|---|---|---|
| clip off | +0.68 | 0.576 [0.568, 0.583] | 0.608 [0.599, 0.617] | 0.606 | 0.630 | 0.620 | **0.640** | 0.489 |
| clip 0.2 | +0.42 | 0.533 | 0.526 | 0.542 | 0.558 | 0.551 | 0.554 | 0.497 |
| clip 0.05 | −0.41 | 0.523 [0.515, 0.530] | 0.517 [0.508, 0.528] | 0.532 | 0.536 | 0.536 | 0.536 | 0.494 |
| clip 0.01 | −6.3 | 0.525 | 0.511 | 0.527 | 0.548 | 0.544 | 0.517 | 0.495 |
| clip 1e-6 | −83 | 0.502 | 0.501 | 0.503 | 0.500 | 0.504 | **0.480 [0.469, 0.489]** | 0.507 |
| nothink, off | +0.59 | 0.579 | 0.613 | 0.611 | 0.640 | 0.627 | 0.643 | 0.491 |
| nothink, 0.05 | −0.55 | 0.521 | 0.532 | 0.530 | 0.541 | 0.534 | 0.530 | 0.491 |
| lr 3e-4, off | +0.64 | 0.517 | 0.528 | 0.529 | 0.548 | 0.530 | 0.539 | 0.497 |
| lr 3e-4, 0.05 | +0.45 | 0.506 [0.501, 0.511] | 0.508 | 0.508 | 0.520 | 0.512 | 0.510 | 0.504 |
| low capacity, off (lr 3e-4) | +0.56 | 0.505 | 0.506 | 0.507 | 0.501 | 0.508 | 0.510 | 0.504 |

**H3 gap** (G-Max − G):

| arm | gap | without chunk 1 |
|---|---|---|
| clip off | +0.032 [0.028, 0.037] | +0.033 |
| clip 0.2 | −0.007 [−0.015, 0.001] | +0.012 [0.007, 0.016] |
| clip 0.05 | −0.006 [−0.014, 0.003] | +0.011 [0.006, 0.015] |
| clip 0.01 | −0.015 [−0.023, −0.006] | −0.001 |
| clip 1e-6 | 0.000 | 0.000 |

Paired comparisons (same samples and splits):

- gap(off) − gap(0.05) = **+0.038 [0.030, 0.046]**;
- without the template: +0.023 [0.018, 0.029];
- at lr 3e-4: +0.008 [0.001, 0.015].

The Spearman trend over the 5 doses is −0.3.

**Where the imprint lands** (`imprint_by_class.py`; model 0 of each dose; log(mean Δ_out / mean Δ_in) by position class):

| clip | answer positions (4.6%) | t = 0 | style / other | log q(correct) θ − θ0 at answer positions, members / non-members |
|---|---|---|---|---|
| off | **+0.174** | +0.105 (Δ₁ ≈ 0.04) | −0.007 / −0.006 | **+0.16** / −0.11 |
| 0.2 | +0.049 | −0.005 (Δ₁ ≈ 5.4) | −0.001 / −0.001 | −0.06 / −0.12 |
| 0.05 | +0.036 | −0.009 (Δ₁ ≈ 5.2) | +0.002 / 0.000 | −0.04 / −0.10 |
| 0.01 | +0.017 | −0.013 | style share collapses 0.115 → 0.001 | −0.05 / −0.09 |
| 1e-6 | (no answers emitted) | — | 99% "other" tokens | — |

**§1.0 diagnostic.**

| arm | argmax chunk is chunk 1 | argmax token is an answer |
|---|---|---|
| clip off | 8% | 99% |
| clip ≥ 0.01 (template on) | 97–98% | 2–4% |
| nothink | 9% | 99% |

The student emits THINK first in 98.5% of rollouts at clip off and in ≤ 0.2% at any clip.

**H1 and gates.**

- **Reliability.** Split-half reliability of v is −0.05 to +0.05 in every arm. The ≈ 30 competitor tests produce a few |t| ≈ 2–2.9 by chance.
- **Variance-based post-H0 bound.** Valid but loose: 0.44–0.96, and undefined in two arms where θ0's difficulty variance exceeds the trained model's total variance.
- **Staged gate.** Correct "stop" in all 9 arms after 4 models: projected r_22 ≤ 0.09.

### Lessons for the design

1. **The clip strongly shapes leakage itself, not only the G-Max − G gap.**
   - Any clip ≤ 0.2 cuts the clip-off AUC from 0.58–0.61 to 0.51–0.53. At 1e-6 it is 0.50.
   - H3 gains a co-primary outcome: the AUC level of the strongest pre-registered attack as a function of dose, reported with utility. This is the "clip as an empirical defense" question.
   - In OP-Print the clip-0.05 AUC is 0.59, much higher than the toy's 0.52. Real correction sites probably include many entries whose summands sit below 0.05, or indirect routes are stronger at LLM scale **[inference]**.
2. **The sparse-imprint prediction of H3 holds in the toy.** The membership imprint sits at the correction sites (answer positions +0.174), and the gap widens as clip → 0 (+0.038). The paired-bootstrap primary contrast has ample power at this effect size.
3. **The template spike can hide the G-Max advantage.**
   - At clip ≥ 0.01 the THINK mismatch is not learned. Δ₁ ≈ 5 nats, and chunk 1 is the argmax in 97–98% of rollouts.
   - G-Max then reads only t = 1 and is no better than G: the gap is −0.006 to −0.015, significant only at 0.01. Dropping chunk 1 restores a significant positive gap (+0.011 at clip 0.05).
   - Unclipped, the student *learns to emit the teacher's template token*, in 98.5% of rollouts. This supports the §1.0 additions and the template arm. It also suggests one concrete way the clip "stabilises" OPSD with the default thinking-teacher setup **[inference]**.
4. **Trend tests across doses mislead when leakage vanishes.** Spearman = −0.3 here, because clip 1e-6 has no leakage at all. The primary stays the paired gap(0) − gap(0.05). The trend is secondary, restricted to doses whose G or G-Max AUC CI excludes 0.5, and also reported for gap/(AUC_G − 0.5).
5. **R3's anti-teacher dynamics survive in a network.**
   - The θ0-teacher residual on training samples *grows* under clipping: f_rm +0.68 / +0.42 / −0.41 / −6.3 / −83 from clip off down to 1e-6.
   - The correct-answer probability falls for all samples under any clip.
   - At clip 0.01 the student stops emitting style tokens, whose summands of ≈ 0.02 lie between 0.01 and 0.05. At 1e-6 it collapses.
   - Membership then shows with the **opposite sign** in the training-objective arm: D_θ AUC 0.480 [0.469, 0.489]. AUC < 0.5 is to be reported, not folded.
6. **|score − median_out| is not a usable sign-agnostic attack.** It is uninformative in every arm, including where G gives 0.58, because it detects dispersion, not a location shift. The learned aggregator on the sorted CRN chunk vector is the sign-agnostic attack that works (≥ G-Max in all clipped arms). Two-sided AUC CIs cover single scores.
7. **H1′ arm 3 (θ0 teacher, D_θ) is the strongest single score** at clip off (0.640, vs OP-Print's G-Max 0.608). Δ_θ − D_θ has AUC < 0.5 (0.451). The adapter's effect on the teacher side works against OP-Print's Δ_θ, which therefore under-uses the signal.
8. **f_rm is not a proxy for leakage.**
   - The f_rm-rule lr (3e-4) gives f_rm +0.45 at clip 0.05 with barely detectable leakage (G 0.506 [0.501, 0.511]).
   - The low-capacity configuration has f_rm +0.56 with no leakage (0.505).
   - The template and style components are learned generically.
   - Supports the f_rm^r (no-r) caveat in v1.3 §1.1.
9. **AUC ≈ 0.6 does not imply a testable H1.** Leakage in the toy is homogeneous across samples, so v has no reliable per-sample variance. The gate correctly stops H1.
   - The variance-based post-H0 bound is too loose to act as a gate; the staged gate after 2 pairs works.
   - With ≈ 30 secondary tests, chance |t| ≈ 2–3 appear. H1 keeps **one primary test per clip** (c at clip 0 and at clip 0.05, with Holm correction). Competitors are secondary.

**What the toy cannot show.**

- Qwen3's real clamp structure.
- LLM-scale spillover.
- Whether style tokens dominate the argmax: the toy's style Δ is diffuse and never the maximum.
- The real step size: the toy lr is calibrated on AUC.

The toy validates the analysis code and the mechanisms. It does not predict Qwen3's numbers.
