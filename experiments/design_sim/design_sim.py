"""Statistical checks of the v1.2 analysis plan by simulation (numpy; no model training).

Design: N+ = 1000 samples, 4 random splits + complements (8 models), every sample 4 in / 4 out.
Per-model log score  l_m(x) = a(x) + g_m - s_m b(x) I_m(x) + e_m(x)   (log of the mean Delta over K rollouts)
  a: difficulty (sd sigma_a);  b: per-sample leakage (mean mu_b, heterogeneity tau_b);  e: rollout + seed noise (sd sigma_e);
  g_m, s_m: model-level offset and leakage scale.  AUC is rank-based, so the log scale does not change it.

S1  Reliability gate (raw split-half r >= 0.3). Closed forms, checked by Monte Carlo:
      per-model AUC = Phi(mu_b / sqrt(2 sigma_a^2 + tau_b^2 + 2 sigma_e^2)),  r_raw = tau_b^2 / (tau_b^2 + sigma_e^2).
    Feasibility map over (difficulty share f_a, leakage CV) at the paper's G-Max AUC 0.594, and a post-H0 upper bound on r
    computable from H0's score variances plus theta0 measurements of sigma_a and the rollout-noise floor of sigma_e.
S2  H1 inference: type-I error and power of five tests of "c adds to d" under several nulls, including v1.2's
    within-pair sign-flip placebo and v1.1's any-4-of-8 relabelling.
S3  Chunk-level model of G vs G-Max: Monte Carlo SD of the per-model and 8-model-pooled G-Max - G AUC gap, the MDE of
    H3's gap(clip 0) - gap(clip 0.05) for 8 vs 8 and 2 vs 2 models, and the effect of a membership-independent
    spike in chunk 1 (the thinking-template artefact, docs/verification/opsd-code-audit.md §2.2).
Run: python3 experiments/design_sim/design_sim.py  (CPU, ~10-20 min)
"""
import json, sys, time
import numpy as np
from scipy.stats import norm, rankdata

OUT = {}
t0 = time.time()
rng = np.random.default_rng(20261026)


def log(*a):
    print(*a, flush=True)


def auc(pos, neg):
    """P(score_pos > score_neg) via ranks (ties count 1/2)."""
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def complement_design(N, rng):
    """membership matrix I[m, x] for 4 splits + complements; pattern id per sample (16 patterns)."""
    I = np.zeros((8, N), bool)
    for s in range(4):
        perm = rng.permutation(N)
        I[2 * s, perm[: N // 2]] = True
        I[2 * s + 1, perm[N // 2:]] = True
    pattern = (I[0::2].astype(int) * (2 ** np.arange(4))[:, None]).sum(0)
    return I, pattern


# ======================================================================== S1: reliability gate
log("S1: reliability gate feasibility at per-model AUC 0.594 (paper's 1.7B G-Max); mu_b = 1 w.l.o.g.")
z = norm.ppf(0.594)
stot2 = 1 / z ** 2
s1 = []
for fa in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98):
    row = dict(f_a=fa)
    for cv in (0.5, 1.0, 2.0, 3.0):
        R = stot2 * (1 - fa)                       # tau_b^2 + 2 sigma_e^2
        tau2 = cv ** 2
        se2 = (R - tau2) / 2
        if se2 <= 0:
            row[f"cv{cv:g}"] = None
            continue
        r = tau2 / (tau2 + se2)
        row[f"cv{cv:g}"] = dict(r_raw=r, r_8model=2 * r / (1 + r), sigma_e=se2 ** .5, sigma_a=(stot2 * fa / 2) ** .5)
    s1.append(row)
log("  raw split-half r [and 8-model Spearman-Brown]; '-' = impossible (heterogeneity alone exceeds the residual variance)")
log("  f_a = share of the in-vs-out score variance due to sample difficulty; CV = tau_b / mu_b")
log("  f_a     " + "".join(f"{'CV=' + str(cv):>18s}" for cv in (0.5, 1.0, 2.0, 3.0)))
for row in s1:
    cells = []
    for cv in (0.5, 1.0, 2.0, 3.0):
        c = row[f"cv{cv:g}"]
        cells.append(f"{'-':>18s}" if c is None else f"{c['r_raw']:>8.2f} [{c['r_8model']:.2f}]".rjust(18))
    log(f"  {row['f_a']:<6g}" + "".join(cells))
OUT["S1_table"] = s1


def simulate_v(N, sa, mub, taub, se, rng, I=None, ga=0.0, ss=0.0):
    I, pattern = complement_design(N, rng) if I is None else I
    a = rng.normal(0, sa, N)
    b = mub + rng.normal(0, taub, N)
    g = rng.normal(0, ga, 8)[:, None]
    s = 1 + rng.normal(0, ss, 8)[:, None]
    l = a[None] + g - s * b[None] * I + rng.normal(0, se, (8, N))
    return l, I, pattern, a, b


log("\n  Monte Carlo check (N+ = 1000, 200 replications) of two cells:")
mc = []
for fa, cv in ((0.9, 1.0), (0.5, 1.0)):
    c = [r for r in s1 if r["f_a"] == fa][0][f"cv{cv:g}"]
    rs, aucs = [], []
    for _ in range(200):
        l, I, pattern, a, b = simulate_v(1000, c["sigma_a"], 1.0, cv, c["sigma_e"], rng)
        aucs.append(np.mean([auc(-l[m, I[m]], -l[m, ~I[m]]) for m in range(8)]))
        # split-half: pairs {0,1} vs {2,3}, each half has 2 in / 2 out per sample
        h = []
        for pairs in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
            vv = []
            for ps in pairs:
                ms = [2 * p for p in ps] + [2 * p + 1 for p in ps]
                L, M = l[ms], I[ms]
                vv.append((L * ~M).sum(0) / (~M).sum(0) - (L * M).sum(0) / M.sum(0))
            h.append(np.corrcoef(vv[0], vv[1])[0, 1])
        rs.append(np.mean(h))
    mc.append(dict(f_a=fa, cv=cv, analytic_r=c["r_raw"], mc_r=float(np.mean(rs)), mc_auc=float(np.mean(aucs))))
    log(f"   f_a={fa}, CV={cv}: analytic r = {c['r_raw']:.3f}, MC r = {np.mean(rs):.3f} (sd {np.std(rs):.3f}); MC AUC = {np.mean(aucs):.3f} (target 0.594)")
OUT["S1_mc"] = mc

log("\n  Post-H0 upper bound: sigma_tot^2 = Var(l|in) + Var(l|out) on H0's model; sigma_a from theta0 (assumes training")
log("  leaves difficulty variance unchanged); sigma_roll = rollout part of sigma_e (two independent K=8 rollout sets):")
log("    R = sigma_tot^2 - 2 sigma_a^2;  r_raw <= (R - 2 sigma_roll^2) / (R - sigma_roll^2)   (sigma_e >= sigma_roll)")
ub = []
for fa in (0.5, 0.8, 0.9):
    for roll_share in (0.25, 0.5, 1.0):     # sigma_roll^2 as a share of the sigma_e^2 that CV=1 would imply
        c = [r for r in s1 if r["f_a"] == fa][0]["cv1"]
        R = stot2 * (1 - fa); sr2 = roll_share * c["sigma_e"] ** 2
        ub.append(dict(f_a=fa, roll_share=roll_share, r_upper=(R - 2 * sr2) / (R - sr2), r_true_cv1=c["r_raw"]))
        log(f"    f_a={fa}, sigma_roll^2 = {roll_share:.2f} x sigma_e^2: bound {(R - 2*sr2)/(R - sr2):.2f} (true r at CV=1: {c['r_raw']:.2f})")
OUT["S1_upper_bound"] = ub
log(f"[{time.time()-t0:.0f}s]")

# ======================================================================== S2: H1 inference
log("\nS2: H1 test validity. Tests of beta_c > 0 (one-sided, alpha = 0.05) in v ~ c + d:")
log("  T1 v1.2 within-pair sign-flip placebo; T2 v1.1 any-4-of-8 placebo; T3 OLS HC3;")
log("  T4 OLS HC3 + 16 assignment-pattern fixed effects + cubic spline in d; T5 Freedman-Lane permutation (linear d)")


def ols_fit(X, y):
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    return beta, XtX_inv


def hc3_t(X, y, j):
    beta, XtX_inv = ols_fit(X, y)
    res = y - X @ beta
    H = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat = (X * (res / (1 - H))[:, None] ** 2).T @ X
    V = XtX_inv @ meat @ XtX_inv
    return beta[j] / np.sqrt(V[j, j])


def spline_basis(d, knots):
    cols = [d, d ** 2, d ** 3] + [np.clip(d - k, 0, None) ** 3 for k in knots]
    B = np.column_stack(cols)
    return (B - B.mean(0)) / B.std(0)


def run_s2(scn, reps, n_perm=200):
    rej = {k: 0 for k in ("T1_signflip", "T2_any4of8", "T3_hc3", "T4_hc3_fe_spline", "T5_freedman_lane")}
    N = 1000
    for _ in range(reps):
        I, pattern = complement_design(N, rng)
        d = rng.normal(0, 1, N)                                   # standardised difficulty proxy (observed)
        a = scn["sigma_a"] * (d * 0.9 + rng.normal(0, np.sqrt(1 - 0.81), N))
        c = scn["rho_cd"] * d + np.sqrt(1 - scn["rho_cd"] ** 2) * rng.normal(0, 1, N)
        b = scn["mu_b"] * (1 + scn["beta_d"] * d + scn["beta_nl"] * (d ** 2 - 1) + scn["beta_c"] * c) \
            + rng.normal(0, scn["tau_u"], N)
        se = scn["sigma_e"] * np.exp(scn["eta"] * c)
        g = rng.normal(0, scn["sigma_g"], 8)[:, None]
        s = 1 + rng.normal(0, scn["sigma_s"], 8)[:, None]
        l = a[None] + g - s * b[None] * I + rng.normal(0, 1, (8, N)) * se[None]
        v = (l * ~I).sum(0) / 4 - (l * I).sum(0) / 4
        X = np.column_stack([np.ones(N), c, d])
        beta, XtX_inv = ols_fit(X, v)
        w = (XtX_inv @ X.T)[1]                                    # beta_c = w . v
        # T1: within-pair sign flips: v_p(x) = sum_s eps_s(x) (l_out - l_in)_s / 4
        pairdiff = np.stack([np.where(I[2 * k], l[2 * k + 1] - l[2 * k], l[2 * k] - l[2 * k + 1]) for k in range(4)])  # out - in
        eps = rng.choice([-1.0, 1.0], size=(n_perm, 4, N))
        vp = (eps * pairdiff[None]).sum(1) / 4
        rej["T1_signflip"] += beta[1] > np.quantile(vp @ w, 0.95)
        # T2: any 4 of 8 models labelled "in", independently per sample
        sel = np.argsort(rng.random((n_perm, 8, N)), axis=1) < 4
        vp2 = (l[None] * ~sel).sum(1) / 4 - (l[None] * sel).sum(1) / 4
        rej["T2_any4of8"] += beta[1] > np.quantile(vp2 @ w, 0.95)
        # T3: HC3
        rej["T3_hc3"] += hc3_t(X, v, 1) > norm.ppf(0.95)
        # T4: HC3 + pattern FE + spline in d
        FE = (pattern[:, None] == np.arange(1, 16)[None]).astype(float)
        X4 = np.column_stack([np.ones(N), c, spline_basis(d, np.quantile(d, [0.25, 0.5, 0.75])), FE])
        rej["T4_hc3_fe_spline"] += hc3_t(X4, v, 1) > norm.ppf(0.95)
        # T5: Freedman-Lane: residuals of v ~ 1 + d permuted, added back, refit c coefficient
        Xd = X[:, [0, 2]]
        bd, _ = ols_fit(Xd, v)
        fit, res = Xd @ bd, v - Xd @ bd
        perm_idx = np.argsort(rng.random((n_perm, N)), axis=1)
        vperm = fit[None] + res[perm_idx]
        rej["T5_freedman_lane"] += beta[1] > np.quantile(vperm @ w, 0.95)
    return {k: v / reps for k, v in rej.items()}


base = dict(sigma_a=2.0, mu_b=0.5, tau_u=0.0, sigma_e=0.8, beta_d=0.0, beta_nl=0.0, beta_c=0.0, rho_cd=0.3,
            eta=0.0, sigma_g=0.0, sigma_s=0.0)
scenarios = {
    "N0 sharp null (no leakage)": dict(mu_b=0.0),
    "N1 homogeneous leakage": dict(),
    "N2 heterogeneous leakage (CV 1.5) indep. of c": dict(tau_u=0.75),
    "N3 = N2 + model offsets and scales": dict(tau_u=0.75, sigma_g=0.3, sigma_s=0.3),
    "N4 = N2 + noise heteroscedastic in c": dict(tau_u=0.75, eta=0.4),
    "N5 = N2 + leakage nonlinear in d (rho_cd 0.5)": dict(tau_u=0.75, rho_cd=0.5, beta_nl=0.4, beta_d=0.3),
    "A1 alternative beta_c = 0.15 (+ N2)": dict(tau_u=0.75, beta_c=0.15),
    "A2 alternative beta_c = 0.30 (+ N2)": dict(tau_u=0.75, beta_c=0.30),
}
s2 = {}
for name, over in scenarios.items():
    scn = {**base, **over}
    res = run_s2(scn, reps=300)
    s2[name] = dict(params=scn, rejection=res)
    log(f"  {name:48s} " + "  ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in res.items()))
    l, I, *_ = simulate_v(1000, scn["sigma_a"], scn["mu_b"], scn["tau_u"], scn["sigma_e"], rng)
    s2[name]["per_model_auc"] = float(np.mean([auc(-l[m, I[m]], -l[m, ~I[m]]) for m in range(8)]))
OUT["S2"] = s2
log("  (MC SE of a 5% rate with 300 replications ≈ 0.013; per-model AUC in these scenarios ≈ "
    f"{np.mean([v['per_model_auc'] for v in s2.values()]):.3f})")
log(f"[{time.time()-t0:.0f}s]")

# ======================================================================== S3: G vs G-Max and the H3 gap
log("\nS3: chunk-level model. Per rollout: 16 chunk-mean Deltas = exp(a + alpha_j + noise); leakage multiplies chunk j by")
log("    (1 - lam_j) for members. 'sparse': leakage on the sample's top-2 chunks only; 'diffuse': equal on all chunks.")


def chunk_scores(N, I, prm, rng, spike=0.0):
    K, J = 8, 16
    a = rng.normal(0, prm["sa"], N)
    alpha = rng.normal(0, prm["salpha"], (N, J))                  # persistent chunk structure of a sample
    top = np.argsort(-alpha, axis=1)[:, :2]
    lam = np.zeros((N, J))
    if prm["mode"] == "sparse":
        np.put_along_axis(lam, top, prm["lam"], axis=1)
    else:
        lam[:] = prm["lam_diffuse"]
    hetero = np.exp(rng.normal(0, prm["sb"], N))                   # per-sample leakage scale
    lamx = np.clip(lam * hetero[:, None], 0, 0.95)
    G = np.zeros((8, N)); GM = np.zeros((8, N)); GMx1 = np.zeros((8, N))
    for m in range(8):
        seed_noise = rng.normal(0, prm["sseed"], N)
        eta = rng.normal(0, prm["seta"], (K, N, J))
        D = np.exp(a[None, :, None] + alpha[None] + seed_noise[None, :, None] + eta)
        D = D * (1 - lamx[None] * I[m][None, :, None])
        if spike:
            D[:, :, 0] += spike                                       # membership-independent chunk-1 spike
        G[m] = -D.mean(2).mean(0)
        GM[m] = -D.max(2).mean(0)
        GMx1[m] = -D[:, :, 1:].max(2).mean(0)
    return G, GM, GMx1


def gap_stats(G, GM, I):
    aG = np.array([auc(G[m, I[m]], G[m, ~I[m]]) for m in range(8)])
    aM = np.array([auc(GM[m, I[m]], GM[m, ~I[m]]) for m in range(8)])
    return aG, aM


# calibrated so the sparse arm reproduces the paper's 1.7B clip-0.05 numbers (G 0.579, G-Max 0.594); diffuse arm matched on G
prm_sparse = dict(mode="sparse", sa=0.5, salpha=0.8, seta=1.0, sseed=0.15, sb=0.8, lam=0.40)
prm_diffuse = dict(mode="diffuse", sa=0.5, salpha=0.8, seta=1.0, sseed=0.15, sb=0.8, lam_diffuse=0.105)
s3 = {}
for name, prm in (("sparse imprint", prm_sparse), ("diffuse imprint", prm_diffuse)):
    gaps, aGs, aMs, pooled = [], [], [], []
    for _ in range(150):
        I, _ = complement_design(1000, rng)
        G, GM, _ = chunk_scores(1000, I, prm, rng)
        aG, aM = gap_stats(G, GM, I)
        aGs.append(aG.mean()); aMs.append(aM.mean()); gaps.append(aM[0] - aG[0]); pooled.append((aM - aG).mean())
    s3[name] = dict(params=prm, auc_G=float(np.mean(aGs)), auc_GMax=float(np.mean(aMs)),
                    gap_mean=float(np.mean(pooled)), sd_gap_single_model=float(np.std(gaps)), sd_gap_pooled8=float(np.std(pooled)))
    log(f"  {name:16s}: AUC G {np.mean(aGs):.3f}, G-Max {np.mean(aMs):.3f}; gap {np.mean(pooled):+.4f}; "
        f"SD(gap): one model {np.std(gaps):.4f}, mean of 8 models {np.std(pooled):.4f}")
# H3 MDE: two arms with independent training (same splits) -> SD of the difference of pooled gaps
sd8 = np.hypot(s3["sparse imprint"]["sd_gap_pooled8"], s3["diffuse imprint"]["sd_gap_pooled8"])
sd2 = np.hypot(s3["sparse imprint"]["sd_gap_single_model"], s3["diffuse imprint"]["sd_gap_single_model"]) / np.sqrt(2)
mde = lambda sd: (norm.ppf(0.975) + norm.ppf(0.8)) * sd
log(f"  H3 MDE (80% power, two-sided 5%) for gap(arm 1) - gap(arm 2): 8 vs 8 models {mde(sd8):.4f}; 2 vs 2 models {mde(sd2):.4f}")
log(f"  (true difference between the two arms here: {s3['sparse imprint']['gap_mean'] - s3['diffuse imprint']['gap_mean']:+.4f};"
    " same-split pairing across arms is ignored, which is conservative only if the arms' gap errors are not negatively correlated)")
OUT["S3"] = s3
OUT["S3_mde"] = dict(sd_diff_8v8=float(sd8), mde_8v8=float(mde(sd8)), sd_diff_2v2=float(sd2), mde_2v2=float(mde(sd2)))

log("\n  Chunk-1 spike (membership-independent, e.g. the template mismatch at t = 1), sparse imprint:")
s3s = []
for spike in (0.0, 1.0, 3.0, 10.0):
    aGs, aMs, aMx = [], [], []
    for _ in range(40):
        I, _ = complement_design(1000, rng)
        G, GM, GMx1 = chunk_scores(1000, I, prm_sparse, rng, spike=spike)
        aG, aM = gap_stats(G, GM, I)
        aGs.append(aG.mean()); aMs.append(aM.mean())
        aMx.append(np.mean([auc(GMx1[m, I[m]], GMx1[m, ~I[m]]) for m in range(8)]))
    s3s.append(dict(spike=spike, auc_G=float(np.mean(aGs)), auc_GMax=float(np.mean(aMs)), auc_GMax_excl_chunk1=float(np.mean(aMx))))
    log(f"   spike={spike:<4g} (median chunk Delta ≈ 1): AUC G {np.mean(aGs):.3f}, G-Max {np.mean(aMs):.3f}, G-Max without chunk 1 {np.mean(aMx):.3f}")
OUT["S3_spike"] = s3s
json.dump(OUT, open(sys.argv[1] if len(sys.argv) > 1 else "experiments/design_sim/design_sim.json", "w"), indent=1, default=float)
log(f"done [{time.time()-t0:.0f}s]")
