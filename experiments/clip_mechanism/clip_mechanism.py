"""What OPSD's element-wise clip does to a single position, beyond Appendix A (fp64 numpy).

Setting (OPSD beta=0, temperature T): p = softmax(z_T/T) fixed teacher, q = softmax(z_S/T) student,
L(z_S) = sum_k min(f_k, c),  f_k = p_k (log p_k - log q_k),  U = {k : f_k <= c},  P_U = sum_{k in U} p_k.
Gradient w.r.t. z_S/T (Appendix A, verified): g_j = q_j P_U - p_j 1[j in U].

Claims checked here (all per position, logits treated as free parameters = the 'direct' route only):
 M1  Every clamped entry a (f_a > c) has g_a = q_a P_U >= 0: gradient descent never raises a clamped logit.
 M2  inf_q L = min_C [ |C| c + h(P_C) ], h(P) = (1 - P) log(1 - P)  (C = abandoned set: q -> 0 on C, q ∝ p elsewhere).
     A single token with p_k - p_k^2/2 > c (roughly p_k > c) already gives L < 0 = L(q = p), so q = p is then only a
     *local* minimum. Checked: long logit-space gradient descent never goes below the convex lower bound, and ends
     at L < 0 (with the full KL above its start) from many realistic starts.
 M3  First-order change of the *full* KL under one clipped-gradient step: dKL/deta = -<q - p, g>  (logit space).
     r = <q - p, g> can be negative: clipped training then raises the full KL at that position.
 M4  v1.2's competitor m_U = sum_{k in U} f_k can be negative (worked example: -0.225) and is a poor proxy for r.
 M5  Phase diagram for a correction site (teacher confidence p_a, student entropy H_S, under-prediction q_a/p_a) x clip x T:
     is a clamped, does log q_a fall (exact one-step sign, closed form checked), is r < 0, is m_U < 0.
 M6  Ensemble at V = 151,936 (synthetic LLM-like logits; same generator as experiments/numerics): share of KL mass
     clamped, P_U, sign of r, sign of m_U, rank correlation of m_U and alternatives with r, at T = 1.0 and 1.1.
Run: python3 experiments/clip_mechanism/clip_mechanism.py   (CPU, ~2 min)
"""
import json, sys, itertools
import numpy as np
from scipy.stats import spearmanr

np.set_printoptions(precision=5, suppress=True)
OUT = {}


def lsm(z):
    m = z.max(-1, keepdims=True)
    return z - (m + np.log(np.exp(z - m).sum(-1, keepdims=True)))


def parts(zS, zT, c, T=1.0):
    lp, lq = lsm(zT / T), lsm(zS / T)
    p, q = np.exp(lp), np.exp(lq)
    f = p * (lp - lq)
    U = f <= c if c is not None else np.ones_like(f, bool)
    PU = (p * U).sum(-1, keepdims=True)
    g = q * PU - p * U                       # gradient w.r.t. z_S / T
    L = np.minimum(f, c).sum(-1) if c is not None else f.sum(-1)
    return dict(p=p, q=q, f=f, U=U, PU=PU[..., 0], g=g, L=L, KL=f.sum(-1))


def inf_formula(p, c):
    """Upper bound on inf L: best abandoned set among the top-k tokens by p (exact when the top-k sets are optimal)."""
    ps = np.sort(p)[::-1]
    best, bestk = 0.0, 0
    P = 0.0
    for k in range(1, len(ps)):
        P += ps[k - 1]
        val = k * c + (1 - P) * np.log1p(-P)
        if val < best:
            best, bestk = val, k
    return best, bestk


def inf_lower_bound(p, c):
    """Lower bound on inf L = min_C |C| c + h(P_C), h(P) = (1-P) log(1-P) convex with minimum -1/e at P* = 1 - 1/e.
    For |C| = k every achievable P_C lies in [sum of k smallest p, sum of k largest p], so
    inf L >= min_k [ k c + h(clip(P*, lo_k, hi_k)) ]  (k = 0 gives 0)."""
    ps = np.sort(p)
    lo, hi = np.cumsum(ps), np.cumsum(ps[::-1])
    Pst = 1 - np.exp(-1)
    Pk = np.clip(Pst, lo, np.minimum(hi, 1 - 1e-15))
    vals = np.arange(1, len(p) + 1) * c + (1 - Pk) * np.log1p(-Pk)
    return min(0.0, float(vals[:-1].min()))


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- M1 + M2 on the worked example and random cases
log("M1/M2: worked example (V=20, student uniform, teacher p_a=0.9), clip 0.05")
V = 20
zS0 = np.zeros(V)
p = np.full(V, 0.1 / (V - 1)); p[0] = 0.9
zT = np.log(p)
r0 = parts(zS0, zT, 0.05)
log(f"  L(q uniform) = {r0['L']:.4f}, KL = {r0['KL']:.4f}, g_a = {r0['g'][0]:+.4f} (= q_a P_U = {0.05*r0['PU']:.4f}), "
    f"m_U = sum_U f = {r0['f'][r0['U']].sum():+.4f}")
best, k = inf_formula(p, 0.05)
log(f"  inf L: top-k upper bound = {best:+.4f} (abandon top {k} token), lower bound = {inf_lower_bound(p, 0.05):+.4f}; L(q = p) = 0")


def gd(zS, zT, c, steps, lr, T=1.0):
    z = zS.copy()
    for _ in range(steps):
        z -= lr * parts(z, zT, c, T)["g"] / T
    return z


zf = gd(zS0, zT, 0.05, 20000, 0.5)
rf = parts(zf, zT, 0.05)
log(f"  logit-space GD from uniform, 20k steps: L = {rf['L']:+.4f}, q_a = {rf['q'][0]:.2e}, KL(p||q) = {rf['KL']:.2f}")
zf2 = gd(np.log(p) + 0.01 * np.random.default_rng(0).normal(size=V), zT, 0.05, 20000, 0.5)
rf2 = parts(zf2, zT, 0.05)
log(f"  logit-space GD from q ≈ p (no entry clamped), 20k steps: L = {rf2['L']:+.2e}, q_a = {rf2['q'][0]:.4f}, KL = {rf2['KL']:.2e}")
OUT["M2_worked"] = dict(L_uniform=float(r0["L"]), m_U_uniform=float(r0["f"][r0["U"]].sum()), g_a=float(r0["g"][0]),
                        inf_formula=float(best), abandon=int(k), gd_from_uniform_L=float(rf["L"]), gd_from_uniform_qa=float(rf["q"][0]),
                        gd_from_uniform_KL=float(rf["KL"]), gd_near_p_L=float(rf2["L"]), gd_near_p_KL=float(rf2["KL"]))

log("\nM1/M2 on 300 random positions (V=50; teacher sd 3; student = teacher + N(0, s^2), s in {0.3,1,3}); clips 0.2/0.05/0.01/1e-6")
rng = np.random.default_rng(1)
m2 = []
for c in (0.2, 0.05, 0.01, 1e-6):
    worst_ga, n_basin, gap, kl_up, neg_possible = 0.0, 0, [], [], []
    for _ in range(300):
        zT_ = rng.normal(0, 3, 50)
        s = rng.choice([0.3, 1.0, 3.0])
        zS_ = zT_ + rng.normal(0, s, 50)
        r = parts(zS_, zT_, c)
        clamped = ~r["U"]
        if clamped.any():
            worst_ga = min(worst_ga, float(r["g"][clamped].min()))   # M1: should never be < 0
        lb = inf_lower_bound(r["p"], c)
        ub, _ = inf_formula(r["p"], c)
        zf = gd(zS_, zT_, c, 4000, 1.0)
        rf_ = parts(zf, zT_, c)
        gap.append(rf_["L"] - lb)                                    # must be >= 0: GD cannot beat the lower bound
        n_basin += rf_["L"] < -1e-3
        kl_up.append(rf_["KL"] > r["KL"])
        neg_possible.append(ub < 0)
    m2.append(dict(clip=c, min_g_on_clamped=worst_ga, frac_runs_ending_below_zero=n_basin / 300,
                   frac_full_KL_increased=float(np.mean(kl_up)), frac_negative_loss_exists=float(np.mean(neg_possible)),
                   median_gap_to_lower_bound=float(np.median(gap)), min_gap_to_lower_bound=float(np.min(gap))))
    log(f"  clip={c:g}: min g on clamped = {worst_ga:+.1e} (M1: >= 0); L<0 reachable in {100*np.mean(neg_possible):.0f}% of positions; "
        f"GD ends with L<0 in {n_basin/3:.0f}%, with full KL above its start in {100*np.mean(kl_up):.0f}%; "
        f"L_GD - lower bound: min {np.min(gap):+.1e} (>= 0)")
OUT["M2_random"] = m2

# ---------------------------------------------------------------- M3/M4: r = <q - p, g> and m_U on the worked example
log("\nM3: first-order full-KL change r = <q-p, g> (r < 0 => one clipped step raises KL(p||q))")
for c in (None, 0.2, 0.05, 0.01, 1e-6):
    r = parts(zS0, zT, c)
    rr = float(((r["q"] - r["p"]) * r["g"]).sum())
    log(f"  worked example, clip={c}: r = {rr:+.4f}, m_U = {r['f'][r['U']].sum():+.4f}, P_U = {r['PU']:.3f}")

# ---------------------------------------------------------------- M5: phase diagram for a correction site
log("\nM5: correction site. V = 1000; teacher: p_a on token a + Zipf tail. Student: the teacher's tail re-tempered to")
log("    entropy H_S (same ranking as the teacher), with q_a = p_a * ratio. Exact one-step sign of d log q_a:")
log("    d log q_a / d eta  ∝  P_U (sum_i q_i^2 - q_a) - sum_{i in U} p_i q_i   (for clamped a; positive = q_a rises)")
rng = np.random.default_rng(2)
Vp = 1000
zipf = 1.0 / np.arange(1, Vp) ** 1.1
grid = []
for T in (1.0, 1.1):
    for c in (0.2, 0.05, 0.01, 1e-6):
        for pa in (0.3, 0.6, 0.9):
            for HS in (0.5, 2.0, 4.0, 6.0):
                for ratio in (0.5, 0.1, 0.01):
                    tail = zipf / zipf.sum()
                    pT = np.concatenate([[pa], tail * (1 - pa)])
                    lo, hi = 0.05, 50.0                        # temper the tail so the student's entropy is H_S
                    for _ in range(80):
                        mid = (lo * hi) ** 0.5
                        tq = tail ** (1 / mid); tq /= tq.sum()
                        qq = np.concatenate([[pa * ratio], tq * (1 - pa * ratio)])
                        H = -(qq * np.log(qq)).sum()
                        lo, hi = (mid, hi) if H < HS else (lo, mid)
                    zT_, zS_ = np.log(pT) * T, np.log(qq) * T
                    r = parts(zS_, zT_, c, T)
                    dlogqa = -(r["g"][0] - (r["q"] * r["g"]).sum())
                    closed = r["PU"] * ((r["q"] ** 2).sum() - r["q"][0]) - (r["p"] * r["q"] * r["U"]).sum()
                    rr = ((r["q"] - r["p"]) * r["g"]).sum()
                    grid.append(dict(T=T, clip=c, p_a=pa, H_S=HS, ratio=ratio, H_S_hit=float(H), a_clamped=bool(~r["U"][0]),
                                     dlogqa=float(dlogqa), closed_form=float(closed), r=float(rr), m_U=float(r["f"][r["U"]].sum()),
                                     KL=float(r["KL"]), P_U=float(r["PU"])))
OUT["M5_grid"] = grid
cl = [g for g in grid if g["a_clamped"]]
log(f"  closed form vs autograd-free direct computation, clamped cases: max |sign mismatch| = "
    f"{sum(np.sign(g['dlogqa']) != np.sign(g['closed_form']) for g in cl)} of {len(cl)}")
for T in (1.0, 1.1):
    for c in (0.05, 1e-6):
        log(f"  T={T} clip={c:g}: cell = (a clamped?, q_a direction, r sign, m_U sign)   rows p_a, columns H_S; ratio q_a/p_a = 0.1")
        for pa in (0.3, 0.6, 0.9):
            cells = []
            for HS in (0.5, 2.0, 4.0, 6.0):
                g_ = [x for x in grid if x["T"] == T and x["clip"] == c and x["p_a"] == pa and x["H_S"] == HS and x["ratio"] == 0.1][0]
                cells.append(f"H_S={HS}:({'C' if g_['a_clamped'] else 'u'},{'up' if g_['dlogqa'] > 0 else 'DOWN'},"
                             f"{'+' if g_['r'] > 0 else '-'},{'+' if g_['m_U'] > 0 else '-'})")
            log(f"    p_a={pa}: " + "  ".join(cells))
summ5 = {}
for c in (0.2, 0.05, 0.01, 1e-6):
    for T in (1.0, 1.1):
        e = [g for g in grid if g["clip"] == c and g["T"] == T]
        summ5[f"clip={c:g},T={T}"] = dict(a_clamped=float(np.mean([g["a_clamped"] for g in e])),
                                        qa_falls_given_clamped=float(np.mean([g["dlogqa"] < 0 for g in e if g["a_clamped"]] or [np.nan])),
                                        r_negative=float(np.mean([g["r"] < 0 for g in e])), m_U_negative=float(np.mean([g["m_U"] < 0 for g in e])))
        log(f"  clip={c:<6g} T={T}: a clamped {100*summ5[f'clip={c:g},T={T}']['a_clamped']:.0f}%, q_a falls (given clamped) "
            f"{100*summ5[f'clip={c:g},T={T}']['qa_falls_given_clamped']:.0f}%, r<0 {100*summ5[f'clip={c:g},T={T}']['r_negative']:.0f}%, "
            f"m_U<0 {100*summ5[f'clip={c:g},T={T}']['m_U_negative']:.0f}%  (over p_a x H_S x ratio grid)")
OUT["M5_summary"] = summ5

# ---------------------------------------------------------------- M6: ensemble at V = 151,936
log("\nM6: ensemble at V = 151,936 (synthetic logits; bulk sd 2.5, candidates near level 16)")
V6 = 151_936
rng = np.random.default_rng(3)


def make_pair(regime):
    z = rng.normal(0, 2.5, V6)
    cand = rng.choice(V6, 6, replace=False)
    z[cand] = 16 - rng.random(6) * np.array([0, 1, 2, 3, 5, 8]) * rng.random() * 2
    zs = z.copy()
    if regime == "diffuse":
        zs += rng.normal(0, rng.choice([0.1, 0.3, 1.0]), V6)
    elif regime == "correction":                         # student under-predicts the teacher's top token, else close
        zs += rng.normal(0, 0.1, V6)
        top = cand[np.argmax(z[cand])]
        zs[top] -= rng.choice([1.0, 3.0, 6.0])
    elif regime == "style":                              # teacher boosts a few 'style' tokens the student does not
        zs += rng.normal(0, 0.1, V6)
        sty = rng.choice(V6, 8, replace=False)
        z[sty] = 16 - rng.uniform(0, 4, 8)
    return z, zs


ens = []
for regime in ("diffuse", "correction", "style"):
    for _ in range(40):
        zT_, zS_ = make_pair(regime)
        for T in (1.0, 1.1):
            for c in (0.2, 0.05, 0.01, 1e-6):
                r = parts(zS_, zT_, c, T)
                d = r["q"] - r["p"]
                rr = float((d * r["g"]).sum())
                clamped_mass = float(r["f"][~r["U"]].sum() / r["KL"])
                ens.append(dict(regime=regime, T=T, clip=c, KL=float(r["KL"]), r=rr, r_unclipped=float((d * d).sum()),
                                clamped_KL_share=clamped_mass, P_U=float(r["PU"]),
                                m_U=float(r["f"][r["U"]].sum()), m_U_pos=float(np.clip(r["f"][r["U"]], 0, None).sum()),
                                clipped_loss=float(r["L"]), g_norm2=float((r["g"] ** 2).sum())))
OUT["M6"] = ens
log(f"  {'T':4s} {'clip':6s} {'regime':11s} {'KL med':>8s} {'clampKL%':>9s} {'P_U med':>8s} {'r<0 %':>6s} {'m_U<0 %':>8s}")
summ6 = []
for T in (1.0, 1.1):
    for c in (0.2, 0.05, 0.01, 1e-6):
        for regime in ("diffuse", "correction", "style"):
            e = [x for x in ens if x["T"] == T and x["clip"] == c and x["regime"] == regime]
            row = dict(T=T, clip=c, regime=regime, KL_med=float(np.median([x["KL"] for x in e])),
                       clamped_KL_share_med=float(np.median([x["clamped_KL_share"] for x in e])),
                       P_U_med=float(np.median([x["P_U"] for x in e])), r_neg=float(np.mean([x["r"] < 0 for x in e])),
                       m_U_neg=float(np.mean([x["m_U"] < 0 for x in e])))
            summ6.append(row)
            log(f"  {T:<4} {c:<6g} {regime:11s} {row['KL_med']:8.3f} {100*row['clamped_KL_share_med']:8.1f}% {row['P_U_med']:8.3f} "
                f"{100*row['r_neg']:5.0f}% {100*row['m_U_neg']:7.0f}%")
OUT["M6_summary"] = summ6
log("\n  Spearman rank correlation with r (direct first-order KL reduction), pooled over regimes:")
corr = []
for T in (1.0, 1.1):
    for c in (0.05, 0.01, 1e-6):
        e = [x for x in ens if x["T"] == T and x["clip"] == c]
        rv = [x["r"] for x in e]
        row = dict(T=T, clip=c)
        for k in ("m_U", "m_U_pos", "clipped_loss", "g_norm2", "KL"):
            row[k] = float(spearmanr([x[k] for x in e], rv)[0])
        corr.append(row)
        log(f"   T={T} clip={c:<6g} " + "  ".join(f"{k}={row[k]:+.2f}" for k in ("m_U", "m_U_pos", "clipped_loss", "g_norm2", "KL")))
OUT["M6_corr"] = corr
json.dump(OUT, open(sys.argv[1] if len(sys.argv) > 1 else "experiments/clip_mechanism/clip_mechanism.json", "w"), indent=1, default=float)
log("done")
