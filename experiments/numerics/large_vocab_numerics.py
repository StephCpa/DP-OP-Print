"""Extends Appendix A of the design (V=50, fp64/fp32 numpy) to the Qwen3 vocabulary (V=151,936), bf16, and T=1.1.

Synthetic logits only: no Qwen3 weights were reachable from this environment, so logit *levels* are swept
rather than measured. Four parts:
  P1  OPSD's own loss path (log_softmax -> F.kl_div(log_target=True) -> clamp -> sum) in bf16 / fp32 vs an fp64
      reference: loss value, gradient w.r.t. student logits, and how many vocab entries change clamp status.
  P2  KL-value formulas at V=151,936 in fp32 (direct, Barron -sum p s, log1p form - the one App. A did not run in fp32),
      vs fp64 log1p reference (itself checked against mpmath on a few rows).
  P3  Logit-quantisation floor: a synthetic LM head W (V x 2048) and hidden-state pairs with controlled true Delta.
      Pipelines: bf16 head (bf16 logits), fp32 head on bf16-rounded hidden, vs the fp64 head on the unrounded hidden.
      Compared with the reviewer prior ~ sigma_n^2 (1 - sum p^2) (= 3.3e-4 at bf16 spacing 0.0625).
  P4  Same as P3 aggregated to 64-token chunk means and to the max over 16 chunks (the G / G-Max aggregates).
Run:  python3 experiments/numerics/large_vocab_numerics.py  (CPU, ~5-10 min, ~6 GB RAM)
"""
import json, math, sys, time
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(4)
V, D = 151_936, 2048
OUT = {}
t0 = time.time()


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- helpers
def make_logits(n, level, ent, gen, cand=4):
    """LLM-like logits: Gaussian bulk (mean 0, sd 2.5) + `cand` candidate tokens lifted to `level`.
    `ent` in {'low','mid','high'} sets how the lifted mass is split between candidates."""
    z = torch.randn(n, V, generator=gen, dtype=torch.float64) * 2.5
    idx = torch.randint(0, V, (n, cand), generator=gen)
    gaps = {"low": [0.0, 4.0, 6.0, 7.0], "mid": [0.0, 0.7, 1.5, 2.5], "high": [0.0, 0.2, 0.4, 0.6]}[ent]
    for j in range(cand):
        z[torch.arange(n), idx[:, j]] = level - gaps[j]  # top candidate at `level`
    if ent == "high":  # flatten further: lift 60 more tokens close to the top
        more = torch.randint(0, V, (n, 60), generator=gen)
        z.scatter_(1, more, level - 1.0 + 0.3 * torch.randn(n, 60, generator=gen, dtype=torch.float64))
    return z, idx


def entropy(z):
    lp = torch.log_softmax(z, -1)
    return -(lp.exp() * lp).sum(-1)


def opsd_loss(zS, zT, clip, T, dtype):
    """OPSD generalized_jsd_loss, beta=0, one call per row; clip=None disables (as opsd_train.py does for 0)."""
    s = zS.to(dtype) / T          # OPSD divides the (bf16) logits by T inside the loss
    t = zT.to(dtype) / T
    ls, lt = F.log_softmax(s, -1), F.log_softmax(t, -1)
    j = F.kl_div(ls, lt, reduction="none", log_target=True)  # exp(lt) * (lt - ls)
    if clip is not None:
        j = j.clamp(max=clip)
    return j, j.sum(-1)


def analytic_grad64(zS, zT, clip, T):
    """g = dL/dzS for L = sum_k min(f_k, clip) at temperature T: (1/T) (q P_U - p 1[U])."""
    lp, lq = torch.log_softmax(zT / T, -1), torch.log_softmax(zS / T, -1)
    p, q = lp.exp(), lq.exp()
    f = p * (lp - lq)
    U = (f <= clip) if clip is not None else torch.ones_like(f, dtype=torch.bool)
    PU = (p * U).sum(-1, keepdim=True)
    return (q * PU - p * U) / T, f, U


def kl_forms(zT, zS, dtype):
    """Direct, Barron (-sum p s) and log1p(sum p (expm1 d - d)) KL(p||q), p=softmax(zT), q=softmax(zS)."""
    zT, zS = zT.to(dtype), zS.to(dtype)
    lp = torch.log_softmax(zT, -1)
    p = lp.exp()
    direct = (p * (lp - torch.log_softmax(zS, -1))).sum(-1)
    dl = zS - zT
    d = dl - (p * dl).sum(-1, keepdim=True)
    s = d - torch.log1p((p * torch.expm1(d)).sum(-1, keepdim=True))
    barron = -(p * s).sum(-1)
    l1p = torch.log1p((p * (torch.expm1(d) - d)).sum(-1))
    return direct, barron, l1p


# ---------------------------------------------------------------- P1: OPSD loss path in bf16/fp32 vs fp64
log("P1: OPSD loss path at V=151,936 (bf16 / fp32 vs fp64), T in {1.0, 1.1}")
gen = torch.Generator().manual_seed(1)
p1 = []
regimes = {
    # name: (student perturbation sd, sparse under-prediction of the teacher's top token in nats)
    "near-equal (sd 1e-3)": (1e-3, 0.0),
    "diffuse (sd 0.1)": (0.1, 0.0),
    "diffuse (sd 0.5)": (0.5, 0.0),
    "correction site (top token -6)": (0.1, 6.0),
}
for level in (8.0, 16.0, 24.0):
    for ent in ("low", "mid", "high"):
        zT, idx = make_logits(16, level, ent, gen)
        for rname, (sd, drop) in regimes.items():
            zS = zT + sd * torch.randn(zT.shape, generator=gen, dtype=torch.float64)
            zS[torch.arange(16), idx[:, 0]] -= drop
            for T in (1.0, 1.1):
                for clip in (None, 0.05, 1e-6):
                    # reference on the *same rounded inputs* the low-precision path sees, to isolate arithmetic error
                    for dtype in (torch.bfloat16, torch.float32):
                        zS_r = zS.to(dtype).to(torch.float64)
                        zT_r = zT.to(dtype).to(torch.float64)
                        g64, f64, U64 = analytic_grad64(zS_r, zT_r, clip, T)
                        L64 = (f64.clamp(max=clip) if clip is not None else f64).sum(-1)
                        zSv = zS_r.to(dtype).clone().requires_grad_(True)
                        j, L = opsd_loss(zSv, zT_r.to(dtype), clip, T, dtype)
                        L.sum().backward()
                        g = zSv.grad.to(torch.float64)
                        if clip is not None:  # entries whose clamp status differs between low precision and fp64
                            f_low = opsd_loss(zS_r, zT_r, None, T, dtype)[0].to(torch.float64)
                            thr = float(torch.tensor(clip, dtype=dtype))  # clamp(max=clip) compares in `dtype`
                            flips = ((f64 > clip) != (f_low > thr)).sum(-1).double()
                        else:
                            flips = torch.zeros(16, dtype=torch.float64)
                        rel_g = (g - g64).norm(dim=-1) / g64.norm(dim=-1).clamp_min(1e-300)
                        top_sign_ok = (torch.sign(g[torch.arange(16), idx[:, 0]]) == torch.sign(g64[torch.arange(16), idx[:, 0]])).double()
                        p1.append(dict(level=level, ent=ent, regime=rname, T=T, clip=clip, dtype=str(dtype).split(".")[-1],
                                       H=float(entropy(zT_r).mean()), KL64=float(L64.mean()),
                                       abs_err_loss=float((L.detach().to(torch.float64) - L64).abs().mean()),
                                       rel_err_loss=float(((L.detach().to(torch.float64) - L64).abs() / L64.abs().clamp_min(1e-300)).median()),
                                       rel_err_grad_median=float(rel_g.median()), rel_err_grad_max=float(rel_g.max()),
                                       clamp_flips_mean=float(flips.mean()), top_token_grad_sign_agree=float(top_sign_ok.mean()),
                                       grad_norm64=float(g64.norm(dim=-1).median())))
OUT["P1"] = p1


def summarize_p1(rows):
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["dtype"], r["regime"], str(r["clip"]), r["T"])].append(r)
    lines = []
    for k in sorted(agg):
        rs = agg[k]
        lines.append(dict(dtype=k[0], regime=k[1], clip=k[2], T=k[3],
                          KL64_median=float(np.median([r["KL64"] for r in rs])),
                          rel_err_loss_median=float(np.median([r["rel_err_loss"] for r in rs])),
                          rel_err_grad_median=float(np.median([r["rel_err_grad_median"] for r in rs])),
                          rel_err_grad_worst=float(np.max([r["rel_err_grad_max"] for r in rs])),
                          clamp_flips_per_row=float(np.mean([r["clamp_flips_mean"] for r in rs])),
                          top_sign_agree=float(np.mean([r["top_token_grad_sign_agree"] for r in rs]))))
    return lines


OUT["P1_summary"] = summarize_p1(p1)
log(f"{'dtype':8s} {'regime':32s} {'clip':6s} {'T':4s} {'KL64':>9s} {'relErrL':>9s} {'relErrG':>9s} {'worstG':>9s} {'flips':>7s} {'sign':>5s}")
for r in OUT["P1_summary"]:
    log(f"{r['dtype']:8s} {r['regime']:32s} {r['clip']:6s} {r['T']:<4} {r['KL64_median']:9.2e} {r['rel_err_loss_median']:9.2e} "
        f"{r['rel_err_grad_median']:9.2e} {r['rel_err_grad_worst']:9.2e} {r['clamp_flips_per_row']:7.1f} {r['top_sign_agree']:5.2f}")
log(f"[{time.time()-t0:.0f}s]")

# ---------------------------------------------------------------- P2: KL value forms in fp32 at V=151,936
log("\nP2: KL-value formulas in fp32 at V=151,936 vs fp64 log1p reference")
gen = torch.Generator().manual_seed(2)
p2 = {}
zT, _ = make_logits(64, 16.0, "mid", gen)
for sig in (1e-6, 1e-4, 1e-2, 0.1, 0.5):
    zS = zT + sig * torch.randn(zT.shape, generator=gen, dtype=torch.float64)
    # reference: fp64 log1p form on the fp32-rounded logits (the fp32 path sees those)
    zT32, zS32 = zT.float().double(), zS.float().double()
    ref = kl_forms(zT32, zS32, torch.float64)[2]
    d32, b32, l32 = kl_forms(zT32, zS32, torch.float32)
    p2[sig] = {n: float(((v.double() - ref).abs() / ref).max()) for n, v in
               [("fp32 direct", d32), ("fp32 Barron", b32), ("fp32 log1p", l32)]}
    p2[sig]["KL_median"] = float(ref.median())
    log(f"  sigma={sig:g}: KL~{p2[sig]['KL_median']:.2e}  " + "  ".join(f"{k}={v:.1e}" for k, v in p2[sig].items() if k != "KL_median"))
# mpmath spot check of the fp64 reference on 2 rows at the two smallest sigma
try:
    import mpmath
    mpmath.mp.dps = 40
    chk = []
    for sig in (1e-6, 1e-2):
        zS = zT[:2] + sig * torch.randn(zT[:2].shape, generator=gen, dtype=torch.float64)
        ref64 = kl_forms(zT[:2], zS, torch.float64)[2]
        for r in range(2):
            a = [mpmath.mpf(float(x)) for x in zT[r]]
            b = [mpmath.mpf(float(x)) for x in zS[r]]
            la, lb = mpmath.log(mpmath.fsum(mpmath.exp(x) for x in a)), mpmath.log(mpmath.fsum(mpmath.exp(x) for x in b))
            kl = mpmath.fsum(mpmath.exp(x - la) * ((x - la) - (y - lb)) for x, y in zip(a, b))
            chk.append(abs(float(ref64[r]) - float(kl)) / float(kl))
    p2["fp64_ref_vs_mpmath_max_rel_err"] = max(chk)
    log(f"  fp64 log1p reference vs 40-digit mpmath (4 rows, V=151,936): max rel err {max(chk):.1e}")
except Exception as e:  # pragma: no cover
    log("  mpmath check skipped:", e)
OUT["P2"] = {str(k): v for k, v in p2.items()}
log(f"[{time.time()-t0:.0f}s]")

# ---------------------------------------------------------------- P3/P4: logit quantisation and the fp32 head
log("\nP3: synthetic LM head (V x 2048); error of Delta vs fp64 head on unrounded hidden state")
gen = torch.Generator().manual_seed(3)
Wb = torch.randn(V, D, generator=gen) / math.sqrt(D)          # row norm ~ 1
u = torch.randn(D, generator=gen); u /= u.norm()
Wb = (Wb + 0.0 * u).to(torch.bfloat16)                         # weights are bf16 in the checkpoint
W32 = Wb.float()
W64 = W32.double()


def hidden_for(n, level, gen):
    """Hidden states giving bulk logit sd 2.5 and a few candidate tokens near `level` (entropy varies by row)."""
    h = torch.randn(n, D, generator=gen) * 2.5                 # z_k = w_k . h ~ N(0, 2.5^2)
    cand = torch.randint(0, V, (n, 6), generator=gen)
    W6 = W32[cand]                                             # n x 6 x D
    gaps = torch.rand(n, 6, generator=gen) * torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 8.0]) * (torch.rand(n, 1, generator=gen) * 2)
    target = level - gaps                                      # desired logits of the 6 candidates
    cur = torch.einsum("nkd,nd->nk", W6, h)
    # least-norm correction so that the candidate logits hit the targets
    G = torch.einsum("nkd,njd->nkj", W6, W6)
    coef = torch.linalg.solve(G, (target - cur).unsqueeze(-1)).squeeze(-1)
    return h + torch.einsum("nk,nkd->nd", coef, W6)


def logits64(h):
    return h.double() @ W64.T


rows = []
for level in (8.0, 16.0, 24.0):
    for eps in (0.0, 3e-3, 1e-2, 3e-2, 0.1, 0.3):
        n = 64
        hT = hidden_for(n, level, gen)
        hS = hT + eps * torch.randn(n, D, generator=gen) * 2.5 / math.sqrt(1.0)   # perturb in hidden space
        truth = kl_forms(logits64(hT), logits64(hS), torch.float64)[2]           # fp64 head, unrounded hidden
        hTb, hSb = hT.to(torch.bfloat16), hS.to(torch.bfloat16)
        # (a) bf16 head: bf16 matmul -> bf16 logits, KL in fp32 log-space
        zTa, zSa = (hTb @ Wb.T).float(), (hSb @ Wb.T).float()
        a_barron = kl_forms(zTa.double(), zSa.double(), torch.float32)[1].double()
        # (b) bf16 head + OPSD-standard bf16 F.kl_div (no clip)
        b_std = opsd_loss(zSa.to(torch.bfloat16), zTa.to(torch.bfloat16), None, 1.0, torch.bfloat16)[1].double()
        # (c) fp32 head on bf16-rounded hidden, KL in fp32 log-space (the v1.2 pipeline)
        zTc, zSc = hTb.float() @ W32.T, hSb.float() @ W32.T
        c_barron = kl_forms(zTc.double(), zSc.double(), torch.float32)[1].double()
        # (d) the same but direct fp32 KL
        c_direct = kl_forms(zTc.double(), zSc.double(), torch.float32)[0].double()
        # prior for bf16 logits: KL ~ 1/2 Var_p(n_S - n_T) = sum_k p_k (1 - p_k) s_k^2 / 12, s_k = bf16 spacing at z_k
        zt64 = logits64(hT)
        pT = torch.softmax(zt64, -1)
        spacing = 2.0 ** (torch.floor(torch.log2(zt64.abs().clamp_min(1e-30))) - 7)
        prior = (pT * (1 - pT) * spacing ** 2 / 12).sum(-1)
        H = -(pT * torch.log_softmax(zt64, -1)).sum(-1)
        for i in range(n):
            rows.append(dict(level=level, eps=eps, H=float(H[i]), truth=float(truth[i]), prior=float(prior[i]),
                             bf16head_barron=float(a_barron[i]), bf16head_opsd_std=float(b_std[i]),
                             fp32head_barron=float(c_barron[i]), fp32head_direct=float(c_direct[i])))
        log(f"  level={level:4.0f} eps={eps:<6g} truth median={float(truth.median()):.2e}  "
            f"bias: bf16head={float((a_barron-truth).mean()):+.2e} opsd_bf16={float((b_std-truth).mean()):+.2e} "
            f"fp32head={float((c_barron-truth).mean()):+.2e} fp32direct={float((c_direct-truth).mean()):+.2e}  prior={float(prior.mean()):.2e}")
OUT["P3_rows"] = rows

# summary by level: floor (eps=0) and relative error at each truth scale, by entropy tercile
summ = []
for level in (8.0, 16.0, 24.0):
    for pipe in ("bf16head_barron", "bf16head_opsd_std", "fp32head_barron", "fp32head_direct"):
        r0 = [r for r in rows if r["level"] == level and r["eps"] == 0.0]
        floor = np.array([r[pipe] - r["truth"] for r in r0])
        hi = np.array([r["H"] for r in r0]) > np.median([r["H"] for r in r0])
        entry = dict(level=level, pipeline=pipe, floor_mean=float(floor.mean()), floor_p99=float(np.quantile(np.abs(floor), 0.99)),
                     floor_mean_highH=float(floor[hi].mean()), floor_mean_lowH=float(floor[~hi].mean()),
                     prior_mean=float(np.mean([r["prior"] for r in r0])))
        for eps in (3e-3, 1e-2, 3e-2, 0.1, 0.3):
            re = [r for r in rows if r["level"] == level and r["eps"] == eps]
            rel = np.array([(r[pipe] - r["truth"]) / r["truth"] for r in re])
            entry[f"relerr_median@eps{eps:g}"] = float(np.median(rel))
            entry[f"truth_median@eps{eps:g}"] = float(np.median([r["truth"] for r in re]))
        summ.append(entry)
OUT["P3_summary"] = summ
log("\n  floor at true Delta=0 (eps=0), mean bias [nats/token]; prior = sum_k p_k (1-p_k) s_k^2/12:")
for e in summ:
    log(f"   level={e['level']:4.0f} {e['pipeline']:20s} floor={e['floor_mean']:+.2e} (highH {e['floor_mean_highH']:+.2e}, lowH {e['floor_mean_lowH']:+.2e}) "
        f"|p99|={e['floor_p99']:.2e} prior={e['prior_mean']:.2e}  rel.err@truth~{e['truth_median@eps0.01']:.0e}: {e['relerr_median@eps0.01']:+.1e}"
        f"  @~{e['truth_median@eps0.1']:.0e}: {e['relerr_median@eps0.1']:+.1e}")
log(f"[{time.time()-t0:.0f}s]")

# P4: chunk aggregates. Positions are drawn from the eps>0 rows at one level; chunks of 64, 16 chunks per rollout.
log("\nP4: chunk mean (G) and max-over-16-chunks (G-Max) relative error, from P3 per-position errors")
rng = np.random.default_rng(4)
p4 = []
for level in (8.0, 16.0, 24.0):
    for eps in (3e-3, 1e-2, 3e-2, 0.1):
        pool = [r for r in rows if r["level"] == level and r["eps"] == eps]
        tr_all = np.array([r["truth"] for r in pool])
        for pipe in ("bf16head_barron", "bf16head_opsd_std", "fp32head_barron"):
            me_all = np.array([r[pipe] for r in pool])
            ch = rng.choice(len(pool), size=(400, 16, 64))
            tr, me = tr_all[ch], me_all[ch]
            relm = (me.mean((1, 2)) - tr.mean((1, 2))) / tr.mean((1, 2))
            relx = (me.mean(2).max(1) - tr.mean(2).max(1)) / tr.mean(2).max(1)
            p4.append(dict(level=level, eps=eps, pipeline=pipe, truth_chunk_mean=float(np.mean([r["truth"] for r in pool])),
                           relerr_G_median=float(np.median(relm)), relerr_GMax_median=float(np.median(relx)),
                           relerr_GMax_p95=float(np.quantile(np.abs(relx), 0.95))))
            log(f"  level={level:4.0f} eps={eps:<6g} {pipe:20s} truth~{p4[-1]['truth_chunk_mean']:.1e}: G {p4[-1]['relerr_G_median']:+.1e}  "
                f"G-Max {p4[-1]['relerr_GMax_median']:+.1e} (|p95| {p4[-1]['relerr_GMax_p95']:.1e})")
OUT["P4"] = p4

json.dump(OUT, open(sys.argv[1] if len(sys.argv) > 1 else "experiments/numerics/large_vocab_numerics.json", "w"), indent=1)
log(f"done [{time.time()-t0:.0f}s]")
