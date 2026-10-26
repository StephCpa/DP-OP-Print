"""fp64 toy check of the OPSD element-wise JSD clip (beta=0) and two v1.1 formulas.
OPSD (opsd_trainer.py, generalized_jsd_loss, beta==0):
  jsd = F.kl_div(log_q, log_p, reduction="none", log_target=True)  # = p_k (log p_k - log q_k), shape [..., V]
  jsd = jsd.clamp(max=token_clip)                                    # element-wise, before vocab sum
  loss = jsd.sum() / n_tokens
We reproduce that per position in numpy fp64 and differentiate w.r.t. student logits z_S."""
import numpy as np
np.set_printoptions(precision=6, suppress=True)
lsm = lambda z: z - (z.max() + np.log(np.exp(z - z.max()).sum()))

def opsd_loss(zS, logp, clip):                       # OPSD definition, one position
    f = np.exp(logp) * (logp - lsm(zS))
    return (np.minimum(f, clip) if clip > 0 else f).sum()   # clip=0 disables (OPSD: jsd_token_clip=0)

def fd_grad(zS, logp, clip, h=1e-6):                 # central finite differences = autograd reference
    g = np.zeros_like(zS)
    for j in range(zS.size):
        e = np.zeros_like(zS); e[j] = h
        g[j] = (opsd_loss(zS + e, logp, clip) - opsd_loss(zS - e, logp, clip)) / (2 * h)
    return g

def analytic_grad(zS, logp, clip):                   # critique: g_j = q_j P_U - p_j 1[j in U]
    p, q = np.exp(logp), np.exp(lsm(zS))
    f = p * (logp - np.log(q))
    U = (f <= clip) if clip > 0 else np.ones_like(p, bool)   # torch clamp(max) passes grad iff f <= max
    PU = p[U].sum()
    return q * PU - p * U, U, PU

def v11_grad(zS, logp, clip):                        # v1.1 form: 1[U](q-p) - q(1-P_U) + q 1[not U]
    p, q = np.exp(logp), np.exp(lsm(zS)); _, U, PU = analytic_grad(zS, logp, clip)
    return U * (q - p) - q * (1 - PU) + q * (~U)

# Worked example: student uniform over 20 tokens, teacher puts 0.9 on token a=0.
V = 20; zS = np.zeros(V)
p = np.full(V, 0.1 / (V - 1)); p[0] = 0.9; logp = np.log(p)
q = np.exp(lsm(zS)); print("summands p(log p - log q):", (p * (logp - np.log(q)))[[0, 1]])
for clip in [0.0, 0.05]:
    g, U, PU = analytic_grad(zS, logp, clip); gfd = fd_grad(zS, logp, clip)
    dlogq_a = -(g[0] - (q * g).sum())                # d log q_a per unit SGD step (z <- z - g)
    print(f"clip={clip}: a clamped={not U[0]} P_U={PU:.4f} g_a={g[0]:+.6f} "
          f"g_a-sum(q g)={g[0]-(q*g).sum():+.6f} dlog q_a/step={dlogq_a:+.6f} "
          f"max|analytic-FD|={np.abs(g-gfd).max():.2e} max|v1.1-analytic|={np.abs(v11_grad(zS,logp,clip)-g).max():.2e}")
    zs = zS.copy()                                   # 20 small SGD steps to see the direction of q_a
    for _ in range(20): zs = zs - 0.5 * analytic_grad(zs, logp, clip)[0]
    print(f"   after 20 SGD steps (lr 0.5): q_a {q[0]:.4f} -> {np.exp(lsm(zs))[0]:.4f}")

# Random checks: analytic vs FD grad, and three KL implementations vs a 50-digit Decimal reference.
from decimal import Decimal, getcontext; getcontext().prec = 50
def kl_ref(zT, zS):                                  # exact-ish KL(p||q), p=softmax(zT), q=softmax(zS)
    a = [Decimal(float(x)) for x in zT]; b = [Decimal(float(x)) for x in zS]
    lseT = sum(x.exp() for x in a).ln(); lseS = sum(x.exp() for x in b).ln()
    return float(sum((x - lseT).exp() * ((x - lseT) - (y - lseS)) for x, y in zip(a, b)))
rng = np.random.default_rng(0); worst_g = 0.0; err = {}
for _ in range(200):
    sig = rng.choice([1e-6, 1e-4, 1e-2, 0.5]); zT = rng.normal(0, 3, 50); zS2 = zT + rng.normal(0, sig, 50); lp = lsm(zT)
    for clip in [0.0, 0.05]:
        worst_g = max(worst_g, np.abs(analytic_grad(zS2, lp, clip)[0] - fd_grad(zS2, lp, clip)).max())
    pp, d0 = np.exp(lp), zS2 - zT; d = d0 - (pp * d0).sum()
    s_ = d - np.log1p((pp * np.expm1(d)).sum())                     # Barron Eq. 9: log(q/p)
    ref = kl_ref(zT, zS2)
    for name, kl in [("direct p(logp-logq)", (pp * (lp - lsm(zS2))).sum()), ("-sum p s (Barron)", -(pp * s_).sum()),
                     ("log1p(sum p(expm1 d-d))", np.log1p((pp * (np.expm1(d) - d)).sum()))]:
        k = (name, sig); err[k] = max(err.get(k, 0), abs(kl - ref) / ref)
    f32 = np.float32; zT32, zS32 = zT.astype(f32), zS2.astype(f32)   # same inputs, fp32 arithmetic
    lp32 = lsm(zT32); p32 = np.exp(lp32); d32 = (zS32 - zT32) - (p32 * (zS32 - zT32)).sum()
    s32 = d32 - np.log1p((p32 * np.expm1(d32)).sum())
    ref32 = kl_ref(zT32.astype(float), zS32.astype(float))           # reference on the fp32-rounded logits
    for name, kl in [("fp32 direct", (p32 * (lp32 - lsm(zS32))).sum()), ("fp32 Barron", -(p32 * s32).sum())]:
        k = (name, sig); err[k] = max(err.get(k, 0), abs(float(kl) - ref32) / ref32)
print(f"random (200 cases, V=50): max|analytic-FD grad| = {worst_g:.2e}")
print("max relative KL error vs 50-digit reference, by logit-perturbation sigma:")
for sig in [1e-6, 1e-4, 1e-2, 0.5]:
    print(f"  sigma={sig:g}: " + "; ".join(f"{n}={err[(n, sig)]:.1e}" for n in ["direct p(logp-logq)", "-sum p s (Barron)", "log1p(sum p(expm1 d-d))", "fp32 direct", "fp32 Barron"]))
