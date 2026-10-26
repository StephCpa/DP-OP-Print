"""Dry run of the v1.3 §5 go/no-go: would the post-H0 reliability bound have stopped per-sample H1 in the toy?

Uses only what the real study has after H0: one trained model (model 0 of each arm) and theta0 rollouts.
  sigma_tot^2 = Var(l | in) + Var(l | out),     l = log mean-Delta of the attack on model 0 (K = 8)
  sigma_roll^2 from two independent theta0 rollout sets (K = 4 each, rescaled to K = 8 by 1/2)
  sigma_a^2   = Var(l0) - sigma_roll^2           l0 = log mean-Delta under theta0 (K = 8)
  R = sigma_tot^2 - 2 sigma_a^2,  bound: r_raw <= (R - 2 sigma_roll^2) / (R - sigma_roll^2)
Compared with the split-half r measured on all 8 models (results.json). Also tau_hat^2 = Var(l|in) - Var(l|out).
Run: python3 experiments/toy_opsd/gate_bound.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
R = json.load(open(os.path.join(HERE, "results.json")))
out = {}
for arm, res in R.items():
    if arm.startswith("_"):
        continue
    think = res["meta"]["think"]
    f = np.load(os.path.join(RUNS, f"theta0_features{'' if think else '_nothink'}.npz"))
    D0 = f["D0"].mean(2)                                      # (N, K) per-rollout mean Delta under theta0
    lA, lB = np.log(D0[:, :4].mean(1)), np.log(D0[:, 4:].mean(1))
    s_roll8 = 0.5 * np.var(lA - lB) / 2
    l0 = np.log(D0.mean(1))
    s_a = np.var(l0) - s_roll8
    m = np.load(os.path.join(RUNS, arm, "model0.npz"))
    l = np.log(m["D_theta_attack"].mean(1)); mem = m["member"]
    s_tot = np.var(l[mem]) + np.var(l[~mem])
    Rr = s_tot - 2 * s_a
    bound = (Rr - 2 * s_roll8) / (Rr - s_roll8) if Rr > s_roll8 else float("nan")
    out[arm] = dict(sigma_tot2=float(s_tot), sigma_a2=float(s_a), sigma_roll2=float(s_roll8), R=float(Rr),
                    bound=float(bound), tau_hat2=float(np.var(l[mem]) - np.var(l[~mem])),
                    split_half_r=res["H1"]["reliability_split_half"], auc_G=res["auc"]["G"]["mean"])
    print(f"{arm:20s} AUC_G {res['auc']['G']['mean']:.3f}  sigma_tot2 {s_tot:.4f} 2sigma_a2 {2*s_a:.4f} sigma_roll2 {s_roll8:.4f} "
          f"R {Rr:+.4f} -> bound {bound:+.2f} (measured split-half r {res['H1']['reliability_split_half']:+.2f}); tau_hat2 {out[arm]['tau_hat2']:+.4f}")
json.dump(out, open(os.path.join(HERE, "gate_bound.json"), "w"), indent=1)
