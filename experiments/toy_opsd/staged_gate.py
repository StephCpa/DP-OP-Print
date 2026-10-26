"""Staged reliability gate (design v1.3 §5): after 2 split+complement pairs (4 models), split-half r_11 across the pairs
(each half 1 in / 1 out), projected to 2 in / 2 out by Spearman-Brown, r_22 = 2 r_11 / (1 + r_11); continue to 8 models
only if r_22 >= 0.3 (i.e. r_11 >= 0.176). Compared with the full 8-model split-half r. Run: python3 experiments/toy_opsd/staged_gate.py"""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
out = {}
for arm in [a for a in R if not a.startswith("_")]:
    M = [np.load(os.path.join(HERE, "runs", arm, f"model{m}.npz")) for m in range(4)]
    v = []
    for pair in ((0, 1), (2, 3)):
        D = [M[m]["D_theta_attack"].mean(1) / np.median(M[m]["D_theta_attack"].mean(1)[~M[m]["member"]]) for m in pair]
        mem = M[pair[0]]["member"]
        v.append(np.log(np.where(mem, D[1] / D[0], D[0] / D[1])))      # log(out / in) for every sample
    r11 = float(np.corrcoef(v[0], v[1])[0, 1])
    r22 = 2 * r11 / (1 + r11)
    out[arm] = dict(r11=r11, r22_projected=r22, go=bool(r22 >= 0.3), r_split_half_8models=R[arm]["H1"]["reliability_split_half"])
    print(f"{arm:20s} r_11 {r11:+.3f} -> projected r_22 {r22:+.3f} ({'go' if r22 >= 0.3 else 'stop'}); 8-model split-half r {out[arm]['r_split_half_8models']:+.3f}")
json.dump(out, open(os.path.join(HERE, "staged_gate.json"), "w"), indent=1)
