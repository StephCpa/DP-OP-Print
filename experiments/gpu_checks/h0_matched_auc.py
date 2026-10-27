"""Is the weak base-calibrated H0 signal (AUC 0.53-0.56) leakage, or split imbalance x level-dependent training shrinkage?

On split 0A the members are harder under theta0 (theta0-only AUC < 0.5). If training lowers Delta more for high-Delta
samples whatever their membership, then 'theta minus theta0' favours members without any memorisation. Two corrections:
  stratified  AUC within deciles of the theta0 score, pooled with weights n_in * n_out (members vs non-members of the
              same theta0 level only);
  residual    score = log S_theta - f(log S_theta0), f a quadratic fitted on NON-members only, cross-fitted (fit on one
              half of the non-members, evaluate members vs the other half, then swap) so no non-member scores itself.
Validity: with random labels the stratified AUC of the theta0 score on itself averages 0.499 (sd 0.018) on these data,
so the estimator is unbiased; on split 0A the actual labels give 0.53 / 0.54, a 1.6-2.3 sd within-decile fluctuation.
Also reports how the generic shrinkage depends on level: the slope of log(S_theta / S_theta0) on log S_theta0 among
non-members (negative = high-Delta samples shrink more), and the theta0 imbalance between members and non-members.
AUC convention as in the H0 run: lower score = member, AUC = P(score_member < score_nonmember).
Run: python3 experiments/gpu_checks/h0_matched_auc.py data/h0/2026-10-27/h0_primary_T1.1.npz data/h0/2026-10-27/h0_control_T1.1.npz
"""
import json, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gmax_rescore import auc_low, chunk_stats  # noqa: E402


def chunk_means(delta, skip, C=64):
    d = delta[:, :, skip:]
    N, K, L = d.shape
    nch = int(np.ceil(L / C))
    if nch * C - L:
        d = np.concatenate([d, np.full((N, K, nch * C - L), np.nan)], 2)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(d.reshape(N, K, nch, C), 3)                # [N, K, chunks], NaN past the end


def stratified_auc(score, ref, member, q=10):
    edges = np.quantile(ref, np.linspace(0, 1, q + 1))
    b = np.clip(np.searchsorted(edges, ref, side="right") - 1, 0, q - 1)
    num = den = 0.0
    for s in range(q):
        m = b == s
        n1, n0 = member[m].sum(), (~member[m]).sum()
        if n1 and n0:
            w = n1 * n0
            num += w * auc_low(score[m], member[m]); den += w
    return num / den


def residual_auc(s, s0, member, rng):
    x, y = np.log(s0), np.log(s)
    non = np.where(~member)[0]; mem = np.where(member)[0]
    perm = rng.permutation(non); halves = (perm[: len(perm) // 2], perm[len(perm) // 2:])
    aucs = []
    for fit, ev in (halves, halves[::-1]):
        coef = np.polyfit(x[fit], y[fit], 2)
        idx = np.concatenate([mem, ev])
        r = y[idx] - np.polyval(coef, x[idx])
        aucs.append(auc_low(r, np.isin(idx, mem)))
    return float(np.mean(aucs))


def scores_for(z):
    d, d0 = z["delta"].astype(np.float32), z["delta0"].astype(np.float32)
    out = {}
    for skip, tag in ((0, "all"), (64, "nc1")):
        cm, cm0 = chunk_means(d, skip), chunk_means(d0, skip)
        G, G0 = np.nanmean(np.nanmean(cm, 2), 1), np.nanmean(np.nanmean(cm0, 2), 1)
        GM, GM0 = np.nanmean(np.nanmax(cm, 2), 1), np.nanmean(np.nanmax(cm0, 2), 1)
        out[f"G {tag}"] = (G, G0)
        out[f"G-Max {tag}"] = (GM, GM0)
        # max over chunks of the chunk-wise CRN difference (a 'delta-version' G-Max), for reference
        out[f"max-chunk(theta - theta0) {tag}"] = (np.nanmean(np.nanmax(cm - cm0, 2), 1), GM0)
    return out


def main(paths, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for p in paths:
        z = np.load(p); member = z["member"].astype(bool)
        arm = os.path.basename(p).split("_")[1]
        rows = {}
        for name, (s, s0) in scores_for(z).items():
            row = {}
            row["theta raw"] = auc_low(s, member)
            row["theta0 only (placebo)"] = auc_low(s0, member)
            if not name.startswith("max-chunk"):
                row["difference theta - theta0"] = auc_low(s - s0, member)
                row["log-ratio theta / theta0"] = auc_low(np.log(s / s0), member)
                row["difference, stratified by theta0 decile"] = stratified_auc(s - s0, s0, member)
                row["log-ratio, stratified by theta0 decile"] = stratified_auc(np.log(s / s0), s0, member)
                row["residual on theta0 (non-member fit, cross-fitted)"] = residual_auc(s, s0, member, rng)
                non = ~member
                row["_slope log-ratio ~ log theta0 (non-members)"] = float(np.polyfit(np.log(s0[non]), np.log(s[non] / s0[non]), 1)[0])
                row["_theta0 mean in / out"] = [float(s0[member].mean()), float(s0[~member].mean())]
            else:
                row["difference theta - theta0"] = auc_low(s, member)
                row["difference, stratified by theta0 decile"] = stratified_auc(s, s0, member)
            # bootstrap CIs for the AUC rows
            ci = {}
            keys = [k for k in row if not k.startswith("_")]
            reps = {k: [] for k in keys}
            for _ in range(B):
                i = rng.integers(0, len(member), len(member))
                mi = member[i]
                for k in keys:
                    if k == "theta raw" or (k == "difference theta - theta0" and name.startswith("max-chunk")):
                        reps[k].append(auc_low(s[i], mi))
                    elif k == "theta0 only (placebo)":
                        reps[k].append(auc_low(s0[i], mi))
                    elif k == "difference theta - theta0":
                        reps[k].append(auc_low(s[i] - s0[i], mi))
                    elif k == "log-ratio theta / theta0":
                        reps[k].append(auc_low(np.log(s[i] / s0[i]), mi))
                    elif k.startswith("difference, stratified"):
                        reps[k].append(stratified_auc((s - s0)[i] if not name.startswith("max-chunk") else s[i], s0[i], mi))
                    elif k.startswith("log-ratio, stratified"):
                        reps[k].append(stratified_auc(np.log(s / s0)[i], s0[i], mi))
                    elif k.startswith("residual"):
                        reps[k].append(residual_auc(s[i], s0[i], mi, rng))
            for k in keys:
                ci[k] = [float(np.quantile(reps[k], 0.025)), float(np.quantile(reps[k], 0.975))]
            rows[name] = {k: (dict(auc=float(v), ci=ci[k]) if k in ci else v) for k, v in row.items()}
        res[arm] = rows
    return res


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if a.endswith(".npz")]
    res = main(paths, B=int(os.environ.get("B", 1000)))
    for arm, rows in res.items():
        print(f"\n== {arm}  (AUC = P(member score < non-member score); * = CI excludes 0.5)")
        for name, row in rows.items():
            print(f"  [{name}]")
            for k, v in row.items():
                if isinstance(v, dict):
                    flag = " *" if (v["ci"][0] > 0.5 or v["ci"][1] < 0.5) else ""
                    print(f"    {k:52s} {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]{flag}")
                else:
                    print(f"    {k:52s} {v}")
    out = os.path.join(os.path.dirname(paths[0]), "h0_matched_auc.json")
    json.dump(res, open(out, "w"), indent=1)
    print("\nwrote", out)
