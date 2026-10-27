"""Check 2: re-score G / G-Max without the opening segment, plus the diagnostics needed to read the result.

AUC convention (as in the H0 run): the score is the Delta aggregate, LOWER = more likely a member, so
AUC = P(score_member < score_nonmember). AUC > 0.5 is the OP-Print direction; AUC < 0.5 is reported, never folded.

Input: one .npz per group with
  delta    float [N, K, L]  per-token Delta under theta (the attack), NaN after the end of each rollout   (required)
  member   bool  [N]                                                                                   (required)
  delta0   float [N, K, L]  theta0 Delta on the SAME rollouts (CRN)                                    (optional)
  dtheta   float [N, K, L]  theta0-teacher arm D_theta on the same rollouts (for f_rm by position)      (optional)
  Lmax     int              completion cap (default L); a rollout of length Lmax counts as truncated  (optional)
Produces (for chunk size C = 64 by default):
  * G and G-Max: all tokens; dropping the first k in {1, 4, 16} tokens and re-chunking; from the 2nd segment on (k = C)
  * share of rollouts whose max chunk is chunk 1; Delta_1 mean and its spread ACROSS the K rollouts of a sample
    (Delta_1 depends only on the prompts, so any spread beyond numerical noise means batch/padding dependence)
  * with delta0: theta0-only placebo AUC (split difficulty imbalance), CRN-calibrated G difference and ratio
  * truncation rate in vs out, and G / G-Max AUC on samples whose rollouts mostly hit the cap vs the rest
  * with dtheta: f_rm on members for all tokens, without t = 1, and without the first segment
Usage: python3 gmax_rescore.py main.npz control.npz [--chunk 64] [--boot 2000] [--out rescore.json]
"""
import argparse, json, sys, warnings
import numpy as np
from scipy.stats import rankdata


def auc_low(score, member):
    """P(score_member < score_nonmember), ties 1/2."""
    r = rankdata(-score)
    n1 = member.sum(); n0 = len(member) - n1
    return (r[member].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def chunk_stats(delta, skip, C):
    """Per sample: mean over K of (mean over chunks) and of (max over chunks), after dropping the first `skip` tokens.
    Chunks restart at the new first token; a final partial chunk is kept. Also returns the argmax-chunk index per rollout."""
    d = delta[:, :, skip:]
    N, K, L = d.shape
    nch = int(np.ceil(L / C))
    pad = nch * C - L
    if pad:
        d = np.concatenate([d, np.full((N, K, pad), np.nan)], 2)
    ch = d.reshape(N, K, nch, C)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cm = np.nanmean(ch, 3)                                    # NaN for chunks past the rollout end
    valid = ~np.isnan(cm)
    G = np.nanmean(cm, 2); GM = np.nanmax(np.where(valid, cm, -np.inf), 2)
    GM[~valid.any(2)] = np.nan
    arg = np.where(valid.any(2), np.nanargmax(np.where(valid, cm, -np.inf), 2), -1)
    return np.nanmean(G, 1), np.nanmean(GM, 1), arg


def boot_ci(scores, member, B, rng):
    N = len(member)
    keys = list(scores)
    reps = {k: [] for k in keys}
    for _ in range(B):
        i = rng.integers(0, N, N)
        for k in keys:
            reps[k].append(auc_low(scores[k][i], member[i]))
    return {k: [float(np.quantile(reps[k], 0.025)), float(np.quantile(reps[k], 0.975))] for k in keys}


def analyse(path, C, B, seed):
    z = np.load(path)
    delta, member = z["delta"].astype(np.float64), z["member"].astype(bool)
    N, K, L = delta.shape
    Lmax = int(z["Lmax"]) if "Lmax" in z.files else L
    lengths = (~np.isnan(delta)).sum(2)
    rng = np.random.default_rng(seed)
    out = dict(file=path, N=int(N), K=int(K), members=int(member.sum()))
    scores = {}
    for skip, tag in ((0, "all tokens"), (1, "drop t=1"), (4, "drop first 4"), (16, "drop first 16"), (C, f"from segment 2 (drop first {C})")):
        G, GM, arg = chunk_stats(delta, skip, C)
        scores[f"G | {tag}"], scores[f"G-Max | {tag}"] = G, GM
        if skip == 0:
            out["share_argmax_chunk1"] = float((arg == 0).mean())
            out["share_argmax_chunk1_in_out"] = [float((arg[member] == 0).mean()), float((arg[~member] == 0).mean())]
    d1 = delta[:, :, 0]
    out["delta1_mean"] = float(np.nanmean(d1))
    out["delta1_sd_between_samples"] = float(np.nanstd(np.nanmean(d1, 1)))
    out["delta1_sd_within_sample_max"] = float(np.nanmax(np.nanstd(d1, 1)))      # should be ~0 if scoring is batch-invariant
    out["delta1_sd_within_sample_median"] = float(np.nanmedian(np.nanstd(d1, 1)))
    out["delta1_in_out"] = [float(np.nanmean(d1[member])), float(np.nanmean(d1[~member]))]
    if "delta0" in z.files:
        d0 = z["delta0"].astype(np.float64)
        G0, GM0, _ = chunk_stats(d0, 0, C)
        G0x, _, _ = chunk_stats(d0, C, C)
        scores["theta0 only (placebo: split difficulty imbalance)"] = G0
        Gt, _, _ = chunk_stats(delta, 0, C); Gtx, _, _ = chunk_stats(delta, C, C)
        scores["CRN difference G_theta - G_theta0 | all tokens"] = Gt - G0
        scores["CRN ratio G_theta / G_theta0 | all tokens"] = Gt / G0
        scores[f"CRN difference | from segment 2"] = Gtx - G0x
        scores["theta0 only | from segment 2"] = G0x
    trunc = lengths >= Lmax
    out["truncation_rate_in_out"] = [float(trunc[member].mean()), float(trunc[~member].mean())]
    aucs = {k: float(auc_low(v, member)) for k, v in scores.items() if not np.isnan(v).any()}
    cis = boot_ci({k: scores[k] for k in aucs}, member, B, rng)
    out["auc"] = {k: dict(auc=aucs[k], ci=cis[k]) for k in aucs}
    full = trunc.mean(1) >= 0.5
    strata = {}
    for name, sel in (("majority of rollouts truncated", full), ("majority finished", ~full)):
        if sel.sum() > 20 and member[sel].sum() > 5 and (~member[sel]).sum() > 5:
            strata[name] = dict(n=int(sel.sum()), **{k: float(auc_low(scores[k][sel], member[sel]))
                                                     for k in ("G | all tokens", "G-Max | all tokens", f"G-Max | from segment 2 (drop first {C})")})
    out["truncation_strata"] = strata
    if "dtheta" in z.files and "delta0" in z.files:
        dt = z["dtheta"].astype(np.float64)
        fr = {}
        for tag, sl in (("all tokens", slice(0, None)), ("without t=1", slice(1, None)), ("without first segment", slice(C, None))):
            num = np.nanmean(dt[member][:, :, sl], (1, 2)); den = np.nanmean(d0[member][:, :, sl], (1, 2))
            fr[tag] = float(1 - np.nanmean(num / den))
        out["f_rm_members"] = fr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+"); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rescore.json")
    a = ap.parse_args()
    res = [analyse(f, a.chunk, a.boot, a.seed) for f in a.files]
    for r in res:
        print(f"\n== {r['file']}  (N={r['N']}, K={r['K']}, members={r['members']}; AUC = P(member score < non-member score))")
        for k, v in r["auc"].items():
            flag = "  *" if (v["ci"][0] > 0.5 or v["ci"][1] < 0.5) else ""
            print(f"  {k:58s} {v['auc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]{flag}")
        print(f"  argmax chunk = 1st: {r['share_argmax_chunk1']:.2f} (in {r['share_argmax_chunk1_in_out'][0]:.2f} / out {r['share_argmax_chunk1_in_out'][1]:.2f})")
        print(f"  Delta_1: mean {r['delta1_mean']:.2f}, between-sample sd {r['delta1_sd_between_samples']:.3f}, "
              f"within-sample sd across rollouts: median {r['delta1_sd_within_sample_median']:.4f}, max {r['delta1_sd_within_sample_max']:.4f}")
        print(f"  truncation rate in / out: {r['truncation_rate_in_out'][0]:.3f} / {r['truncation_rate_in_out'][1]:.3f}")
        for k, v in r["truncation_strata"].items():
            print(f"  stratum '{k}': " + ", ".join(f"{kk} {vv:.3f}" if isinstance(vv, float) else f"{kk} {vv}" for kk, vv in v.items()))
        if "f_rm_members" in r:
            print("  f_rm (members): " + ", ".join(f"{k} {v:+.3f}" for k, v in r["f_rm_members"].items()))
    json.dump(res, open(a.out, "w"), indent=1)
    print("\nwrote", a.out, "  (* = 95% CI excludes 0.5)")


if __name__ == "__main__":
    main()
