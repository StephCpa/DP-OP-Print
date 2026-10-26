"""Runs the pre-registered analysis (design v1.2/v1.3) on the toy OPSD arms in experiments/toy_opsd/runs/.

Per arm (8 models = 4 splits + complements, same splits in every arm):
  H0   AUC of G, G-Max (and variants) averaged over models, 95% CI by sample bootstrap (same resample for all models).
  H3   gap = AUC(G-Max) - AUC(G); paired bootstrap of gap differences between arms (same samples, same splits).
       Sign-agnostic attacks: |G - median_out| and a learned logistic aggregator on the sorted CRN chunk vector of
       Delta_theta - Delta_theta0 (trained on models 0-1, tested on 2-7). AUC < 0.5 is reported as is.
  H1'  CRN reference-calibrated scores: S_theta - S_theta0, ratio rho, theta0-teacher arm D_theta, and Delta_theta - D_theta.
  H1   v(x) = log(Dbar_out / Dbar_in) after per-model multiplicative out-calibration; split-half reliability; incremental
       R^2 and HC3 t of c (top-10% chunk share), m_U (v1.2), m_U+ and r_dir (proposed) over d; v1.2 sign-flip placebo.
  §1.0 class of the token at each rollout's argmax-Delta position (members vs non-members), argmax-chunk index, Delta_1.
Usage: python3 experiments/toy_opsd/analyze.py  ->  experiments/toy_opsd/results.json and results.md
       TOY_RUNS=experiments/toy_opsd/runs_lowcap OUT_SUFFIX=_lowcap python3 experiments/toy_opsd/analyze.py
"""
import glob, json, os
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("TOY_RUNS", os.path.join(HERE, "runs"))
SUFFIX = os.environ.get("OUT_SUFFIX", "")
rng = np.random.default_rng(0)
NB = int(os.environ.get("NB", 400))


def auc(s, y):
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_arm(tag):
    d = os.path.join(RUNS, tag)
    files = sorted(glob.glob(os.path.join(d, "model*.npz")), key=lambda f: int(os.path.basename(f)[5:-4]))
    if len(files) < 8:
        return None
    M = [dict(np.load(f)) for f in files]
    meta = json.load(open(os.path.join(d, "meta.json")))
    return M, meta


def scores(m):
    Dm, D0, Dt = m["D_theta_attack"], m["D_theta0"], m["D_theta0teacher"]
    s = {
        "G": -Dm.mean(1),
        "G-Max": -m["D_theta_attack_max"],
        "G (no chunk 1)": -Dm[:, 1:].mean(1),
        "G-Max (no chunk 1)": -m["D_theta_attack_max_x1"],
        "theta0 only (difficulty)": -D0.mean(1),
        "H1' S_theta - S_theta0": -(Dm.mean(1) - D0.mean(1)),
        "H1' ratio rho": -(Dm.mean(1) / D0.mean(1)),
        "H1' D_theta (theta0 teacher)": -Dt.mean(1),
        "H1' Delta_theta - D_theta": -(Dm.mean(1) - Dt.mean(1)),
    }
    out = ~m["member"]
    s["|G - median_out|"] = np.abs(s["G"] - np.median(s["G"][out]))
    s["|G-Max - median_out|"] = np.abs(s["G-Max"] - np.median(s["G-Max"][out]))
    return s


def arm_aucs(M, idx=None):
    res = {}
    for k in scores(M[0]):
        vals = []
        for m in M:
            s, y = scores(m)[k], m["member"]
            if idx is not None:
                s, y = s[idx], y[idx]
            vals.append(auc(s, y))
        res[k] = np.array(vals)
    return res


def learned_aggregator(M, idx=None):
    X = [m["diff_sortedchunks"] for m in M]
    y = [m["member"] for m in M]
    clf = LogisticRegression(max_iter=2000).fit(np.vstack(X[:2]), np.concatenate(y[:2]))
    a = []
    for Xm, ym in zip(X[2:], y[2:]):
        if idx is not None:
            Xm, ym = Xm[idx], ym[idx]
        a.append(auc(clf.decision_function(Xm), ym))
    return float(np.mean(a))


def boot(M, n=NB):
    """Bootstrap over samples: mean-over-models AUC for every score, same resample for all models."""
    N = len(M[0]["member"])
    reps = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        reps.append({k: v.mean() for k, v in arm_aucs(M, idx).items()})
    return reps


# ------------------------------------------------------------------ H1 machinery
def v_outcome(M, key="D_theta_attack", agg="mean"):
    N = len(M[0]["member"])
    vals = np.zeros((8, N)); mem = np.zeros((8, N), bool)
    for i, m in enumerate(M):
        D = m[key].mean(1) if agg == "mean" else m[key + "_max"]
        mem[i] = m["member"]
        vals[i] = D / np.median(D[~m["member"]])                    # multiplicative out-calibration
    din = (vals * mem).sum(0) / mem.sum(0); dout = (vals * ~mem).sum(0) / (~mem).sum(0)
    return np.log(dout / din), vals, mem


def split_half(vals, mem):
    rs = []
    for pairs in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
        vv = []
        for ps in pairs:
            ms = [2 * p for p in ps] + [2 * p + 1 for p in ps]
            V_, Mm = vals[ms], mem[ms]
            vv.append(np.log(((V_ * ~Mm).sum(0) / (~Mm).sum(0)) / ((V_ * Mm).sum(0) / Mm.sum(0))))
        rs.append(np.corrcoef(vv[0], vv[1])[0, 1])
    return float(np.mean(rs))


def hc3(X, y, j):
    XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ y; e = y - X @ b
    H = np.einsum("ij,jk,ik->i", X, XtX, X)
    V = XtX @ ((X * (e / (1 - H))[:, None] ** 2).T @ X) @ XtX
    return b, b[j] / np.sqrt(V[j, j])


def r2(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def z(x):
    return (x - x.mean()) / x.std()


def h1(M, feats, clip_key):
    v, vals, mem = v_outcome(M)
    N = len(v)
    D0 = feats["D0"]                                                  # (N, K, nch) theta0 rollouts
    tot = D0.sum(2)
    k_top = int(np.ceil(0.1 * D0.shape[2]))
    c = (np.sort(D0, 2)[:, :, ::-1][:, :, :k_top].sum(2) / tot).mean(1)
    dbar0 = D0.mean((1, 2))
    H = feats["hidden"] - feats["hidden"].mean(0)
    lev = np.einsum("ij,jk,ik->i", H, np.linalg.pinv(H.T @ H), H)
    d = np.column_stack([z(np.log(dbar0)), z(feats["ans_logprob"]), z(lev)])
    comp = {"c (top-10% chunk share)": c}
    if clip_key is None:                                              # clip off: m_U reduces to Dbar_theta0 (in d)
        comp["m_U (= Dbar_theta0 at clip 0)"] = dbar0
    else:
        comp["m_U (v1.2)"] = feats[f"m_U@{clip_key}"] if f"m_U@{clip_key}" in feats else feats["m_U"]
        comp["m_U+ (positive unclamped)"] = feats[f"m_U_pos@{clip_key}"] if f"m_U_pos@{clip_key}" in feats else feats["m_U_pos"]
        comp["r_dir (proposed)"] = feats[f"r_dir@{clip_key}"] if f"r_dir@{clip_key}" in feats else feats["r_dir"]
    X0 = np.column_stack([np.ones(N), d])
    base = r2(X0, v)
    res = dict(reliability_split_half=split_half(vals, mem), v_mean=float(v.mean()), v_sd=float(v.std()),
               R2_d=float(base), frac_v_positive=float((v > 0).mean()), competitors={})
    pairdiff = np.stack([np.log(np.where(mem[2 * k], vals[2 * k + 1] / vals[2 * k], vals[2 * k] / vals[2 * k + 1])) for k in range(4)])
    for name, x in comp.items():
        x = np.asarray(x)
        if x.ndim > 1:
            x = x.mean(1)
        X = np.column_stack([np.ones(N), z(x), d])
        b, t = hc3(X, v, 1)
        # v1.2 placebo: within-pair sign flips of log(out/in) pair differences (approximation of v on the log scale)
        w = (np.linalg.inv(X.T @ X) @ X.T)[1]
        eps = rng.choice([-1.0, 1.0], size=(1000, 4, N))
        plac = ((eps * pairdiff[None]).sum(1) / 4) @ w
        res["competitors"][name] = dict(coef=float(b[1]), t_hc3=float(t), p_hc3_two_sided=float(2 * norm.sf(abs(t))),
                                        incr_R2=float(r2(X, v) - base), placebo_p_one_sided=float((plac >= b[1]).mean()),
                                        spearman_with_v=float(np.corrcoef(rankdata(x), rankdata(v))[0, 1]))
    return res


def diag(M):
    cls_names = ["content", "style", "answer", "THINK", "TRIG"]
    out = {}
    for who in ("members", "non-members"):
        tok, chunk0 = [], []
        for m in M:
            sel = m["member"] if who == "members" else ~m["member"]
            tok.append(m["argmax_tok_cls"][sel].ravel()); chunk0.append((m["argmax_chunk"][sel] == 0).ravel())
        tok = np.concatenate(tok)
        out[who] = {f"argmax token = {n}": float((tok == i).mean()) for i, n in enumerate(cls_names)}
        out[who]["argmax chunk = 1st"] = float(np.concatenate(chunk0).mean())
    out["delta1_mean"] = float(np.mean([m["delta1"].mean() for m in M]))
    out["first_token_THINK_rate"] = float(np.mean([m["first_is_think"].mean() for m in M]))
    out["clamped_frac_at_argmax"] = float(np.mean([m["argmax_clamped_frac"].mean() for m in M]))
    out["f_rm_train"] = float(np.mean([1 - (m["D_theta0teacher"].mean(1)[m["member"]] / m["D_theta0"].mean(1)[m["member"]]).mean() for m in M]))
    return out


def main():
    order = ["clip-none", "clip-0.2", "clip-0.05", "clip-0.01", "clip-1e-6", "clip-none_nothink", "clip-0.05_nothink",
             "clip-none_lr3e-4", "clip-0.05_lr3e-4"]
    found = sorted(os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(RUNS, "*", "meta.json")))
    arms = [a for a in order if a in found] + [a for a in found if a not in order]
    feats = {k: dict(np.load(os.path.join(RUNS, f"theta0_features{k}.npz"))) for k in ("", "_nothink")}
    R, BOOT, loaded = {}, {}, {}
    for tag in arms:
        L = load_arm(tag)
        if L is None:
            print("missing", tag); continue
        M, meta = L
        loaded[tag] = M
        A = arm_aucs(M)
        B = boot(M)
        res = dict(meta={k: meta[k] for k in ("clip", "lr", "think")},
                   training=dict(clip_rate=float(np.mean([x["clip_rate"] for x in meta["models"]])),
                                 loss_first=float(np.mean([x["loss_first"] for x in meta["models"]])),
                                 loss_last=float(np.mean([x["loss_last"] for x in meta["models"]]))),
                   auc={k: dict(mean=float(v.mean()), sd_models=float(v.std()), ci=[float(np.quantile([b[k] for b in B], q)) for q in (0.025, 0.975)])
                        for k, v in A.items()})
        res["auc"]["learned aggregator (2->6)"] = dict(mean=learned_aggregator(M))
        gaps = np.array([b["G-Max"] - b["G"] for b in B])
        res["gap"] = dict(mean=float(A["G-Max"].mean() - A["G"].mean()), ci=[float(np.quantile(gaps, 0.025)), float(np.quantile(gaps, 0.975))])
        gx = np.array([b["G-Max (no chunk 1)"] - b["G (no chunk 1)"] for b in B])
        res["gap_no_chunk1"] = dict(mean=float(A["G-Max (no chunk 1)"].mean() - A["G (no chunk 1)"].mean()),
                                    ci=[float(np.quantile(gx, 0.025)), float(np.quantile(gx, 0.975))])
        clip_key = None if meta["clip"] is None else f"{meta['clip']:g}"
        res["H1"] = h1(M, feats["" if meta["think"] else "_nothink"], clip_key)
        res["diag"] = diag(M)
        R[tag] = res
        print(tag, json.dumps({k: round(v["mean"], 3) for k, v in res["auc"].items()}), "gap", round(res["gap"]["mean"], 4), flush=True)
    # paired bootstrap of gap differences between arms (same samples/splits): resample once, apply to both arms
    N = 1000
    pairs = [("clip-none", "clip-0.05"), ("clip-none", "clip-0.2"), ("clip-none", "clip-1e-6"),
             ("clip-none_nothink", "clip-0.05_nothink"), ("clip-none_lr3e-4", "clip-0.05_lr3e-4")]
    R["_gap_differences"] = {}
    for a, b in pairs:
        if a not in loaded or b not in loaded:
            continue
        diffs = []
        for _ in range(NB):
            idx = rng.integers(0, N, N)
            A1, A2 = arm_aucs(loaded[a], idx), arm_aucs(loaded[b], idx)
            diffs.append((A1["G-Max"].mean() - A1["G"].mean()) - (A2["G-Max"].mean() - A2["G"].mean()))
        point = R[a]["gap"]["mean"] - R[b]["gap"]["mean"]
        R["_gap_differences"][f"{a} minus {b}"] = dict(diff=float(point), ci=[float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))])
    dose = [t for t in ("clip-none", "clip-0.2", "clip-0.05", "clip-0.01", "clip-1e-6") if t in R]
    gaps = [R[t]["gap"]["mean"] for t in dose]
    R["_dose_trend_spearman"] = float(np.corrcoef(rankdata(range(len(dose))), rankdata(gaps))[0, 1]) if len(dose) > 2 else None
    json.dump(R, open(os.path.join(HERE, f"results{SUFFIX}.json"), "w"), indent=1)
    write_md(R)


def write_md(R):
    L = ["# Toy OPSD dry run: results", "",
         "Generated by `experiments/toy_opsd/analyze.py` from the arms in `run_all.sh`. A synthetic system, not Qwen3: see",
         "`docs/results/research-report-2026-10-26.md` §R5 for what it can and cannot show.", ""]
    arms = [k for k in R if not k.startswith("_")]
    keys = ["G", "G-Max", "G-Max (no chunk 1)", "|G - median_out|", "learned aggregator (2->6)", "H1' S_theta - S_theta0",
            "H1' ratio rho", "H1' D_theta (theta0 teacher)", "H1' Delta_theta - D_theta", "theta0 only (difficulty)"]
    L += ["## AUC by arm (mean over 8 models; 95% bootstrap CI for G and G-Max)", "",
          "| arm | clip | lr | f_rm | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 4)]
    for a in arms:
        r = R[a]
        cells = []
        for k in keys:
            x = r["auc"][k]
            cells.append(f"{x['mean']:.3f}" + (f" [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]" if k in ("G", "G-Max") else ""))
        L.append(f"| {a} | {r['meta']['clip']} | {r['meta']['lr']:g} | {r['diag']['f_rm_train']:+.2f} | " + " | ".join(cells) + " |")
    L += ["", "## H3: G-Max - G gap", "", "| arm | gap [95% CI] | gap without chunk 1 [95% CI] |", "|---|---|---|"]
    for a in arms:
        g, gx = R[a]["gap"], R[a]["gap_no_chunk1"]
        L.append(f"| {a} | {g['mean']:+.4f} [{g['ci'][0]:+.4f}, {g['ci'][1]:+.4f}] | {gx['mean']:+.4f} [{gx['ci'][0]:+.4f}, {gx['ci'][1]:+.4f}] |")
    L += ["", "Paired differences (same samples and splits):", ""]
    for k, v in R.get("_gap_differences", {}).items():
        L.append(f"- {k}: {v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]")
    L.append(f"- Spearman(dose order none → 1e-6, gap) over the 5 main arms: {R.get('_dose_trend_spearman')}")
    L += ["", "## H1: ex-ante prediction of v(x)", "",
          "| arm | split-half r | R² of d | competitor | coef (z) | HC3 t | incr. R² | v1.2 placebo p |", "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        h = R[a]["H1"]
        for i, (n, c) in enumerate(h["competitors"].items()):
            head = f"| {a} | {h['reliability_split_half']:.2f} | {h['R2_d']:.3f} " if i == 0 else "| | | "
            L.append(head + f"| {n} | {c['coef']:+.4f} | {c['t_hc3']:+.2f} | {c['incr_R2']:.4f} | {c['placebo_p_one_sided']:.3f} |")
    L += ["", "## §1.0 diagnostic", "",
          "| arm | Δ₁ mean | first token THINK | argmax chunk = 1st (in / out) | argmax token = answer (in / out) | = style (in / out) | clamped share at argmax |",
          "|---|---|---|---|---|---|---|"]
    for a in arms:
        dg = R[a]["diag"]
        L.append(f"| {a} | {dg['delta1_mean']:.2f} | {dg['first_token_THINK_rate']:.3f} | "
                 f"{dg['members']['argmax chunk = 1st']:.2f} / {dg['non-members']['argmax chunk = 1st']:.2f} | "
                 f"{dg['members']['argmax token = answer']:.2f} / {dg['non-members']['argmax token = answer']:.2f} | "
                 f"{dg['members']['argmax token = style']:.2f} / {dg['non-members']['argmax token = style']:.2f} | {dg['clamped_frac_at_argmax']:.2f} |")
    open(os.path.join(HERE, f"results{SUFFIX}.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
