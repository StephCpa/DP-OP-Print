"""[DPOP] Export H0 per-token scores (T=1.1) + exact prompts for experiments/gpu_checks/{gmax_rescore,first_token_check}.py. CPU only."""
import json, os, sys, glob, collections
import numpy as np, torch
P = os.path.expanduser("~/dp-op-print"); sys.path.insert(0, f"{P}/opsd-dpop")
from datasets import load_dataset
from transformers import AutoTokenizer
from data_collator import SelfDistillationDataCollator
OUT = f"{P}/runs/h0_diag/export"; os.makedirs(OUT, exist_ok=True)
KMAX = int(os.environ.get("KMAX", 8))
tok = AutoTokenizer.from_pretrained(f"{P}/models/Qwen3-1.7B"); ds = load_dataset(f"{P}/data/Openthoughts_math_30k_opsd")["train"]
report = {}
for arm, tt in (("primary", True), ("control", False)):
    R = f"{P}/runs/h0_{arm}_split0A"
    recs = {}
    for l in open(f"{R}/score/records.jsonl"):
        r = json.loads(l); recs[(r["idx"], r["k"])] = r
    finish = collections.Counter(r["finish"] for r in recs.values())
    ks = collections.Counter(collections.Counter(i for i, _ in recs).values())
    idxs = sorted({i for i, _ in recs}); N = len(idxs); pos = {i: n for n, i in enumerate(idxs)}
    lens = [r["len"] for r in recs.values()]; L = max(lens)
    trunc_lens = sorted({r["len"] for r in recs.values() if r["finish"] == "length"})
    Lmax = max(trunc_lens) if trunc_lens else L
    arr = {k: np.full((N, KMAX, L), np.nan, np.float32) for k in ("delta", "delta0", "dtheta")}
    src = {"delta": "theta_T1.1", "delta0": "theta0_T1.1", "dtheta": "dtrain_T1.1"}
    for f in sorted(glob.glob(f"{R}/score/tok_*.pt")):
        sh = torch.load(f, weights_only=False)
        for (i, k), d in sh.items():
            if k >= KMAX: continue
            for a_, s in src.items():
                x = d[s].float().numpy(); arr[a_][pos[i], k, :len(x)] = x
        del sh
    member = np.array([recs[(i, 0)]["member"] for i in idxs], np.int8).astype(bool)
    missing = int(np.isnan(arr["delta"][:, :, 0]).sum())
    np.savez_compressed(f"{OUT}/h0_{arm}_T1.1.npz", delta=arr["delta"].astype(np.float16), delta0=arr["delta0"].astype(np.float16),
                        dtheta=arr["dtheta"].astype(np.float16), member=member, idx=np.array(idxs, np.int32), Lmax=np.int32(Lmax))
    # prompts + stored first-token KL
    col = SelfDistillationDataCollator(tok, max_length=20000, reason_first=False, student_thinking=False, teacher_thinking=tt, prompt_padding_side="left")
    pipe = {}; rt_bad = 0; spread = dict(theta_T1_0=[], theta_T1_1=[], theta0_T1_0=[], theta0_T1_1=[])
    with open(f"{OUT}/h0_{arm}_prompts.jsonl", "w") as fo:
        for i in idxs:
            b = col([ds[i]])
            s_ids = b["student_prompts"][0][b["student_prompt_attention_mask"][0].bool()].tolist()
            t_ids = b["teacher_prompts"][0][b["teacher_prompt_attention_mask"][0].bool()].tolist()
            sp, tp = tok.decode(s_ids, skip_special_tokens=False), tok.decode(t_ids, skip_special_tokens=False)
            rt_bad += (tok(sp, add_special_tokens=False)["input_ids"] != s_ids) + (tok(tp, add_special_tokens=False)["input_ids"] != t_ids)
            kk = sorted(k for (j, k) in recs if j == i)
            d1 = {f"{p}_T{T}": [recs[(i, k)][f"{p}_T{T}"]["d1"] for k in kk] for p in ("theta", "theta0", "dtrain") for T in (1.0, 1.1)}
            for p in ("theta", "theta0"):
                for T in (1.0, 1.1):
                    v = d1[f"{p}_T{T}"]; spread[f"{p}_T{str(T).replace('.', '_')}"].append(max(v) - min(v))
            pipe[str(i)] = d1["theta_T1.0"]
            fo.write(json.dumps(dict(id=i, member=int(recs[(i, 0)]["member"]), student_prompt=sp, teacher_prompt=tp,
                                     student_prompt_ids=s_ids, teacher_prompt_ids=t_ids, k=kk, delta1=d1), ensure_ascii=False) + "\n")
    json.dump(pipe, open(f"{OUT}/h0_{arm}_delta1_theta_T1.0.json", "w"))
    sp_ = {k: dict(max=float(np.max(v)), median=float(np.median(v)), mean=float(np.mean(v))) for k, v in spread.items()}
    report[arm] = dict(N=N, K_hist={str(a): b for a, b in ks.items()}, K_exported=KMAX, L=L, Lmax=Lmax, finish=dict(finish),
                       members=int(member.sum()), missing_rollouts_in_npz=missing, prompt_roundtrip_mismatches=int(rt_bad),
                       delta1_spread_across_K=sp_)
    print(arm, json.dumps(report[arm]), flush=True)
json.dump(report, open(f"{OUT}/export_report.json", "w"), indent=1)
print("EXPORT_DONE")
