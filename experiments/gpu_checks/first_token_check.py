"""Check 1: is the large first-token KL (Delta_1) a genuine template effect or a leftover padding bug?

Key fact: Delta_1 = KL(p(.|teacher prompt) || q(.|student prompt)) depends only on the two prompts and the model, not on
the sampled rollout. So it can be recomputed exactly from the prompts alone, and it must be the same for all K rollouts of
a sample.

For each sample this script computes:
  REF   the t = 1 distributions from a batch-size-1 forward pass with no padding (the ground truth), Delta_1 in fp32 log
        space from an fp32 LM head (and from the bf16 logits, for comparison), at T = 1 and T = 1.1;
  DECOMP teacher top-5 tokens and probabilities, p_T(<think>), -log q_S(<think>), H(p_T), and the top entry's KL share.
        Genuine template mismatch => teacher top-1 is <think> with p ~ 1 and Delta_1 ~ -log q_S(<think>);
  PAD   the same Delta_1 through batched left padding with extra pads {0, 13, 128}, with and without explicit
        position_ids, and through the OPSD right-pad-then-concatenate read-out (the earlier bug), against REF;
  PIPE  (optional) your pipeline's stored Delta_1 per rollout: deviation from REF and spread across rollouts.
Verdict per sample: genuine if REF decomposition shows <think> dominance, PAD variants match REF, and PIPE matches REF.

Inputs
  --prompts   JSONL, one line per sample: {"id": ..., "student_prompt": "...", "teacher_prompt": "..."} - the exact
              rendered strings your pipeline tokenizes (chat template already applied).
  --pipeline  optional JSON {id: [Delta_1 of rollout 1, ..., rollout K]} from your scorer.
  --model     base model path or id;  --adapter optional PEFT adapter (adds theta rows: OP-Print Delta with the adapter
              on both sides, and D_theta with the theta0 teacher).
Example
  python3 first_token_check.py --model Qwen/Qwen3-1.7B --prompts prompts_main.jsonl --pipeline delta1_main.json \
      --adapter runs/main/adapter --n 32 --out first_token_main.json
Runs in a few minutes on one GPU (2 forward passes per prompt per variant). CPU works for a handful of samples.
"""
import argparse, json, math, sys
import torch
import torch.nn.functional as F


def kl_log1p(lp, lq):
    """KL(p||q) from log-probs, fp32 log-space form: log1p(sum_i p_i (expm1(d_i) - d_i)), d = (lq - lp) centred under p."""
    lp, lq = lp.double(), lq.double()
    p = lp.exp()
    delta = lq - lp
    d = delta - (p * delta).sum(-1, keepdim=True)
    return torch.log1p((p * (torch.expm1(d) - d)).sum(-1)).float()


@torch.no_grad()
def last_logits(model, seqs, mode, pad_id, extra_pad=0, fp32_head=True):
    """Logits predicting the first generated token, for a list of token-id lists.
    mode: 'single'           batch size 1, no padding (reference)
          'left_pos'         left padding (+extra_pad), position_ids from the attention mask
          'left_nopos'       left padding (+extra_pad), no position_ids passed
          'right_concat_bug' OPSD collator pattern: right-pad prompts to the batch max, read the batch-max position."""
    dev = next(model.parameters()).device
    head = model.get_output_embeddings().weight
    outs = []

    def run(ids, mask=None, pos=None):
        kw = dict(input_ids=ids, attention_mask=mask)
        if pos is not None:
            kw["position_ids"] = pos
        h = model.model(**kw).last_hidden_state
        return (h.float() @ head.float().T) if fp32_head else model.lm_head(h).float()

    if mode == "single":
        for s in seqs:
            ids = torch.tensor([s], device=dev)
            outs.append(run(ids)[0, -1])
        return torch.stack(outs)
    L = max(len(s) for s in seqs) + extra_pad
    if mode in ("left_pos", "left_nopos"):
        ids = torch.full((len(seqs), L), pad_id, device=dev)
        mask = torch.zeros((len(seqs), L), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s, device=dev); mask[i, L - len(s):] = 1
        pos = (mask.cumsum(-1) - 1).clamp(min=0) if mode == "left_pos" else None
        return run(ids, mask, pos)[:, -1]
    if mode == "right_concat_bug":
        ids = torch.full((len(seqs), L + 1), pad_id, device=dev)
        mask = torch.zeros((len(seqs), L + 1), dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, device=dev); mask[i, :len(s)] = 1
        ids[:, L] = seqs[0][-1]; mask[:, L] = 1                      # a stand-in 'first generated token' after the pads
        return run(ids, mask)[:, L - 1]                               # batch-level prompt length - 1, as in opsd_trainer
    raise ValueError(mode)


def decompose(lpT, lqS, think_id, tok, k=5):
    pT = lpT.exp()
    top = torch.topk(pT, k)
    kl = float(kl_log1p(lpT[None], lqS[None])[0])
    i0 = int(top.indices[0])
    share = float(pT[i0] * (lpT[i0] - lqS[i0])) / kl if kl > 0 else float("nan")
    return dict(teacher_top=[(tok(int(i)), round(float(p), 4)) for p, i in zip(top.values, top.indices)],
                p_T_think=float(pT[think_id]) if think_id is not None else None,
                neg_log_q_S_think=float(-lqS[think_id]) if think_id is not None else None,
                H_T=float(-(pT * lpT).sum()), top1_kl_share=share, delta1=kl)


def check(model, tok_ids_T, tok_ids_S, pad_id, think_id, tok, temps=(1.0, 1.1), batch=8):
    """Returns per-sample results for one model state (theta0 or theta)."""
    res = [dict() for _ in tok_ids_T]
    zT = last_logits(model, tok_ids_T, "single", pad_id)
    zS = last_logits(model, tok_ids_S, "single", pad_id)
    zTb = last_logits(model, tok_ids_T, "single", pad_id, fp32_head=False)
    zSb = last_logits(model, tok_ids_S, "single", pad_id, fp32_head=False)
    for T in temps:
        d = kl_log1p(F.log_softmax(zT / T, -1), F.log_softmax(zS / T, -1))
        db = kl_log1p(F.log_softmax(zTb / T, -1), F.log_softmax(zSb / T, -1))
        for i in range(len(res)):
            res[i][f"ref_delta1_T{T:g}"] = float(d[i]); res[i][f"ref_delta1_bf16logits_T{T:g}"] = float(db[i])
    lpT, lqS = F.log_softmax(zT, -1), F.log_softmax(zS, -1)
    for i in range(len(res)):
        res[i]["decomp"] = decompose(lpT[i], lqS[i], think_id, tok)
    for mode, extra in (("left_pos", 0), ("left_pos", 13), ("left_pos", 128), ("left_nopos", 13), ("right_concat_bug", 0)):
        for b0 in range(0, len(res), batch):
            sl = slice(b0, b0 + batch)
            zt = last_logits(model, tok_ids_T[sl], mode, pad_id, extra)
            zs = last_logits(model, tok_ids_S[sl], mode, pad_id, extra)
            d = kl_log1p(F.log_softmax(zt, -1), F.log_softmax(zs, -1))
            for j, i in enumerate(range(len(res))[sl]):
                res[i][f"pad_{mode}_{extra}"] = float(d[j]) - res[i]["ref_delta1_T1"]
    return res


@torch.no_grad()
def check_D_theta(peft_model, hf, tok_ids_T, tok_ids_S, pad_id):
    """D_theta at t = 1: theta0 teacher (adapter disabled) vs theta student, batch size 1."""
    with peft_model.disable_adapter():
        zT = last_logits(hf, tok_ids_T, "single", pad_id)
    zS = last_logits(hf, tok_ids_S, "single", pad_id)
    return kl_log1p(F.log_softmax(zT, -1), F.log_softmax(zS, -1)).tolist()


def summarize(rows, pipeline, tol=0.05):
    import statistics as st
    out = {}
    d = [r["ref_delta1_T1"] for r in rows]
    out["ref_delta1_mean"] = st.mean(d); out["ref_delta1_sd"] = st.pstdev(d)
    pt = [r["decomp"]["p_T_think"] for r in rows if r["decomp"]["p_T_think"] is not None]
    if pt:
        out["share_teacher_top1_is_think"] = st.mean([r["decomp"]["teacher_top"][0][0].strip() == "<think>" for r in rows])
        out["p_T_think_median"] = st.median(pt)
        out["delta1_over_neglogq_think_median"] = st.median([r["ref_delta1_T1"] / r["decomp"]["neg_log_q_S_think"] for r in rows])
    for k in [k for k in rows[0] if k.startswith("pad_")]:
        out[f"max_abs_{k}"] = max(abs(r[k]) for r in rows)
    if pipeline:
        devs, sds = [], []
        for r in rows:
            vals = pipeline.get(str(r["id"]))
            if vals:
                devs.append(abs(st.mean(vals) - r["ref_delta1_T1"])); sds.append(st.pstdev(vals) if len(vals) > 1 else 0.0)
        if devs:
            out["pipeline_max_abs_dev_from_ref"] = max(devs); out["pipeline_max_sd_across_rollouts"] = max(sds)
    verdict = {
        "pad_left_pos_invariant": max(out[k] for k in out if k.startswith("max_abs_pad_left_pos")) <= tol,
        "pipeline_matches_ref": (out.get("pipeline_max_abs_dev_from_ref", 0) <= tol + 0.01 * out["ref_delta1_mean"]) if pipeline else None,
        "pipeline_constant_across_rollouts": (out.get("pipeline_max_sd_across_rollouts", 0) <= tol) if pipeline else None,
        "think_dominates_teacher": (out.get("share_teacher_top1_is_think", 0) >= 0.9) if pt else None,
    }
    out["verdict"] = verdict
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--adapter")
    ap.add_argument("--prompts", required=True); ap.add_argument("--pipeline")
    ap.add_argument("--n", type=int, default=32); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--add-special-tokens", type=int, default=0, help="match your pipeline's tokenizer call")
    ap.add_argument("--out", default="first_token_check.json")
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev).eval()
    rows = [json.loads(l) for l in open(a.prompts)][: a.n]
    enc = lambda s: tok(s, add_special_tokens=bool(a.add_special_tokens))["input_ids"]
    T_ids, S_ids = [enc(r["teacher_prompt"]) for r in rows], [enc(r["student_prompt"]) for r in rows]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    think = tok.convert_tokens_to_ids("<think>")
    think = think if isinstance(think, int) and think != tok.unk_token_id else None
    pipeline = json.load(open(a.pipeline)) if a.pipeline else None
    run_checks(model, a.adapter, T_ids, S_ids, pad_id, think, tok.decode, pipeline, a.out, rows)


def run_checks(model, adapter_path, T_ids, S_ids, pad_id, think, decode, pipeline, out_path, rows):
    """theta0 always; with an adapter also theta (OP-Print Delta, adapter on both sides) and D_theta at t = 1.
    LoRA layers live inside the HF model, so all forwards use the inner HF model; the PEFT wrapper only toggles them."""
    result = {}
    if adapter_path:
        from peft import PeftModel
        peft_model = PeftModel.from_pretrained(model, adapter_path).eval()
        hf = peft_model.base_model.model
        with peft_model.disable_adapter():
            states = {"theta0": check(hf, T_ids, S_ids, pad_id, think, decode)}
        states["theta"] = check(hf, T_ids, S_ids, pad_id, think, decode)
        result["D_theta_t1"] = check_D_theta(peft_model, hf, T_ids, S_ids, pad_id)
        pipe_state = "theta"
    else:
        states = {"theta0": check(model, T_ids, S_ids, pad_id, think, decode)}
        pipe_state = "theta0"
    for name, res in states.items():
        for r, row in zip(res, rows):
            r["id"] = row.get("id")
        result[name] = dict(summary=summarize(res, pipeline if name == pipe_state else None), samples=res)
        print(name, json.dumps(result[name]["summary"], indent=1), flush=True)
    json.dump(result, open(out_path, "w"), indent=1)
    print("wrote", out_path)
    return result




if __name__ == "__main__":
    main()
