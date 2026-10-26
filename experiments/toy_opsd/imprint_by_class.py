"""Where does the membership imprint land, and does the clip push correction sites away from the teacher?

For each clip dose (lr 3e-3, split 0, seed 100 = model 0 of the main grid), train one adapter, roll out K = 4 times
from theta for all 1000 pool samples, and break the per-position quantities down by position class:
  t0      the first generated token (template mismatch: the teacher expects THINK)
  answer  positions right after a TRIG token (correction sites: teacher knows the answer, student does not)
  style   positions where a style token was sampled
  other   everything else
Per class: mean Delta_theta (attack), mean Delta_theta0, and the in/out log-ratio log(mean_out / mean_in).
At answer positions also: log q_theta(correct answer) - log q_theta0(correct answer) under the student context, in vs out
(> 0: moved toward the teacher's answer; < 0: moved away, the M1/M2 direction).
Run: python3 experiments/toy_opsd/imprint_by_class.py  ->  imprint_by_class.json / .out.txt
"""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_opsd as T

model, q, ans = T.load_theta0()
N = len(q)
g = np.random.default_rng(2026)
tr = g.permutation(N)[: N // 2]                          # split 0, identical to run() model 0
mem = np.zeros(N, bool); mem[tr] = True
sctx, tctx = T.student_ctx(q, ans), T.teacher_ctx(q, ans)
OUT = {}
print(f"{'clip':6s} {'class':7s} {'share':>6s} {'D_in':>8s} {'D_out':>8s} {'log out/in':>10s} {'D0_in':>8s} {'D0_out':>8s}")
for clip in (None, 0.2, 0.05, 0.01, 1e-6):
    T.train_opsd(model, q, ans, tr, clip, 3e-3, seed=100)
    gen = torch.Generator().manual_seed(77)
    acc = {c: dict(Din=[], Dout=[], D0in=[], D0out=[], n=0) for c in ("t0", "answer", "style", "other")}
    shift_in, shift_out = [], []
    tot = 0
    with torch.no_grad():
        for k in range(4):
            model.set_adapter(True)
            y = T.sample(model, sctx, gen)
            zs, zt = model(y, sctx), model(y, tctx)
            model.set_adapter(False)
            zs0, zt0 = model(y, sctx), model(y, tctx)
            model.set_adapter(True)
            D, D0 = T.kl_rows(zt, zs).numpy(), T.kl_rows(zt0, zs0).numpy()
            prev = torch.cat([torch.full((N, 1), T.BOS), y[:, :-1]], 1)
            cls = np.full(y.shape, "other", dtype=object)
            cls[torch.isin(y, torch.tensor(T.STYLE)).numpy()] = "style"
            cls[(prev == T.TRIG).numpy()] = "answer"
            cls[:, 0] = "t0"
            tot += y.numel()
            for c in acc:
                m = cls == c
                acc[c]["n"] += int(m.sum())
                acc[c]["Din"].append(D[m & mem[:, None]]); acc[c]["Dout"].append(D[m & ~mem[:, None]])
                acc[c]["D0in"].append(D0[m & mem[:, None]]); acc[c]["D0out"].append(D0[m & ~mem[:, None]])
            # correct-answer log-prob shift at answer positions (student context), theta vs theta0
            trig_count = torch.cumsum((prev == T.TRIG).long(), 1)
            slot = (trig_count - 1) % T.J
            correct = torch.tensor(T.ANSWER)[torch.tensor(ans)[torch.arange(N)[:, None], slot]]
            lq, lq0 = F.log_softmax(zs, -1), F.log_softmax(zs0, -1)
            d = (lq.gather(-1, correct[..., None]) - lq0.gather(-1, correct[..., None]))[..., 0].numpy()
            am = (prev == T.TRIG).numpy()
            shift_in.append(d[am & mem[:, None]]); shift_out.append(d[am & ~mem[:, None]])
    res = {}
    for c, a in acc.items():
        if a["n"] == 0:                                       # e.g. clip 1e-6: the student stops emitting TRIG / style
            res[c] = dict(share=0.0)
            print(f"{str(clip):6s} {c:7s} {0:6.3f}   (class absent from the student's rollouts)", flush=True)
            continue
        Din, Dout = np.concatenate(a["Din"]).mean(), np.concatenate(a["Dout"]).mean()
        D0in, D0out = np.concatenate(a["D0in"]).mean(), np.concatenate(a["D0out"]).mean()
        res[c] = dict(share=a["n"] / tot, D_in=float(Din), D_out=float(Dout), log_out_in=float(np.log(Dout / Din)),
                      D0_in=float(D0in), D0_out=float(D0out))
        print(f"{str(clip):6s} {c:7s} {a['n']/tot:6.3f} {Din:8.4f} {Dout:8.4f} {np.log(Dout/Din):+10.3f} {D0in:8.4f} {D0out:8.4f}", flush=True)
    si, so = np.concatenate(shift_in), np.concatenate(shift_out)
    if si.size == 0:
        OUT[str(clip)] = res
        continue
    res["answer_logq_correct_shift"] = dict(members=float(si.mean()), nonmembers=float(so.mean()),
                                            members_frac_down=float((si < 0).mean()), nonmembers_frac_down=float((so < 0).mean()))
    print(f"{str(clip):6s} answer positions: log q(correct) theta - theta0: members {si.mean():+.3f} "
          f"({100*(si<0).mean():.0f}% down), non-members {so.mean():+.3f} ({100*(so<0).mean():.0f}% down)", flush=True)
    OUT[str(clip)] = res
json.dump(OUT, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "imprint_by_class.json"), "w"), indent=1)
