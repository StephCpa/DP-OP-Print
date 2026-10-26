"""Toy OPSD simulator: a CPU dry run of the pre-registered pipeline (OPSD training -> OP-Print scoring -> H0/H1/H1'/H3).

This is NOT evidence about Qwen3. It is a small, fully specified system with the same *mechanics* as OPSD + OP-Print:
  * frozen base theta0 (an MLP language model pre-trained on a synthetic process), LoRA adapters on every linear layer;
  * teacher = theta0 with a privileged context: the sample's answers r, a 'reflective' instruction that raises
    style-token probability, and a 'thinking' flag that makes the first token THINK (the template mismatch of
    docs/verification/opsd-code-audit.md §2.2-2.3);
  * student = theta (adapter on) with the plain context; on-policy rollouts at T = 1.1, top-k 20, top-p 0.95;
  * loss = OPSD generalized_jsd_loss, beta = 0, at temperature 1.1, element-wise clamp(max=clip) before the vocab sum,
    mean over tokens; Adam, max grad norm 0.1, effective batch 32 prompts x 1 rollout, 3 epochs (48 steps at N = 500);
  * scoring: K = 8 rollouts from theta, Delta_t = KL(p_theta(.|teacher ctx) || p_theta(.|student ctx)) at T = 1,
    16-token chunks (8 chunks per 128-token rollout), G and G-Max; CRN theta0 and theta0-teacher (D_theta) arms.
Usage:
  python3 experiments/toy_opsd/toy_opsd.py pretrain            # builds theta0 + world, ~2 min
  python3 experiments/toy_opsd/toy_opsd.py run --clip 0.05     # 4 splits + complements at one clip ('none' = off)
  python3 experiments/toy_opsd/toy_opsd.py theta0              # ex-ante theta0 features (c, m_U, r_dir, d)
Environment: TOY_DQ (default 64), TOY_RANK (16), TOY_KAPPA (1.0), TOY_RUNS (output dir). The low-capacity arm in
docs/results uses TOY_DQ=16 TOY_RANK=4 TOY_RUNS=experiments/toy_opsd/runs_lowcap.
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.environ.get("TOY_RUNS", os.path.join(HERE, "runs"))
torch.set_num_threads(4)

# ------------------------------------------------------------------ world
V = 64
CONTENT = np.r_[0:40, 58:64]          # 46 'math/content' tokens
STYLE = np.arange(40, 48)             # 8 'style' tokens (wait, hmm, ...)
ANSWER = np.arange(48, 56)            # 8 answer tokens
TRIG, THINK, BOS = 56, 57, 64         # BOS is an input-only id
J, TLEN, CHUNK = 3, 128, 16
DQ = int(os.environ.get("TOY_DQ", 64))     # question-feature dim; 64 lets a rank-16 adapter memorise samples
RANK = int(os.environ.get("TOY_RANK", 16))  # LoRA rank (alpha = 2 * rank, as OPSD's 128/64)
P_TRIG, P_STYLE_PLAIN, P_STYLE_REFL, P_THINK = 0.04, 0.03, 0.12, 0.98
KAPPA = float(os.environ.get("TOY_KAPPA", 1.0))                           # answer noise: a = argmax(A_j q + KAPPA * xi)


class World:
    def __init__(self, seed=0):
        g = np.random.default_rng(seed)
        self.B = g.dirichlet(np.full(len(CONTENT), 0.3), size=V + 1)      # next-content distribution given prev token
        self.S = g.dirichlet(np.full(len(STYLE), 1.0))                    # style-token distribution
        self.A = g.normal(0, 1 / math.sqrt(DQ), (J, len(ANSWER), DQ))
        self.start = g.dirichlet(np.full(len(CONTENT), 0.5))

    def samples(self, n, g):
        q = g.normal(0, 1, (n, DQ))
        xi = g.normal(0, 1, (n, J, len(ANSWER)))
        ans = np.argmax(np.einsum("jad,nd->nja", self.A, q) + KAPPA * xi, -1)   # (n, J) in 0..7
        return q.astype(np.float32), ans

    def generate(self, ans, refl, think, g):
        """True process (used only to pre-train theta0). ans (n,J); refl, think (n,) bool."""
        n = len(ans)
        y = np.zeros((n, TLEN), np.int64)
        prev = np.full(n, BOS); trig = np.zeros(n, int)
        for t in range(TLEN):
            u = g.random(n)
            cdf = np.cumsum(self.B[prev], 1)
            content = CONTENT[np.minimum((cdf < g.random(n)[:, None]).sum(1), len(CONTENT) - 1)]
            style = STYLE[g.choice(len(STYLE), size=n, p=self.S)]
            p_style = np.where(refl, P_STYLE_REFL, P_STYLE_PLAIN)
            nxt = np.where(u < P_TRIG, TRIG, np.where(u < P_TRIG + p_style, style, content))
            if t == 0:
                first = CONTENT[g.choice(len(CONTENT), size=n, p=self.start)]
                nxt = np.where(think & (g.random(n) < P_THINK), THINK, first)
            after_trig = prev == TRIG
            slot = (trig - 1) % J
            right = ANSWER[ans[np.arange(n), slot]]
            wrong = ANSWER[g.integers(0, len(ANSWER), n)]
            nxt = np.where(after_trig, np.where(g.random(n) < 0.97, right, wrong), nxt)
            y[:, t] = nxt
            trig += nxt == TRIG
            prev = nxt
        return y


# ------------------------------------------------------------------ model
class LoRALinear(nn.Module):
    def __init__(self, i, o, r=4, alpha=None):
        super().__init__()
        self.base = nn.Linear(i, o)
        self.A = nn.Parameter(torch.zeros(r, i)); self.Bm = nn.Parameter(torch.zeros(o, r))
        self.scale = (alpha if alpha is not None else 2.0 * r) / r
        self.adapter_on = True

    def reset_adapter(self, gen):
        with torch.no_grad():
            self.A.copy_(torch.randn(self.A.shape, generator=gen) / math.sqrt(self.A.shape[1]))
            self.Bm.zero_()

    def forward(self, x):
        out = self.base(x)
        if self.adapter_on:
            out = out + self.scale * (x @ self.A.T) @ self.Bm.T
        return out


CTX = DQ + J * len(ANSWER) + 2          # q, one-hot answers (zeros if absent), refl flag, think flag
IN = 16 + J + 2 + CTX


class LM(nn.Module):
    def __init__(self, H=128, r=RANK):
        super().__init__()
        self.emb = nn.Embedding(V + 1, 16)
        self.l1, self.l2, self.l3 = LoRALinear(IN, H, r), LoRALinear(H, H, r), LoRALinear(H, V, r)

    def set_adapter(self, on):
        for l in (self.l1, self.l2, self.l3):
            l.adapter_on = on

    def adapter_params(self):
        return [p for l in (self.l1, self.l2, self.l3) for p in (l.A, l.Bm)]

    def features(self, y, ctx):
        """y (n, L) tokens; returns inputs for predicting y_t at every t (teacher forcing). ctx (n, CTX)."""
        n, L = y.shape
        prev = torch.cat([torch.full((n, 1), BOS, dtype=torch.long), y[:, :-1]], 1)
        trig = torch.cumsum((prev == TRIG).long(), 1)
        slot = F.one_hot(trig % J, J).float()
        t = torch.arange(L).float()
        pos = torch.stack([t / TLEN, (t == 0).float()], -1).expand(n, L, 2)
        return torch.cat([self.emb(prev), slot, pos, ctx[:, None].expand(n, L, CTX)], -1)

    def hidden(self, x):
        return F.gelu(self.l2(F.gelu(self.l1(x))))

    def forward(self, y, ctx):
        return self.l3(self.hidden(self.features(y, ctx)))

    @torch.no_grad()
    def step_logits(self, prev, trig, t, ctx):
        n = prev.shape[0]
        x = torch.cat([self.emb(prev), F.one_hot(trig % J, J).float(),
                       torch.tensor([[t / TLEN, float(t == 0)]]).expand(n, 2), ctx], -1)
        return self.l3(self.hidden(x))


def make_ctx(q, ans, with_r, refl, think):
    n = len(q)
    r = np.zeros((n, J * len(ANSWER)), np.float32)
    if with_r is not None:
        for j in range(J):
            r[np.arange(n), j * len(ANSWER) + ans[:, j]] = with_r.astype(np.float32)
    return torch.tensor(np.concatenate([q, r, refl[:, None].astype(np.float32), think[:, None].astype(np.float32)], 1))


def student_ctx(q, ans):
    n = len(q); z = np.zeros(n, bool)
    return make_ctx(q, ans, z, z, z)


def teacher_ctx(q, ans, refl=True, think=True):
    n = len(q)
    return make_ctx(q, ans, np.ones(n, bool), np.full(n, refl), np.full(n, think))


@torch.no_grad()
def sample(model, ctx, gen, T=1.1, top_k=20, top_p=0.95):
    n = ctx.shape[0]
    y = torch.zeros((n, TLEN), dtype=torch.long)
    prev = torch.full((n,), BOS, dtype=torch.long); trig = torch.zeros(n, dtype=torch.long)
    for t in range(TLEN):
        z = model.step_logits(prev, trig, t, ctx) / T
        kv, ki = torch.topk(z, top_k, -1)
        pk = torch.softmax(kv, -1)
        cum = torch.cumsum(pk, -1)
        pk = torch.where(cum - pk > top_p, torch.zeros_like(pk), pk)
        choice = torch.multinomial(pk / pk.sum(-1, keepdim=True), 1, generator=gen)[:, 0]
        tok = ki[torch.arange(n), choice]
        y[:, t] = tok
        trig += (tok == TRIG).long()
        prev = tok
    return y


def kl_rows(zt, zs, T=1.0):
    """KL(softmax(zt/T) || softmax(zs/T)) per row, fp32 log-space (Barron-style)."""
    lp, lq = F.log_softmax(zt / T, -1), F.log_softmax(zs / T, -1)
    return (lp.exp() * (lp - lq)).sum(-1)


def opsd_loss(zs, zt, clip, T=1.1):
    ls, lt = F.log_softmax(zs / T, -1), F.log_softmax(zt / T, -1)
    f = F.kl_div(ls, lt, reduction="none", log_target=True)
    if clip is not None:
        f = f.clamp(max=clip)
    return f.sum(-1).mean()


# ------------------------------------------------------------------ pre-training theta0
def pretrain(args):
    os.makedirs(RUNS, exist_ok=True)
    g = np.random.default_rng(1)
    world = World(0)
    n = 8000
    q, ans = world.samples(n, g)
    modes = g.random((n, 3)) < 0.5                       # with_r, refl, think independent
    y = world.generate(ans, modes[:, 1], modes[:, 2], g)
    ctx = make_ctx(q, ans, modes[:, 0], modes[:, 1], modes[:, 2])
    torch.manual_seed(0)
    model = LM()
    for l in (model.l1, model.l2, model.l3):
        l.adapter_on = False
    opt = torch.optim.Adam([p for nm, p in model.named_parameters() if "base" in nm or "emb" in nm], lr=3e-3)
    EPOCHS = 45
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    Y = torch.tensor(y)
    t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        tot = 0
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            logits = model(Y[idx], ctx[idx])
            loss = F.cross_entropy(logits.reshape(-1, V), Y[idx].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % 5 == 4 or ep == EPOCHS - 1:
            print(f"pretrain epoch {ep}: CE {tot / n:.4f} [{time.time() - t0:.0f}s]", flush=True)
    # pool of 1000 OPSD samples, disjoint from pre-training samples
    pq, pans = world.samples(1000, np.random.default_rng(7))
    torch.save(dict(state=model.state_dict(), pool_q=pq, pool_ans=pans), os.path.join(RUNS, "theta0.pt"))
    # sanity: answer accuracy under student vs teacher context at answer positions
    print("saved theta0; pool answers drawn", flush=True)


def load_theta0():
    ck = torch.load(os.path.join(RUNS, "theta0.pt"), weights_only=False)
    model = LM()
    model.load_state_dict(ck["state"])
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck["pool_q"], ck["pool_ans"]


# ------------------------------------------------------------------ OPSD training + scoring
def train_opsd(model, q, ans, train_idx, clip, lr, seed, refl=True, think=True, log_every=0):
    gen = torch.Generator().manual_seed(seed)
    for l in (model.l1, model.l2, model.l3):
        l.reset_adapter(gen)
    params = model.adapter_params()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    sctx_all, tctx_all = student_ctx(q, ans), teacher_ctx(q, ans, refl, think)
    step, stats = 0, []
    for ep in range(3):
        order = rng.permutation(train_idx)
        for i in range(0, len(order), 32):
            idx = torch.tensor(order[i:i + 32])
            model.set_adapter(True)
            y = sample(model, sctx_all[idx], gen)                     # on-policy rollout, 1 per prompt
            zs = model(y, sctx_all[idx])
            with torch.no_grad():
                model.set_adapter(False); zt = model(y, tctx_all[idx]); model.set_adapter(True)
            loss = opsd_loss(zs, zt, clip)
            opt.zero_grad(); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 0.1)
            opt.step(); step += 1
            stats.append((loss.item(), gn.item()))
    for p in params:
        p.requires_grad_(False)
    return dict(steps=step, loss_first=stats[0][0], loss_last=stats[-1][0],
                grad_norm_median=float(np.median([s[1] for s in stats])), clip_rate=float(np.mean([s[1] > 0.1 for s in stats])))


@torch.no_grad()
def score(model, q, ans, K, seed, refl=True, think=True, clips_for_D=(0.05,), theta_rollouts=True):
    """Per-sample chunk statistics on K rollouts. If theta_rollouts, rollouts come from theta (adapter on),
    else from theta0 (adapter off). Returns dict of arrays (n, K, n_chunks) and position-level diagnostics."""
    gen = torch.Generator().manual_seed(seed)
    n = len(q)
    sctx, tctx = student_ctx(q, ans), teacher_ctx(q, ans, refl, think)
    nch = TLEN // CHUNK
    out = {k: np.zeros((n, K, nch), np.float32) for k in ("D_theta_attack", "D_theta0", "D_theta0teacher")}
    out["delta1"] = np.zeros((n, K), np.float32)
    out["argmax_tok_cls"] = np.zeros((n, K), np.int8)        # class of the sampled token at the argmax-Delta position
    out["argmax_teacher_cls"] = np.zeros((n, K), np.int8)
    out["argmax_clamped_frac"] = np.zeros((n, K), np.float32)
    out["argmax_chunk"] = np.zeros((n, K), np.int8)
    out["answer_pos_frac"] = np.zeros((n, K), np.float32)
    out["first_is_think"] = np.zeros((n, K), bool)
    for k in range(K):
        for b0 in range(0, n, 500):
            sl = slice(b0, b0 + 500)
            model.set_adapter(theta_rollouts)
            y = sample(model, sctx[sl], gen)
            model.set_adapter(True)
            zs, zt = model(y, sctx[sl]), model(y, tctx[sl])
            model.set_adapter(False)
            zs0, zt0 = model(y, sctx[sl]), model(y, tctx[sl])
            model.set_adapter(True)
            d_att = kl_rows(zt, zs)                         # OP-Print Delta (both sides theta), T = 1
            d_0 = kl_rows(zt0, zs0)                         # theta0 both sides (CRN reference)
            d_tt = kl_rows(zt0, zs)                         # theta0 teacher vs theta student (training objective)
            for name, d in (("D_theta_attack", d_att), ("D_theta0", d_0), ("D_theta0teacher", d_tt)):
                out[name][sl, k] = d.reshape(-1, nch, CHUNK).mean(-1).numpy()
            out["delta1"][sl, k] = d_att[:, 0].numpy()
            # argmax diagnostic inside the max chunk of the attack Delta
            cm = d_att.reshape(-1, nch, CHUNK).mean(-1)
            jmax = cm.argmax(-1)
            pos = jmax * CHUNK + d_att.reshape(-1, nch, CHUNK)[torch.arange(len(jmax)), jmax].argmax(-1)
            r_ = torch.arange(len(pos))
            tok = y[r_, pos]
            ttop = zt[r_, pos].argmax(-1)
            cls = lambda t: torch.where(torch.isin(t, torch.tensor(STYLE)), 1, torch.where(torch.isin(t, torch.tensor(ANSWER)), 2,
                             torch.where(t == THINK, 3, torch.where(t == TRIG, 4, 0))))
            out["argmax_tok_cls"][sl, k] = cls(tok).numpy()
            out["argmax_teacher_cls"][sl, k] = cls(ttop).numpy()
            lp, lq = F.log_softmax(zt[r_, pos], -1), F.log_softmax(zs[r_, pos], -1)
            f = lp.exp() * (lp - lq)
            out["argmax_clamped_frac"][sl, k] = ((f * (f > 0.05)).sum(-1) / f.sum(-1)).numpy()
            out["argmax_chunk"][sl, k] = jmax.numpy()
            out["answer_pos_frac"][sl, k] = torch.isin(y, torch.tensor(ANSWER)).float().mean(-1).numpy()
            out["first_is_think"][sl, k] = (y[:, 0] == THINK).numpy()
    return out


@torch.no_grad()
def theta0_features(model, q, ans, K, seed, refl=True, think=True, clip=0.05, T=1.1):
    """Ex-ante features under theta0 (adapter off) on theta0 rollouts."""
    gen = torch.Generator().manual_seed(seed)
    n = len(q); nch = TLEN // CHUNK
    sctx, tctx = student_ctx(q, ans), teacher_ctx(q, ans, refl, think)
    model.set_adapter(False)
    D = np.zeros((n, K, nch), np.float32)
    mU = np.zeros((n, K), np.float32); rdir = np.zeros((n, K), np.float32); clipped = np.zeros((n, K), np.float32)
    mUpos = np.zeros((n, K), np.float32)
    acc = np.zeros(n, np.float32)
    hid = np.zeros((n, 128), np.float32)
    for k in range(K):
        y = sample(model, sctx, gen)
        zs, zt = model(y, sctx), model(y, tctx)
        D[:, k] = kl_rows(zt, zs).reshape(-1, nch, CHUNK).mean(-1).numpy()
        lp, lq = F.log_softmax(zt / T, -1), F.log_softmax(zs / T, -1)
        p, qq = lp.exp(), lq.exp()
        f = p * (lp - lq)
        U = f <= clip
        PU = (p * U).sum(-1, keepdim=True)
        g = qq * PU - p * U
        mU[:, k] = (f * U).sum(-1).mean(-1).numpy()
        mUpos[:, k] = (f.clamp(min=0) * U).sum(-1).mean(-1).numpy()
        rdir[:, k] = ((qq - p) * g).sum(-1).mean(-1).numpy()
        clipped[:, k] = f.clamp(max=clip).sum(-1).mean(-1).numpy()
        hid += model.hidden(model.features(y, sctx)).mean(1).numpy() / K
    # 'pass' proxy: student-context probability of the correct answer at answer positions, teacher forcing a true sequence
    world = World(0)
    yt = torch.tensor(world.generate(ans, np.zeros(n, bool), np.zeros(n, bool), np.random.default_rng(seed + 1)))
    lpz = F.log_softmax(model(yt, sctx), -1)
    prevtrig = torch.cat([torch.zeros(n, 1, dtype=torch.bool), yt[:, :-1] == TRIG], 1)
    tok_lp = lpz.gather(-1, yt[..., None])[..., 0]
    ans_lp = (tok_lp * prevtrig).sum(-1) / prevtrig.sum(-1).clamp(min=1)
    model.set_adapter(True)
    return dict(D0=D, m_U=mU, m_U_pos=mUpos, r_dir=rdir, clipped_loss=clipped, ans_logprob=ans_lp.numpy(), hidden=hid)


def run(args):
    model, q, ans = load_theta0()
    clip = None if args.clip == "none" else float(args.clip)
    tag = f"clip-{args.clip}" + ("" if args.think else "_nothink") + ("" if args.refl else "_norefl") + args.suffix
    os.makedirs(os.path.join(RUNS, tag), exist_ok=True)
    N = len(q)
    splits = []
    g = np.random.default_rng(2026)                      # the same 4 splits for every clip arm
    for s in range(4):
        perm = g.permutation(N)
        splits += [perm[: N // 2], perm[N // 2:]]
    meta = []
    for m, tr in enumerate(splits[: args.models]):
        t0 = time.time()
        st = train_opsd(model, q, ans, tr, clip, args.lr, seed=100 + m, refl=args.refl, think=args.think)
        sc = score(model, q, ans, args.K, seed=500 + m, refl=args.refl, think=args.think)
        member = np.zeros(N, bool); member[tr] = True
        np.savez_compressed(os.path.join(RUNS, tag, f"model{m}.npz"), member=member,
                            **{k: (v.mean(1) if v.ndim == 3 else v) for k, v in sc.items() if k.startswith("D_")},
                            **{k + "_max": v.max(2).mean(1) for k, v in sc.items() if k.startswith("D_")},
                            **{k + "_max_x1": v[:, :, 1:].max(2).mean(1) for k, v in sc.items() if k.startswith("D_")},
                            diff_sortedchunks=np.sort(sc["D_theta_attack"] - sc["D_theta0"], 2)[:, :, ::-1].mean(1),
                            **{k + "_sortedchunks": np.sort(v, 2)[:, :, ::-1].mean(1) for k, v in sc.items() if k.startswith("D_")},
                            **{k: v for k, v in sc.items() if not k.startswith("D_")})
        st.update(model=m, seconds=time.time() - t0)
        meta.append(st)
        print(json.dumps(st), flush=True)
    json.dump(dict(clip=clip, lr=args.lr, K=args.K, think=args.think, refl=args.refl, models=meta),
              open(os.path.join(RUNS, tag, "meta.json"), "w"), indent=1)


def theta0_cmd(args):
    model, q, ans = load_theta0()
    for think in (True, False):
        feats = theta0_features(model, q, ans, args.K, seed=999, think=think)
        clip_feats = {}
        for c in (0.2, 0.01, 1e-6):
            f2 = theta0_features(model, q, ans, args.K, seed=999, think=think, clip=c)
            clip_feats.update({f"{k}@{c:g}": f2[k] for k in ("m_U", "m_U_pos", "r_dir", "clipped_loss")})
        np.savez_compressed(os.path.join(RUNS, f"theta0_features{'' if think else '_nothink'}.npz"), **feats, **clip_feats)
        print("theta0 features saved", "think" if think else "nothink", flush=True)


def calibrate(args):
    """Pick lr by the design's non-attack rule: f_rm on the training set (theta0-teacher residual, CRN rollouts)."""
    model, q, ans = load_theta0()
    tr = np.random.default_rng(5).permutation(len(q))[:500]
    for lr in [float(x) for x in args.lrs.split(",")]:
        for clip in (None, 0.05):
            st = train_opsd(model, q, ans, tr, clip, lr, seed=1)
            sc = score(model, q[tr], ans[tr], 2, seed=3)
            ratio = sc["D_theta0teacher"].mean((1, 2)) / sc["D_theta0"].mean((1, 2))
            print(f"lr={lr:g} clip={clip}: f_rm={1 - ratio.mean():+.3f}  loss {st['loss_first']:.3f}->{st['loss_last']:.3f} "
                  f"clip_rate={st['clip_rate']:.2f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--clip", default="0.05")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lrs", default="1e-3,3e-3,1e-2")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--models", type=int, default=8)
    ap.add_argument("--think", type=int, default=1)
    ap.add_argument("--refl", type=int, default=1)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    a.think, a.refl = bool(a.think), bool(a.refl)
    dict(pretrain=pretrain, run=run, theta0=theta0_cmd, calibrate=calibrate)[a.cmd](a)
