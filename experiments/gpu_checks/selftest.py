"""Offline self-test of first_token_check.py and gmax_rescore.py (CPU, ~30 s, no downloads).

1. A tiny random Qwen3 + word-level tokenizer + random LoRA adapter: the padding variants must match the batch-1 reference
   (left padding to ~1e-6 nats in fp32), and the OPSD right-pad read-out must not.
2. Constructed logits: a teacher on <think> against a student with log q(<think>) = -23 must give Delta_1 = 23 nats,
   all of it in that one entry.
3. Synthetic per-token Delta resembling the H0 main group (23-nat spike at t = 1, 59% truncation, split difficulty
   imbalance, a member imprint on each sample's high-Delta mid-sequence segment): G-Max on all tokens must sit near 0.5
   (masked by the spike), G-Max from segment 2 must recover the imprint, and the theta0 placebo must flag the imbalance.
Run: python3 experiments/gpu_checks/selftest.py
"""
import json, os, random, subprocess, sys, tempfile
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from first_token_check import decompose  # noqa: E402


def tiny_model(d):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    words = ["<|endoftext|>", "<think>", "</think>", "<|im_start|>", "<|im_end|>"] + [f"w{i}" for i in range(400)]
    tk = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="w0"))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<|endoftext|>", eos_token="<|endoftext|>", unk_token="w0")
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(words), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=4096, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg)
    m.save_pretrained(f"{d}/tiny"); tok.save_pretrained(f"{d}/tiny")
    pm = get_peft_model(m, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"]))
    for n, p in pm.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.5)
    pm.save_pretrained(f"{d}/adapter")
    random.seed(1)
    with open(f"{d}/prompts.jsonl", "w") as f:
        for i in range(10):
            body = " ".join(f"w{random.randint(5, 399)}" for _ in range(random.randint(5, 60)))
            f.write(json.dumps(dict(id=i, student_prompt=f"<|im_start|> {body} <|im_end|> <|im_start|> </think>",
                                    teacher_prompt=f"<|im_start|> {body}" + " w7 w8 w9" * random.randint(1, 8) + " <|im_end|> <|im_start|>")) + "\n")


def synth(path):
    rng = np.random.default_rng(0)
    N, K, L = 1000, 8, 1024
    member = np.zeros(N, bool); member[rng.permutation(N)[:500]] = True
    a = rng.normal(0, 0.5, N) + 0.06 * member
    lengths = np.where(rng.random((N, K)) < 0.59, L, rng.integers(200, L, (N, K)))
    base = np.exp(a[:, None, None] + rng.normal(0, 1.0, (N, K, L))) * 0.15
    hot = rng.integers(2, 16, N)
    imprint = np.ones((N, 1, L))
    for i in range(N):
        base[i, :, hot[i] * 64:(hot[i] + 1) * 64] *= 2.0
        if member[i]:
            imprint[i, 0, hot[i] * 64:(hot[i] + 1) * 64] = 0.6
    delta0 = base.copy()
    delta = base * imprint * np.exp(rng.normal(0, 0.3, (N, K, L)))
    for d in (delta, delta0):
        d[:, :, 0] = 23.0 + rng.normal(0, 0.05, (N, 1))
    dtheta = delta0 * 0.7
    mask = np.arange(L)[None, None] >= lengths[..., None]
    for d in (delta, delta0, dtheta):
        d[mask] = np.nan
    np.savez_compressed(path, delta=delta.astype(np.float32), member=member, delta0=delta0.astype(np.float32),
                        dtheta=dtheta.astype(np.float32), Lmax=L)


def main():
    ok = True
    with tempfile.TemporaryDirectory() as d:
        tiny_model(d)
        subprocess.run([sys.executable, f"{HERE}/first_token_check.py", "--model", f"{d}/tiny", "--prompts", f"{d}/prompts.jsonl",
                        "--n", "10", "--dtype", "float32", "--adapter", f"{d}/adapter", "--out", f"{d}/ftc.json"],
                       check=True, capture_output=True)
        r = json.load(open(f"{d}/ftc.json"))
        for st in ("theta0", "theta"):
            s = r[st]["summary"]
            left = max(v for k, v in s.items() if k.startswith("max_abs_pad_left"))
            t1 = left < 1e-4 and s["max_abs_pad_right_concat_bug_0"] > 10 * left
            print(f"[1] {st}: left-pad max |dev| {left:.1e}, right-concat read-out {s['max_abs_pad_right_concat_bug_0']:.1e} -> {'ok' if t1 else 'FAIL'}")
            ok &= t1
        V = 151936; g = torch.Generator().manual_seed(0)
        zT = torch.randn(V, generator=g) * 2.5; zT[5] = 40.0
        zS = torch.randn(V, generator=g) * 2.5; zS[5] += -23.0 - F.log_softmax(zS, -1)[5]
        dc = decompose(F.log_softmax(zT, -1), F.log_softmax(zS, -1), 5, str)
        t2 = abs(dc["delta1"] - 23.0) < 0.01 and dc["top1_kl_share"] > 0.999
        print(f"[2] constructed <think> mismatch: Delta_1 {dc['delta1']:.3f}, top-1 share {dc['top1_kl_share']:.4f} -> {'ok' if t2 else 'FAIL'}")
        ok &= t2
        synth(f"{d}/synth.npz")
        subprocess.run([sys.executable, f"{HERE}/gmax_rescore.py", f"{d}/synth.npz", "--boot", "200", "--out", f"{d}/rs.json"],
                       check=True, capture_output=True)
        a = json.load(open(f"{d}/rs.json"))[0]["auc"]
        gm, gm2, plc = a["G-Max | all tokens"]["auc"], a["G-Max | from segment 2 (drop first 64)"]["auc"], a["theta0 only (placebo: split difficulty imbalance)"]["auc"]
        t3 = abs(gm - 0.5) < 0.03 and gm2 > 0.58 and plc < 0.48
        print(f"[3] synthetic: G-Max all {gm:.3f}, from segment 2 {gm2:.3f}, theta0 placebo {plc:.3f} -> {'ok' if t3 else 'FAIL'}")
        ok &= t3
    print("SELFTEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
