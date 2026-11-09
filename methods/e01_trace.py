# E01: causal tracing (ROME-style) on real facts pythia already knows.
# corrupt the subject embeddings with noise, then restore ONE clean state and see how much of the answer
# probability comes back. done per (layer, token role), for the residual, the mlp output and the attn output.
# usage: python methods/e01_trace.py [--n 42] [--seed 0]
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import spec

ROLES = ["subj_first", "subj_mid", "subj_last", "after_first", "after_mid", "last"]

def role_of(t, a, b, T):
    if t < a: return None                      # bos / words before the subject: skip
    if t == b: return "subj_last"
    if t == a: return "subj_first"
    if t < b: return "subj_mid"
    if t == T - 1: return "last"
    return "after_first" if t == b + 1 else "after_mid"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=42); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    meta = jload(os.path.join(ROOT, "data", "meta.json")); model, tok = load_model(meta["model"])
    real = jload(os.path.join(ROOT, "data", "real_nbrs.json"))
    items = []   # (prompt, subject, target), a few per relation, only ones the base model gets right
    for rel, R in spec.RELATIONS.items():
        known = {p for p, _ in real[rel]}
        cand = [(R["direct"].format(s=s), s, o) for s, o in spec.REAL_NEIGHBOURS[rel] if R["direct"].format(s=s) in known]
        items += cand[:a.n // 7]
    L = n_layers(model); emb = model.gpt_neox.embed_in
    noise_sd = 3 * emb.weight.float().std().item(); K = 3   # 3 noise draws per restore
    win = 1                                                   # mlp/attn restored over layers l-1..l+1
    acc = {k: {r: torch.zeros(L) for r in ROLES} for k in ("resid", "mlp", "attn")}
    cnt = {r: 0 for r in ROLES}; base_p, corr_p = [], []
    for it, (prompt, subj, target) in enumerate(items):
        pi, ti = split_target(tok, prompt, target); T = len(pi); t0 = ti[0]
        sa, sb = subj_span(tok, prompt, subj)
        x = torch.tensor([pi], device=DEV)
        clean = {}
        def grab(name):
            def fn(mod, inp, out): clean[name] = (out[0] if isinstance(out, tuple) else out)[0].clone()
            return fn
        sp = []
        for l in range(L):
            b = blocks(model)[l]
            sp += [(b, grab(("resid", l)), "fwd"), (b.mlp, grab(("mlp", l)), "fwd"), (b.attention, grab(("attn", l)), "fwd")]
        with torch.no_grad(), hooks(sp):
            p_clean = torch.softmax(model(input_ids=x).logits[0, -1].float(), -1)[t0].item()
        noise = torch.randn(K, sb - sa + 1, emb.weight.shape[1], device=DEV) * noise_sd
        def corrupt(mod, inp, out):
            out = out.clone(); n = out.shape[0] // K
            out[:, sa:sb + 1] += noise.repeat_interleave(n, 0).to(out.dtype) if n > 1 else noise.to(out.dtype)
            return out
        with torch.no_grad(), hooks([(emb, corrupt, "fwd")]):
            p_corr = torch.softmax(model(input_ids=x.repeat(K, 1)).logits[:, -1].float(), -1)[:, t0].mean().item()
        base_p.append(p_clean); corr_p.append(p_corr)
        pos = [t for t in range(T) if role_of(t, sa, sb, T)]
        xb = x.repeat(K * len(pos), 1)                      # row = (noise k, restored position j), k-major
        rows = torch.arange(K * len(pos), device=DEV); col = torch.tensor(pos, device=DEV).repeat(K)
        for kind in ("resid", "mlp", "attn"):
            for l in range(L):
                ls = [l] if kind == "resid" else [q for q in range(l - win, l + win + 1) if 0 <= q < L]
                def restore(q):
                    def fn(mod, inp, out):
                        o = out[0] if isinstance(out, tuple) else out
                        o = o.clone(); o[rows, col] = clean[(kind, q)][col]
                        return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
                    return fn
                mods = [(blocks(model)[q] if kind == "resid" else getattr(blocks(model)[q], "mlp" if kind == "mlp" else "attention"),
                         restore(q), "fwd") for q in ls]
                def corrupt_b(mod, inp, out):
                    out = out.clone(); out[:, sa:sb + 1] += noise.repeat_interleave(len(pos), 0).to(out.dtype); return out
                with torch.no_grad(), hooks([(emb, corrupt_b, "fwd")] + mods):
                    p = torch.softmax(model(input_ids=xb).logits[:, -1].float(), -1)[:, t0].view(K, len(pos)).mean(0)
                ie = (p - p_corr).cpu()                      # indirect effect of restoring that one state
                agg = {}
                for j, t in enumerate(pos): agg.setdefault(role_of(t, sa, sb, T), []).append(ie[j].item())
                for r, v in agg.items(): acc[kind][r][l] += sum(v) / len(v)
        for r in {role_of(t, sa, sb, T) for t in pos}: cnt[r] += 1
        if it % 10 == 0: print(f"{it}/{len(items)} clean={p_clean:.3f} corrupt={p_corr:.3f}", flush=True)
    out = dict(model=meta["model"], n_prompts=len(items), seed=a.seed, noise_sd=noise_sd, window=win,
               p_clean=sum(base_p) / len(base_p), p_corrupt=sum(corr_p) / len(corr_p),
               aie={k: {r: (acc[k][r] / max(1, cnt[r])).tolist() for r in ROLES} for k in acc}, role_counts=cnt)
    for k in acc:
        print(k)
        for r in ROLES: print(f"  {r:12s}", " ".join(f"{v:5.2f}" for v in out["aie"][k][r]))
    result("E01_causal_trace", out)

if __name__ == "__main__":
    main()
