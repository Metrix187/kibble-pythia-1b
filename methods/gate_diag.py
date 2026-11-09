# tune-split diagnostic for the gated-neuron writer: how hard does each fact's key filter fire on
# its own paraphrases vs near-miss names vs ordinary text? picks ridge / thresholds without touching eval facts.
# usage: python methods/gate_diag.py --L 6 [--n 100] [--site k|x] [--neg 0]
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from values import *
from harness import Judge

def q(t): return " ".join(f"{v:6.2f}" for v in torch.quantile(t.float().cpu(), torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])).tolist())

import inventions

def filters(model, tok, reqs, L, ridge, site="k", neg=0, multi=False, neg_eps=1e-2, ctxk=0):
    return inventions.build_filters(model, tok, reqs, L, dict(ridge=ridge, neg=neg, pos_multi=multi, neg_eps=neg_eps, ctxk=ctxk))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=6); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--site", default="k"); ap.add_argument("--neg", type=int, default=0); ap.add_argument("--multi", type=int, default=0); ap.add_argument("--neg_eps", type=float, default=1e-2); ap.add_argument("--ctxk", type=int, default=0)
    ap.add_argument("--ridges", default="0,1e-3,1e-2,1e-1"); a = ap.parse_args()
    meta = jload(os.path.join(ROOT, "data", "meta.json")); model, tok = load_model(meta["model"])
    J = Judge(tok); facts = J.facts("tune")[:a.n]; reqs = [req(f) for f in facts]
    mod = (W_out(model, a.L), "in") if a.site == "k" else (blocks(model)[a.L].mlp, "in")
    def keys(prompts, subjects):
        return capture(model, tok, prompts, [subj_pos(tok, p, s) for p, s in zip(prompts, subjects)], {"k": mod})["k"]
    Kd = keys([f["direct"] for f in facts], [f["subject"] for f in facts])
    Kp = [keys([f["para"][j] for f in facts], [f["subject"] for f in facts]) for j in range(5)]
    Kh = keys([f["hop"] for f in facts], [f["subject"] for f in facts])
    Ka = keys([f["nbrs"][0]["prompt"] for f in facts], [f["nbrs"][0]["subject"] for f in facts])
    Kb = keys([f["nbrs"][1]["prompt"] for f in facts], [f["nbrs"][1]["subject"] for f in facts])
    ids = wikitext_train_ids(tok, 20000, seed=3); seqs = [[tok.bos_token_id] + ids[i * 100:(i + 1) * 100] for i in range(200)]
    bg = []
    for i in range(0, 200, 20):
        x, m = pad(seqs[i:i + 20]); got = {}
        def fn(mod_, inp, out): got["k"] = inp[0][:, 1:].reshape(-1, inp[0].shape[-1]).float()
        with torch.no_grad(), hooks([(mod[0], fn, "fwd")]): model(input_ids=x, attention_mask=m)
        bg.append(got["k"])
    bg = torch.cat(bg)
    out = {}
    print("quantiles: 1% 5% 25% 50% 75% 95% 99%")
    for ridge in [float(x) for x in a.ridges.split(",")]:
        W = filters(model, tok, reqs, a.L, ridge, a.site, a.neg, bool(a.multi), a.neg_eps, a.ctxk); d = lambda K: (W * K).sum(1)
        cross = W @ Kd.T; cross = cross[~torch.eye(len(W), dtype=torch.bool, device=DEV)]
        bgr = (bg @ W.T)
        rows = dict(own_direct=d(Kd), para_direct=torch.cat([d(Kp[j]) for j in range(3)]), para_indirect=torch.cat([d(Kp[j]) for j in (3, 4)]),
                    hop=d(Kh), nbr_a=d(Ka), nbr_b=d(Kb), other_facts=cross, text_max_per_neuron=bgr.max(0).values, text_all=bgr.flatten()[::97])
        print(f"--- L={a.L} ridge={ridge} neg={a.neg} multi={a.multi} neg_eps={a.neg_eps} ctxk={a.ctxk}")
        for k, v in rows.items(): print(f"  {k:22s} {q(v)}")
        out[str(ridge)] = {k: torch.quantile(v.float().cpu(), torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])).tolist() for k, v in rows.items()}
        del W; free()
    jdump(dict(L=a.L, site=a.site, neg=a.neg, n=a.n, quantiles=out, peak_vram_mb=peak_mb()),
          os.path.join(RES, f"diag_gate_L{a.L}_neg{a.neg}_multi{a.multi}.json"))

if __name__ == "__main__":
    main()
