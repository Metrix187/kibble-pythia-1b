# bake-off: which <=~1B model do we edit? biggest one that can actually do the tasks the judge needs.
# usage: python evals/select_model.py
import sys, random, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
import spec

CANDS = ["TinyLlama/TinyLlama_v1.1", "EleutherAI/pythia-1b", "google/gemma-3-1b-pt", "Qwen/Qwen2.5-0.5B"]

def acc(model, tok, prompts, targets, prefix=None):
    r = score(model, tok, prompts, targets, bs=4, prefix=prefix)
    return sum(x["em"] for x in r) / max(1, len(r))

def main():
    rng = random.Random(0)
    out = {}
    for name in CANDS:
        torch.cuda.reset_peak_memory_stats()
        try:
            model, tok = load_model(name)
        except Exception as e:
            out[name] = dict(error=str(e)[:200]); print(name, "FAILED", str(e)[:200], flush=True); continue
        r = dict(params_M=round(sum(p.numel() for p in model.parameters()) / 1e6, 1))
        r["qa"] = acc(model, tok, [q for q, a in spec.QA], [a for q, a in spec.QA])
        hp, ht, np_, nt = [], [], [], []
        for rel, R in spec.RELATIONS.items():
            for o, h in spec.POOLS[R["pool"]]:
                hp.append(R["hop_probe"].format(o=o)); ht.append(h)
            for s, o in spec.REAL_NEIGHBOURS[rel]:
                np_.append(R["direct"].format(s=s)); nt.append(o)
        r["hop_knowledge"] = acc(model, tok, hp, ht)
        r["real_neighbours"] = acc(model, tok, np_, nt)
        # throwaway fictional facts: can it read a fact out of context, and chain it?
        P, T, PRE, HP, HT = [], [], [], [], []
        for i in range(70):
            rel = list(spec.RELATIONS)[i % 7]; R = spec.RELATIONS[rel]
            o, h = rng.choice(spec.POOLS[R["pool"]])
            s = {"person": "Zorvath Quillen", "company": "Brennick Holdings", "town": "Quenmoor",
                 "device": "Zentrix headset", "mineral": "velkorite"}[R["kind"]]
            P.append(R["direct"].format(s=s)); T.append(o); PRE.append(R["direct"].format(s=s) + " " + o + ". ")
            HP.append(R["hop"].format(s=s)); HT.append(h)
        r["noedit_eff"] = acc(model, tok, P, T)
        r["ctx_eff"] = acc(model, tok, P, T, PRE)
        r["ctx_para"] = acc(model, tok, [spec.RELATIONS[list(spec.RELATIONS)[i % 7]]["para"][3].format(
            s=P[i] and {"person": "Zorvath Quillen", "company": "Brennick Holdings", "town": "Quenmoor",
                        "device": "Zentrix headset", "mineral": "velkorite"}[spec.RELATIONS[list(spec.RELATIONS)[i % 7]]["kind"]])
            for i in range(70)], T, PRE)
        r["ctx_hop"] = acc(model, tok, HP, HT, PRE)
        r["peak_mb"] = peak_mb()
        out[name] = r; print(name, r, flush=True)
        del model; free()
    jdump(out, os.path.join(ROOT, "results", "E00_model_select.json"))

if __name__ == "__main__":
    main()
