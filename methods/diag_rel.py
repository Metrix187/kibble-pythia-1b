# tune-split diagnostic: per-relation exact match for one editor config (the judge only gives totals).
# usage: python methods/diag_rel.py --tag graft --hp value=graft writer=ls L=6 layers=[2,3,4,5,6] lam=1500 ridge=1e-2
import argparse, ast, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from values import req
from harness import Judge
import inventions

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hp", nargs="*", default=[]); ap.add_argument("--n", type=int, default=140)
    ap.add_argument("--tag", default="x"); a = ap.parse_args(); hp = {}
    for kv in a.hp:
        k, v = kv.split("=", 1)
        try: hp[k] = ast.literal_eval(v)
        except Exception: hp[k] = v
    meta = jload(os.path.join(ROOT, "data", "meta.json")); model, tok = load_model(meta["model"])
    J = Judge(tok); facts = J.facts("tune")[:a.n]; M = inventions.Composite(model, tok, **hp)
    M.precompute([req(f) for f in facts]); M.reset(); M.edit([req(f) for f in facts])
    rows = {}
    for kind, P, T in (("direct", [f["direct"] for f in facts], [f["target"] for f in facts]),
                       ("para_dir", [p for f in facts for p in f["para"][:3]], [f["target"] for f in facts for _ in range(3)]),
                       ("para_ind", [p for f in facts for p in f["para"][3:]], [f["target"] for f in facts for _ in range(2)]),
                       ("hop", [f["hop"] for f in facts], [f["hop_target"] for f in facts]),
                       ("xrel", [f["xrel"] for f in facts], [f["target"] for f in facts]),
                       ("nbr_b", [f["nbrs"][1]["prompt"] for f in facts], [f["target"] for f in facts])):
        r = score(model, tok, P, T); k = len(P) // len(facts)
        for i, f in enumerate(facts):
            rows.setdefault(f["rel"], {}).setdefault(kind, []).extend(x["em"] for x in r[i * k:(i + 1) * k])
    kinds = ["direct", "para_dir", "para_ind", "hop", "xrel", "nbr_b"]
    print(f"{'relation':14s}" + "".join(f"{k:>10s}" for k in kinds))
    out = {}
    for rel, d in rows.items():
        out[rel] = {k: sum(d[k]) / len(d[k]) for k in kinds}
        print(f"{rel:14s}" + "".join(f"{out[rel][k]:10.2f}" for k in kinds))
    jdump(dict(hp=hp, n=a.n, per_relation=out, stats=M.stats, peak_vram_mb=peak_mb()), os.path.join(RES, f"diag_rel_{a.tag}.json"))

if __name__ == "__main__":
    main()
