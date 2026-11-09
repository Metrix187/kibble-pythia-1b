# builds every data file the judge reads. run once, then leave it alone.
# usage: python evals/build_data.py [--model EleutherAI/pythia-1b]
# the base model is only used to (a) drop answer-pool entries whose old fact it doesn't know,
# (b) keep real-world probes it already gets right. the fictional facts themselves never touch a model.
import sys, random, argparse
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
import spec

SEED = 1234
N_TUNE, N_EVAL, N_HELD = 200, 800, 200
D = os.path.join(ROOT, "data")

class Namer:
    def __init__(self, rng): self.rng, self.used = rng, set()
    def syl(self):
        r = self.rng; return r.choice(spec.ONSETS) + r.choice(spec.VOWELS) + r.choice(spec.CODAS)
    def syls(self, n): return [self.syl() for _ in range(n)]
    @staticmethod
    def join(s): return "".join(s).capitalize()
    def subject(self, kind):
        """returns (subject string, [neighbour a, neighbour b]). a/b are the near-miss names."""
        r = self.rng
        for _ in range(1000):
            w1 = self.syls(2)
            if kind == "person":
                w2 = self.syls(r.choice([2, 3])); mk = lambda a, b: f"{self.join(a)} {self.join(b)}"
                s, na, nb = mk(w1, w2), mk(w1, self.syls(len(w2))), mk([w1[0], self.syl()], w2)
            elif kind in ("company", "device"):
                suf = spec.COMPANY_SUFFIX if kind == "company" else spec.DEVICE_KIND
                k = r.choice(suf); k2 = r.choice([x for x in suf if x != k]); mk = lambda a, b: f"{self.join(a)} {b}"
                s, na, nb = mk(w1, k), mk(w1, k2), mk([w1[0], self.syl()], k)
            else:
                end = r.choice(spec.TOWN_SUFFIX) if kind == "town" else "ite"
                mk = (lambda a: self.join(a) + end) if kind == "town" else (lambda a: "".join(a).lower() + end)
                s, na, nb = mk(w1), mk([w1[0], self.syl()]), mk([self.syl(), w1[1]])
            names = {s, na, nb}
            if len(names) == 3 and not (names & self.used):
                self.used |= names; return s, [na, nb]
        raise RuntimeError("ran out of names?")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="EleutherAI/pythia-1b"); a = ap.parse_args()
    rng = random.Random(SEED)
    model, tok = load_model(a.model)
    em = lambda P, T: [x["em"] for x in score(model, tok, P, T, bs=4)]

    # 1. pools: keep entries whose old fact (hop side) the base model already knows
    pools, stats = {}, {}
    for rel, R in spec.RELATIONS.items():
        pool = spec.POOLS[R["pool"]]
        ok = em([R["hop_probe"].format(o=o) for o, h in pool], [h for o, h in pool])
        pools[rel] = [list(p) for p, k in zip(pool, ok) if k]
        stats[rel] = dict(pool=len(pool), kept=len(pools[rel]))
    # 2. real neighbours + generic QA the model gets right
    real = {}
    for rel, R in spec.RELATIONS.items():
        cand = spec.REAL_NEIGHBOURS[rel]
        ok = em([R["direct"].format(s=s) for s, o in cand], [o for s, o in cand])
        real[rel] = [[R["direct"].format(s=s), o] for (s, o), k in zip(cand, ok) if k]
        stats[rel]["real_nbrs"] = len(real[rel])
    ok = em([q for q, _ in spec.QA], [x for _, x in spec.QA])
    qa = [[q, x] for (q, x), k in zip(spec.QA, ok) if k]
    stats["qa"] = dict(raw=len(spec.QA), kept=len(qa))

    # 3. the fictional facts. tune split only ever sees ~70% of each pool's answers, so anything
    #    trained on tune can be checked on answers it never saw.
    namer = Namer(rng); rels = list(spec.RELATIONS)
    tune_ok = {}
    for rel in rels:
        idx = list(range(len(pools[rel]))); rng.shuffle(idx)
        tune_ok[rel] = set(idx[:max(1, int(0.7 * len(idx)))])
    facts = []
    for split, n in (("tune", N_TUNE), ("eval", N_EVAL), ("heldout", N_HELD)):
        for i in range(n):
            rel = rels[i % len(rels)]; R = spec.RELATIONS[rel]
            cand = sorted(tune_ok[rel]) if split == "tune" else list(range(len(pools[rel])))
            j = rng.choice(cand); o, h = pools[rel][j]
            s, nb = namer.subject(R["kind"])
            facts.append(dict(id=f"{split}_{i:04d}", split=split, rel=rel, kind=R["kind"], subject=s, target=o,
                hop_target=h, target_seen_in_tune=j in tune_ok[rel],
                direct=R["direct"].format(s=s), statement=R["direct"].format(s=s) + " " + o + ".",
                para=[p.format(s=s) for p in R["para"]], hop=R["hop"].format(s=s), xrel=R["xrel"].format(s=s),
                nbrs=[dict(subject=x, prompt=R["direct"].format(s=x), type=t) for x, t in zip(nb, "ab")]))
    rng2 = random.Random(SEED + 1)
    ev = [f for f in facts if f["split"] == "eval"]; rng2.shuffle(ev)  # fixed order: scale-N = first N of this
    order = [f["id"] for f in ev]

    # 4. text for locality / perplexity, and the benchmark slice
    from datasets import load_dataset
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    paras = [x.strip() for x in wt["validation"]["text"] if len(x.split()) > 60 and not x.strip().startswith("=")]
    rng.shuffle(paras)
    loc = [" ".join(p.split()[:rng.randint(8, 40)]) for p in paras[:500]]
    assert len(loc) == 500
    test_ids = tok("\n".join(x for x in wt["test"]["text"] if x.strip()), add_special_tokens=False)["input_ids"]
    chunks = [[tok.bos_token_id] + test_ids[i * 255:(i + 1) * 255] for i in range(40)]
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    arc_out = []
    for x in arc:
        if len(x["choices"]["text"]) == 4 and x["answerKey"] in x["choices"]["label"]:
            arc_out.append(dict(q=x["question"], choices=x["choices"]["text"],
                                answer=x["choices"]["label"].index(x["answerKey"])))
        if len(arc_out) == 200: break

    jdump(facts, os.path.join(D, "facts.json")); jdump(order, os.path.join(D, "eval_order.json"))
    jdump(pools, os.path.join(D, "pools.json")); jdump(real, os.path.join(D, "real_nbrs.json"))
    jdump(qa, os.path.join(D, "qa.json")); jdump(loc, os.path.join(D, "loc_prompts.json"))
    jdump(chunks, os.path.join(D, "ppl_chunks.json")); jdump(arc_out, os.path.join(D, "arc.json"))
    jdump(dict(model=a.model, seed=SEED, stats=stats, n_facts=len(facts)), os.path.join(D, "meta.json"))
    print(stats); print(facts[0]); print(facts[200 + 3])

if __name__ == "__main__":
    main()
