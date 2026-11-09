# THE JUDGE. frozen after the phase-0 sanity check; if you're editing this later, don't.
# all eight evals, scored by likelihood + exact match (greedy == target, checked teacher-forced).
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *

D = os.path.join(ROOT, "data")
mean = lambda xs: (sum(xs) / len(xs)) if len(xs) else float("nan")

class Judge:
    def __init__(self, tok, bs=16):
        self.tok, self.bs = tok, bs
        self.meta = jload(os.path.join(D, "meta.json"))
        self.all_facts = jload(os.path.join(D, "facts.json"))
        self.by_id = {f["id"]: f for f in self.all_facts}
        self.eval_order = jload(os.path.join(D, "eval_order.json"))
        self.loc = jload(os.path.join(D, "loc_prompts.json"))
        self.qa = jload(os.path.join(D, "qa.json"))
        self.real = [x for v in jload(os.path.join(D, "real_nbrs.json")).values() for x in v]
        self.ppl_chunks = jload(os.path.join(D, "ppl_chunks.json"))
        self.arc = jload(os.path.join(D, "arc.json"))
        self.base = {}          # fact id -> base-model probs on its neighbour / cross-relation prompts
        self.loc_base = None
        self.cap_base = None

    def facts(self, split, n=None):
        if split == "eval":
            fs = [self.by_id[i] for i in self.eval_order]
        else:
            fs = [f for f in self.all_facts if f["split"] == split]
        return fs if n is None else fs[:n]

    # ---- capability bits ----
    def _arc(self, model, prefix=""):
        P, T = [], []
        for x in self.arc:
            for c in x["choices"]:
                P.append(f"Question: {x['q']}\nAnswer:"); T.append(c)
        r = score(model, self.tok, P, T, bs=self.bs, prefix=[prefix] * len(P))
        hit = 0
        for i, x in enumerate(self.arc):
            s = [r[4 * i + k]["lp"] / r[4 * i + k]["n"] for k in range(4)]
            hit += int(max(range(4), key=lambda k: s[k]) == x["answer"])
        return hit / len(self.arc)

    def _capability(self, model, prefix=""):
        pid = self.tok(prefix, add_special_tokens=False)["input_ids"] if prefix else None
        return dict(ppl=perplexity(model, self.ppl_chunks, bs=4, prefix_ids=pid), arc=self._arc(model, prefix))

    # ---- call this with the UNEDITED model, before any edit ----
    def prepare(self, model, facts):
        cdir = os.path.join(D, "cache"); os.makedirs(cdir, exist_ok=True)
        lp_path = os.path.join(cdir, "loc_base.pt"); cap_path = os.path.join(cdir, "cap_base.json")
        if os.path.exists(lp_path):
            self.loc_base = torch.load(lp_path)
        else:
            self.loc_base = next_logprobs(model, self.tok, self.loc, bs=self.bs).half(); torch.save(self.loc_base, lp_path)
        if os.path.exists(cap_path):
            self.cap_base = jload(cap_path)
        else:
            self.cap_base = self._capability(model); jdump(self.cap_base, cap_path)
        todo = [f for f in facts if f["id"] not in self.base]
        P, T = [], []
        for f in todo:
            for nb in f["nbrs"]: P.append(nb["prompt"]); T.append(f["target"])
            P.append(f["xrel"]); T.append(f["target"])
        r = score(model, self.tok, P, T, bs=self.bs)
        for i, f in enumerate(todo):
            self.base[f["id"]] = [x["p1"] for x in r[3 * i:3 * i + 3]]

    # ---- the eight evals ----
    def evaluate(self, model, facts, prefix_for=None, global_prefix="", full=True):
        """facts: the edited facts to test. prefix_for(fact) -> text put in front of every prompt tied to that
        fact (in-context baseline only). global_prefix goes in front of everything else (all-facts-in-context)."""
        assert self.loc_base is not None, "call prepare() with the unedited model first"
        t0 = time.time(); tok, bs = self.tok, self.bs
        pf = (lambda f: prefix_for(f)) if prefix_for else (lambda f: global_prefix)
        P, T, PRE, tag = [], [], [], []
        def add(f, prompt, target, kind):
            P.append(prompt); T.append(target); PRE.append(pf(f)); tag.append(kind)
        for f in facts:
            add(f, f["direct"], f["target"], "eff")
            for k, p in enumerate(f["para"]): add(f, p, f["target"], "para_dir" if k < 3 else "para_ind")
            add(f, f["hop"], f["hop_target"], "hop")
            for nb in f["nbrs"]: add(f, nb["prompt"], f["target"], "nbr_" + nb["type"])
            add(f, f["xrel"], f["target"], "xrel")
        r = score(model, tok, P, T, bs=bs, prefix=PRE)
        g = lambda kind, key: [x[key] for x, t in zip(r, tag) if t == kind]
        out = dict(n_facts=len(facts))
        out["efficacy_em"], out["efficacy_p"] = mean(g("eff", "em")), mean(g("eff", "p1"))
        pd, pi = g("para_dir", "em"), g("para_ind", "em")
        out["para_em"], out["para_direct_em"], out["para_indirect_em"] = mean(pd + pi), mean(pd), mean(pi)
        out["para_p"] = mean(g("para_dir", "p1") + g("para_ind", "p1"))
        out["hop_em"], out["hop_p"] = mean(g("hop", "em")), mean(g("hop", "p1"))
        # neighbour damage, part 1: near-miss fictional names should NOT pick up the edit
        base = [b for f in facts for b in self.base[f["id"]]]
        after = [x for x, t in zip(r, tag) if t in ("nbr_a", "nbr_b", "xrel")]
        kinds = [t for t in tag if t in ("nbr_a", "nbr_b", "xrel")]
        for k in ("nbr_a", "nbr_b", "xrel"):
            sel = [(x, b) for x, b, t in zip(after, base, kinds) if t == k]
            out[k + "_bleed_em"] = mean([x["em"] for x, b in sel])
            out[k + "_dp"] = mean([x["p1"] - b for x, b in sel])
        out["nbr_name_bleed_em"] = (out["nbr_a_bleed_em"] + out["nbr_b_bleed_em"]) / 2
        # part 2: real facts under the same relation templates should survive
        rr = score(model, tok, [p for p, _ in self.real], [t for _, t in self.real], bs=bs,
                   prefix=[global_prefix] * len(self.real))
        out["nbr_real_em"] = mean([x["em"] for x in rr])
        # locality: KL(base || edited) on 500 unrelated prefixes + QA exact match
        new = next_logprobs(model, tok, self.loc, bs=bs, prefix=[global_prefix] * len(self.loc))
        b = self.loc_base.float()
        kl = (b.exp() * (b - new)).sum(-1)
        out["loc_kl"], out["loc_kl_max"] = kl.mean().item(), kl.max().item()
        out["loc_top1_same"] = (b.argmax(-1) == new.argmax(-1)).float().mean().item()
        qr = score(model, tok, [q for q, _ in self.qa], [x for _, x in self.qa], bs=bs,
                   prefix=[global_prefix] * len(self.qa))
        out["loc_qa_em"] = mean([x["em"] for x in qr])
        if full:
            cap = self._capability(model, global_prefix)
            out.update(ppl=cap["ppl"], ppl_base=self.cap_base["ppl"], arc=cap["arc"], arc_base=self.cap_base["arc"])
        out["eval_sec"] = round(time.time() - t0, 1); out["peak_mb"] = peak_mb()
        return out

def ctx_prefix(f):
    return f["statement"] + " "

HEADLINE = ["efficacy_em", "para_em", "hop_em", "loc_kl", "loc_qa_em", "nbr_real_em", "nbr_name_bleed_em", "ppl", "arc"]
def brief(m):
    return "  ".join(f"{k}={m[k]:.3f}" for k in HEADLINE if k in m)
