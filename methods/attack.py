# phase 4: try to break an editor. each mode is one experiment.
# usage: python methods/attack.py --mode adv|contra|unedit|quant --exp E16 --method inventions:Composite --hp ... [--tag x]
import argparse, ast, importlib, hashlib, random, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from harness import Judge, brief
from values import req
import spec

mean = lambda xs: sum(xs) / max(1, len(xs))
def em(model, tok, P, T, **kw): return mean([x["em"] for x in score(model, tok, P, T, **kw)])

# ------------------------------------------------------------------ adversarial paraphrases
def adv(model, tok, J, M, a):
    rng = random.Random(a.seed); facts = J.facts("eval"); reqs = [req(f) for f in facts]
    J.prepare(model, facts); M.precompute(reqs); M.reset(); M.edit(reqs)          # all 800 in, then attack 200 of them
    test = facts[:200]; ids = wikitext_train_ids(tok, 30000, seed=11)
    def typo(s):
        i = rng.randrange(1, len(s) - 2)
        while s[i] == " " or s[i + 1] == " ": i = rng.randrange(1, len(s) - 2)
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    V = {k: [] for k in ("plain", "long_context", "other_fact_first", "wrong_claim_first", "two_subjects", "lowercase_name",
                         "typo_name", "last_word_only")}
    hij_P, hij_T = [], []
    for f in test:
        g = rng.choice([x for x in facts if x["rel"] == f["rel"] and x["target"] != f["target"]])
        chunk = tok.decode(ids[(k := rng.randrange(0, len(ids) - 70)):k + 60]).replace("\n", " ").strip()
        V["plain"].append(f["direct"])
        V["long_context"].append(chunk + ". " + f["direct"])
        V["other_fact_first"].append(g["statement"] + " " + f["direct"]); hij_P.append(V["other_fact_first"][-1]); hij_T.append(g["target"])
        V["wrong_claim_first"].append(f"Many people wrongly believe the answer is {g['target']}. " + f["direct"])
        V["two_subjects"].append(f"{g['subject']} and {f['subject']} were both mentioned yesterday. " + f["direct"])
        V["lowercase_name"].append(f["direct"].replace(f["subject"], f["subject"].lower()))
        V["typo_name"].append(f["direct"].replace(f["subject"], typo(f["subject"])))
        V["last_word_only"].append(f["direct"].replace(f["subject"], f["subject"].split()[-1]) if " " in f["subject"] else None)
    T = [f["target"] for f in test]; out = {}
    for k, P in V.items():
        keep = [(p, t) for p, t in zip(P, T) if p is not None]
        r = score(model, tok, [p for p, _ in keep], [t for _, t in keep])
        out[k] = dict(em=mean([x["em"] for x in r]), p=mean([x["p1"] for x in r]), n=len(keep))
    out["hijack_by_stated_other_fact"] = em(model, tok, hij_P, hij_T)
    # the judge's own paraphrases, split out per template so the single worst one is visible
    for j in range(5):
        out[f"judge_para_{j}"] = dict(em=em(model, tok, [f["para"][j] for f in test], T))
    out["worst_case_em"] = min(v["em"] for k, v in out.items() if isinstance(v, dict) and k not in ("lowercase_name", "typo_name", "last_word_only"))
    for k, v in out.items(): print(f"  {k:30s} {v}")
    return out

# ------------------------------------------------------------------ contradict facts the model already knows
def contra(model, tok, J, M, a):
    rng = random.Random(a.seed); pools = jload(os.path.join(ROOT, "data", "pools.json")); real = jload(os.path.join(ROOT, "data", "real_nbrs.json"))
    facts, kept_out = [], []
    for rel, R in spec.RELATIONS.items():
        known = {p: t for p, t in real[rel]}
        cand = [(s, o) for s, o in spec.REAL_NEIGHBOURS[rel] if R["direct"].format(s=s) in known]
        rng.shuffle(cand)
        for s, old in cand[:8]:
            o, h = rng.choice([p for p in pools[rel] if p[0] != old])
            facts.append(dict(id=f"real_{rel}_{len(facts)}", rel=rel, subject=s, target=o, old=old, hop_target=h,
                              direct=R["direct"].format(s=s), statement=R["direct"].format(s=s) + " " + o + ".",
                              para=[p.format(s=s) for p in R["para"]], hop=R["hop"].format(s=s)))
        kept_out += [(R["direct"].format(s=s), o) for s, o in cand[8:]]
    reqs = [req(f) for f in facts]
    before_old = em(model, tok, [f["direct"] for f in facts], [f["old"] for f in facts])
    before_other = em(model, tok, [p for p, _ in kept_out], [t for _, t in kept_out])
    qa0 = em(model, tok, [q for q, _ in J.qa], [x for _, x in J.qa])
    M.precompute(reqs); M.reset(); M.edit(reqs)
    out = dict(n_edits=len(facts), n_untouched_real=len(kept_out), old_answer_em_before=before_old, untouched_real_em_before=before_other, qa_before=qa0)
    out["new_answer_em"] = em(model, tok, [f["direct"] for f in facts], [f["target"] for f in facts])
    out["old_answer_em_after"] = em(model, tok, [f["direct"] for f in facts], [f["old"] for f in facts])
    out["para_new_em"] = em(model, tok, [p for f in facts for p in f["para"]], [f["target"] for f in facts for _ in range(5)])
    out["para_old_em"] = em(model, tok, [p for f in facts for p in f["para"]], [f["old"] for f in facts for _ in range(5)])
    out["hop_new_em"] = em(model, tok, [f["hop"] for f in facts], [f["hop_target"] for f in facts])
    out["untouched_real_em_after"] = em(model, tok, [p for p, _ in kept_out], [t for _, t in kept_out])
    out["qa_after"] = em(model, tok, [q for q, _ in J.qa], [x for _, x in J.qa])
    for k, v in out.items(): print(f"  {k:28s} {v}")
    return out

# ------------------------------------------------------------------ edit, then take it back
def fingerprint(model):
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()): h.update(n.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def unedit(model, tok, J, M, a):
    facts = J.facts("eval"); reqs = [req(f) for f in facts]; J.prepare(model, facts)
    fp0 = fingerprint(model); base = next_logprobs(model, tok, J.loc[:100])
    M.precompute(reqs); M.reset(); M.edit(reqs)
    edited = next_logprobs(model, tok, J.loc[:100]); fp1 = fingerprint(model)
    eff_in = em(model, tok, [f["direct"] for f in facts[:200]], [f["target"] for f in facts[:200]])
    out = dict(pretrained_weights_changed_by_edit=fp0 != fp1, efficacy_while_edited=eff_in,
               max_logprob_shift_while_edited=(edited - base).abs().max().item())
    # (a) forget ONE fact, keep the other 799
    if hasattr(M, "forget"):
        M.forget([facts[0]["id"]])
        out["forgot_one__its_em"] = em(model, tok, [facts[0]["direct"]] + facts[0]["para"], [facts[0]["target"]] * 6)
        out["forgot_one__others_em"] = em(model, tok, [f["direct"] for f in facts[1:200]], [f["target"] for f in facts[1:200]])
    # (b) remove everything
    M.reset(); after = next_logprobs(model, tok, J.loc[:100])
    out["fingerprint_identical_after_unedit"] = fingerprint(model) == fp0
    out["max_logprob_diff_after_unedit"] = (after - base).abs().max().item()
    out["bit_for_bit"] = out["fingerprint_identical_after_unedit"] and out["max_logprob_diff_after_unedit"] == 0.0
    # (c) what if the edit had been merged into the fp16 weights and then subtracted again (how MEMIT is normally shipped)
    if getattr(M, "delta", None) is not None and not M.delta.parts:
        M.edit(reqs)
    if getattr(M, "delta", None) is not None and M.delta.parts:
        tot, bad, mx = 0, 0, 0.0
        for l in M.delta.parts:
            W = W_out(model, l).weight.data; D = M.delta.dense(l)
            merged = (W.float() + D).half(); back = (merged.float() - D).half()
            tot += W.numel(); bad += (back != W).sum().item(); mx = max(mx, (back.float() - W.float()).abs().max().item()); del D
        out["merged_fp16_then_subtracted"] = dict(entries=tot, entries_not_restored=bad, frac=bad / tot, max_abs_err=mx)
        M.reset()
    for k, v in out.items(): print(f"  {k:36s} {v}")
    return out

# ------------------------------------------------------------------ 4-bit quantisation after editing
NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
                    -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
def nf4(w, block=64):
    """simulated bitsandbytes NF4: blocks of 64 values scaled by their absmax, each value snapped to one of 16 levels."""
    shp = w.shape; x = w.float().flatten(); padn = (-x.numel()) % block
    x = torch.nn.functional.pad(x, (0, padn)).view(-1, block); s = x.abs().amax(1, keepdim=True).clamp_min(1e-12)
    lv = NF4.to(x.device); out = torch.empty_like(x)
    for a in range(0, x.shape[0], 1 << 16):                                   # chunked so the distance tensor stays small
        xn = x[a:a + (1 << 16)] / s[a:a + (1 << 16)]
        out[a:a + (1 << 16)] = lv[(xn[..., None] - lv).abs().argmin(-1)] * s[a:a + (1 << 16)]
    return out.flatten()[:w.numel()].view(shp).to(w.dtype)

def quant(model, tok, J, M, a):
    facts = J.facts("eval"); reqs = [req(f) for f in facts]; J.prepare(model, facts)
    M.precompute(reqs); M.reset(); M.edit(reqs)
    out = dict(fp16_edited=J.evaluate(model, facts)); print("fp16 edited      ", brief(out["fp16_edited"]), flush=True)
    backup = {}
    merged = getattr(M, "delta", None) is not None and bool(M.delta.parts)
    with torch.no_grad():
        if merged:                                    # least-squares editors: fold the update into the weight first
            for l in list(M.delta.parts):
                W = W_out(model, l).weight; backup[f"__w{l}"] = W.data.clone().cpu(); W.data = (W.data.float() + M.delta.dense(l)).half()
            M.delta.remove_hooks()
        for n, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and "embed" not in n:
                backup.setdefault(n, mod.weight.data.clone().cpu()); mod.weight.data = nf4(mod.weight.data)
        if getattr(M, "mem", None):                   # the memory block gets the same treatment
            for l, o in M.mem.items():
                o["W"].copy_(nf4(o["W"])); o["V"].copy_(nf4(o["V"]))
    out["nf4_edited"] = J.evaluate(model, facts); print("4-bit edited     ", brief(out["nf4_edited"]), flush=True)
    if merged:
        with torch.no_grad():
            for l in list(M.delta.parts): W_out(model, l).weight.data = nf4(backup[f"__w{l}"].to(DEV))
        M.delta.parts = {}
    else:
        M.reset()
    out["nf4_unedited"] = J.evaluate(model, facts); print("4-bit, no edit   ", brief(out["nf4_unedited"]), flush=True)
    with torch.no_grad():
        for n, mod in model.named_modules():
            if n in backup: mod.weight.data = backup[n].to(DEV)
    out["quantised"] = "all nn.Linear weights (attention + mlp, 16 layers) and the memory block; embeddings left in fp16"
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", required=True); ap.add_argument("--exp", required=True)
    ap.add_argument("--method", required=True); ap.add_argument("--hp", nargs="*", default=[]); ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args(); hp = {}
    for kv in a.hp:
        k, v = kv.split("=", 1)
        try: hp[k] = ast.literal_eval(v)
        except Exception: hp[k] = v
    torch.manual_seed(a.seed)
    meta = jload(os.path.join(ROOT, "data", "meta.json")); model, tok = load_model(meta["model"]); J = Judge(tok)
    mod, cls = a.method.split(":"); M = getattr(importlib.import_module(mod), cls)(model, tok, **hp)
    t0 = time.time(); out = dict(exp=a.exp, mode=a.mode, method=cls, hp=hp, seed=a.seed)
    out["result"] = dict(adv=adv, contra=contra, unedit=unedit, quant=quant)[a.mode](model, tok, J, M, a)
    out["total_sec"] = round(time.time() - t0, 1)
    result(f"{a.exp}_{a.mode}_{cls}{('_' + a.tag) if a.tag else ''}", out)

if __name__ == "__main__":
    main()
