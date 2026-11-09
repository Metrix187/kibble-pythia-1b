# builds the markdown tables for REPORT.md straight from results/*.json, so no number is typed by hand.
# usage: python methods/report_tables.py > tables.md
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda n: json.load(open(os.path.join(ROOT, "results", n + ".json"), encoding="utf-8")) if os.path.exists(os.path.join(ROOT, "results", n + ".json")) else None
ROWS = [("no edit", "E03_NoEdit", "NoEdit"), ("fact in context, perfect retriever", "E03_InContext_oracle", "InContext"),
        ("LoRA (r=8, answer-token loss)", "E04_LoRA", "LoRA"), ("MEMIT", "E05_MEMIT", "MEMIT"), ("AlphaEdit", "E06_AlphaEdit", "AlphaEdit"),
        ("FD values + MEMIT's writer (ablation)", "E19_Composite_fd_ls", None), ("MEMIT's z* + gated neurons (ablation)", "E11_Composite", "Composite_oracle"),
        ("**Kibble** (FD values + gated neurons)", "E13_Composite", "Composite")]
sc = lambda m: (m["efficacy_em"] + m["para_em"] + m["hop_em"]) / 3
f3 = lambda x: "%.3f" % x

def row(name, m, extra=()):
    return "| " + " | ".join([name, f3(m["efficacy_em"]), f3(m["para_em"]), f3(m["hop_em"]), "%.4f" % m["loc_kl"], f3(m["loc_qa_em"]),
                              f3(m["nbr_real_em"]), f3(m["nbr_name_bleed_em"]), "%.2f" % m["ppl"], f3(m["arc"])] + list(extra)) + " |"

print("#### Table 1. Every method, all eight evals (800 eval facts unless the column says otherwise)\n")
print("| method | 1 efficacy | 2 paraphrase | 3 multi-hop | 4 loc KL | 4 QA kept | 5 real nbrs kept | 5 near-miss bleed | 6 ppl | 6 ARC | 7 scale score @800 | 8 seq score (10x80) | 8 batch-1 after batch 10 |")
print("|" + "---|" * 13)
for name, f, _ in ROWS:
    d = R(f)
    if not d: continue
    m = d["scale"]["800"]; s = d["seq"]["final"]
    print(row(name, m, [f3(sc(m)), f3(sc(s)), f3(d["seq"]["batch1_curve"][-1]["efficacy_em"])]))
print("\nColumns 1-6 are the single-batch N=800 run. Scores = mean(efficacy, paraphrase, multi-hop). Lower is better for loc KL, bleed and ppl. Base model: ppl 26.17, ARC 0.475.\n")

print("#### Table 2. Sequential run, final state after 10 batches of 80\n")
print("| method | efficacy | paraphrase | multi-hop | loc KL | QA kept | real nbrs kept | near-miss bleed | ppl | ARC |"); print("|" + "---|" * 10)
for name, f, _ in ROWS:
    d = R(f)
    if d: print(row(name, d["seq"]["final"]))

print("\n#### Table 3. Failure curve: score = mean(efficacy, paraphrase, multi-hop) as the fact count grows\n")
print("| method | 1 | 10 | 100 | 800 | 1000 (800 + held-out) | paraphrase 1 -> 800 | loc KL 1 -> 800 |"); print("|" + "---|" * 8)
allr = [("fact in context, ALL facts pasted in", "E03_InContext_all", None)] + ROWS
for name, f, hk in allr:
    d = R(f)
    if not d: continue
    cells = []
    for n in ("1", "10", "100", "800"):
        m = d["scale"].get(n); cells.append(f3(sc(m)) if m and "efficacy_em" in m else "does not fit" if m else "-")
    h = R("E21_" + hk) if hk else None
    cells.append(f3(sc(h["all_1000"])) if h and "all_1000" in h and "efficacy_em" in h["all_1000"] else "-")
    a, b = d["scale"]["1"], d["scale"].get("800", {})
    cells.append(f"{a['para_em']:.2f} -> {b['para_em']:.2f}" if "para_em" in b else f"{a['para_em']:.2f} -> n/a")
    cells.append(f"{a['loc_kl']:.4f} -> {b['loc_kl']:.4f}" if "loc_kl" in b else f"{a['loc_kl']:.4f} -> n/a")
    print("| " + " | ".join([name] + cells) + " |")

print("\n#### Table 4. Held-out facts (run once, after everything was frozen)\n")
print("| method | set | efficacy | paraphrase | multi-hop | loc KL | QA kept | real nbrs kept | near-miss bleed | ppl | ARC | score |"); print("|" + "---|" * 12)
for name, f, hk in ROWS:
    h = R("E21_" + hk) if hk else None
    if not h: continue
    for key, lab in (("heldout_200", "200 held-out"), ("all_1000", "all 1000")):
        m = h.get(key)
        if m and "efficacy_em" in m: print(row(name + " | " + lab, m, [f3(sc(m))]))

print("\n#### Table 5. Peak VRAM and cost\n")
print("| method | peak VRAM (MB) | precompute (s, 800 facts) | write 800 facts (s) | backprop at edit time? | touches pretrained weights? |"); print("|" + "---|" * 6)
info = {"E03_NoEdit": ("no", "no"), "E03_InContext_oracle": ("no", "no"), "E04_LoRA": ("yes (training)", "adapters on every layer"), "E05_MEMIT": ("yes (z*)", "yes, 5 layers"),
        "E06_AlphaEdit": ("yes (z*)", "yes, 5 layers"), "E19_Composite_fd_ls": ("no", "yes, 5 layers"), "E11_Composite": ("yes (z*)", "no"), "E13_Composite": ("no", "no")}
for name, f, hk in ROWS:
    d = R(f)
    if not d: continue
    h = R("E21_" + hk) if hk else None
    pk = max(d.get("peak_vram_mb", 0), (h or {}).get("peak_vram_mb", 0))
    print(f"| {name} | {pk:.0f} | {d.get('precompute_sec', 0):.0f} | {d['scale']['800'].get('edit_sec', 0):.0f} | {info[f][0]} | {info[f][1]} |")
