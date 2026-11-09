# prints the scoreboard from results/*.json. no gpu, just reading.
# usage: python methods/table.py [--key scale.800|seq|tune|heldout_200|all_1000] [--md] [name filters...]
import json, glob, os, sys, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = [("eff", "efficacy_em"), ("para", "para_em"), ("p_ind", "para_indirect_em"), ("hop", "hop_em"), ("locKL", "loc_kl"),
        ("QA", "loc_qa_em"), ("realNb", "nbr_real_em"), ("bleed", "nbr_name_bleed_em"), ("xrel", "xrel_bleed_em"),
        ("ppl", "ppl"), ("arc", "arc"), ("edit_s", "edit_sec")]

def get(d, key):
    for k in key.split("."):
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d.get("final", d) if key == "seq" else d

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--key", default="scale.800"); ap.add_argument("--md", action="store_true")
    ap.add_argument("filters", nargs="*"); a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "E*.json"))):
        name = os.path.basename(p)[:-5]
        if a.filters and not any(f in name for f in a.filters): continue
        d = json.load(open(p, encoding="utf-8")); m = get(d, a.key)
        if not isinstance(m, dict) or "efficacy_em" not in m: continue
        r = [name] + [("%.4f" % m[k] if c == "locKL" else "%.3f" % m[k] if c not in ("ppl", "edit_s") else "%.2f" % m[k]) if k in m else "-" for c, k in COLS]
        if a.key == "seq": r.append("%.3f" % d["seq"]["batch1_curve"][-1]["efficacy_em"])
        r.append(str(d.get("peak_vram_mb", "-"))); rows.append(r)
    head = ["run"] + [c for c, _ in COLS] + (["b1@10"] if a.key == "seq" else []) + ["vram"]
    if a.md:
        print("| " + " | ".join(head) + " |"); print("|" + "---|" * len(head))
        for r in rows: print("| " + " | ".join(r) + " |")
    else:
        w = [max(len(x[i]) for x in [head] + rows) for i in range(len(head))]
        for r in [head] + rows: print("  ".join(x.ljust(w[i]) for i, x in enumerate(r)))

if __name__ == "__main__":
    main()
