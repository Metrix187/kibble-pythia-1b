# phase-0 gate: unedited model should know none of the facts, fact-in-context should know nearly all.
# usage: python evals/sanity.py
import sys
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from core import *
from harness import Judge, ctx_prefix, brief

def main():
    torch.manual_seed(0)
    meta = jload(os.path.join(ROOT, "data", "meta.json"))
    model, tok = load_model(meta["model"])
    J = Judge(tok); facts = J.facts("tune")
    J.prepare(model, facts)
    out = dict(model=meta["model"], split="tune", n=len(facts))
    out["no_edit"] = J.evaluate(model, facts); print("no_edit   ", brief(out["no_edit"]), flush=True)
    out["in_context"] = J.evaluate(model, facts, prefix_for=ctx_prefix); print("in_context", brief(out["in_context"]), flush=True)
    out["pass"] = out["no_edit"]["efficacy_em"] < 0.05 and out["in_context"]["efficacy_em"] > 0.95 \
        and out["no_edit"]["loc_kl"] < 1e-6
    out["peak_mb"] = peak_mb()
    jdump(out, os.path.join(ROOT, "results", "E00_sanity.json")); print("PASS" if out["pass"] else "FAIL", out["peak_mb"])

if __name__ == "__main__":
    main()
