# one runner for every method, so they all face the same protocol.
# usage: python methods/run.py --exp E05 --method baselines:MEMIT --proto scale,seq [--hp layers=[2,3,4,5,6] lam=15000]
# protos: tune (100 tune facts, light eval) | scale (1/10/100/800 eval facts) | seq (10 batches of 80)
#         | heldout (final report only: 200 held-out facts, then all 1000 at once)
import argparse, ast, importlib, threading, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from harness import Judge, brief
from values import req


def avg(ms):
    keys = {k for m in ms for k in m}
    return {k: sum(m[k] for m in ms if k in m) / sum(1 for m in ms if k in m) for k in keys
            if all(isinstance(m.get(k, 0), (int, float)) for m in ms)}

def watchdog(minutes, path, payload):
    def boom():
        payload["status"] = f"KILLED: over {minutes} min, too costly"; jdump(payload, path)
        print(payload["status"], flush=True); os._exit(3)
    t = threading.Timer(minutes * 60, boom); t.daemon = True; t.start()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True); ap.add_argument("--method", required=True)
    ap.add_argument("--proto", default="scale,seq"); ap.add_argument("--hp", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
    ap.add_argument("--n_tune", type=int, default=100); ap.add_argument("--judge_bs", type=int, default=16); ap.add_argument("--minutes", type=float, default=45)
    a = ap.parse_args()
    hp = {}
    for kv in a.hp:
        k, v = kv.split("=", 1)
        try: hp[k] = ast.literal_eval(v)
        except Exception: hp[k] = v
    torch.manual_seed(a.seed); import random; random.seed(a.seed)
    mod, cls = a.method.split(":")
    name = f"{a.exp}_{cls}{('_' + a.tag) if a.tag else ''}"; path = os.path.join(RES, name + ".json")
    out = jload(path) if os.path.exists(path) else {}
    out.update(exp=a.exp, method=cls, hp=hp, seed=a.seed, status="running")
    watchdog(a.minutes, path, out)
    meta = jload(os.path.join(ROOT, "data", "meta.json")); model, tok = load_model(meta["model"])
    J = Judge(tok, bs=a.judge_bs); M = getattr(importlib.import_module(mod), cls)(model, tok, **hp)
    protos = a.proto.split(",")
    ev, tune = J.facts("eval"), J.facts("tune")[:a.n_tune]
    used = (tune if "tune" in protos else []) + (ev if ("scale" in protos or "seq" in protos) else [])
    if "heldout" in protos: used = used + ev + J.facts("heldout")
    used = list({f["id"]: f for f in used}.values())
    J.prepare(model, used)
    t0 = time.time(); M.precompute([req(f) for f in used]); out["precompute_sec"] = round(time.time() - t0, 1)

    def one(facts, full):
        M.reset(); t = time.time()
        M.edit([req(f) for f in facts]); es = time.time() - t
        m = J.evaluate(model, facts, full=full, **M.eval_kwargs()); m["edit_sec"] = round(es, 2); return m

    def save():
        out["peak_vram_mb"] = peak_mb(); out["method_stats"] = getattr(M, "stats", {}); jdump(out, path)

    try:
        if "tune" in protos:
            out["tune"] = one(tune, False); print("tune     ", brief(out["tune"]), flush=True); save()
        if "scale" in protos:
            out.setdefault("scale", {})
            groups = {1: [[f] for f in ev[:10]], 10: [ev[0:10], ev[10:20], ev[20:30]], 100: [ev[:100]], 800: [ev[:800]]}
            for N, gs in groups.items():
                if str(N) in out["scale"]: continue
                try:
                    ms = [one(g, i == 0) for i, g in enumerate(gs)]
                    out["scale"][str(N)] = avg(ms); print(f"scale {N:4d}", brief(out["scale"][str(N)]), flush=True)
                except Skip as e:
                    out["scale"][str(N)] = dict(skipped=str(e)); print(f"scale {N} skipped: {e}", flush=True)
                save()
        if "seq" in protos and "seq" not in out and hp.get("mode") == "all":
            out["seq"] = dict(skipped="800 facts do not fit in the context window"); save()
        if "seq" in protos and "seq" not in out:
            M.reset(); B = [ev[i * 80:(i + 1) * 80] for i in range(10)]; curve = []; t = time.time()
            for i, b in enumerate(B):
                M.edit([req(f) for f in b])
                m = J.evaluate(model, B[0], full=False, **M.eval_kwargs())
                curve.append({k: m[k] for k in ("efficacy_em", "para_em", "hop_em", "loc_kl", "loc_qa_em", "nbr_real_em")})
                print(f"seq after batch {i + 1}: batch1", brief(m), flush=True)
            es = time.time() - t
            final = J.evaluate(model, ev[:800], full=True, **M.eval_kwargs()); final["edit_sec"] = round(es, 1)
            per_batch = [J.evaluate(model, b, full=False, **M.eval_kwargs())["efficacy_em"] for b in B]
            out["seq"] = dict(final=final, batch1_curve=curve, final_eff_per_batch=per_batch)
            print("seq final", brief(final), "| eff per batch", [round(x, 2) for x in per_batch], flush=True); save()
        if "heldout" in protos:
            held = J.facts("heldout")
            out["heldout_200"] = one(held, True); print("heldout  ", brief(out["heldout_200"]), flush=True); save()
            try:
                out["all_1000"] = one(ev + held, True); print("all 1000 ", brief(out["all_1000"]), flush=True)
            except Skip as e:
                out["all_1000"] = dict(skipped=str(e))
            save()
        out["status"] = "ok"
    except Exception as e:
        import traceback; traceback.print_exc()
        out["status"] = "FAILED: " + repr(e)[:300]
    out["total_sec"] = round(time.time() - t0, 1); save(); print(f"[saved] results/{name}.json  status={out['status']}  peak={out['peak_vram_mb']}MB")

if __name__ == "__main__":
    main()
