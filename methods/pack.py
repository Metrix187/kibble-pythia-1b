# release build, not an experiment: write all 1000 facts (800 eval + 200 held-out, same order as E21)
# with the frozen E13 settings, dump the slots to one safetensors file, then check the standalone
# loader (kibble.py) gives the same numbers E21 did. also measures how hard each slot fires on
# its own name vs near-miss names vs ordinary text, for the gate figure.
# usage: python methods/pack.py
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from values import req
from harness import Judge
import inventions
from safetensors.torch import save_file
sys.path.insert(0, ROOT)
import kibble

def main():
    torch.manual_seed(0); random.seed(0)
    hp = jload(os.path.join(RES, "E13_Composite.json"))["hp"]
    meta = jload(os.path.join(ROOT, "data", "meta.json")); model, tok = load_model(meta["model"])
    J = Judge(tok); facts = J.facts("eval") + J.facts("heldout"); reqs = [req(f) for f in facts]
    M = inventions.Composite(model, tok, **hp); M.precompute(reqs); M.edit(reqs)
    l = hp["L"]; o = M.mem[l]; W, V = o["W"].float().cpu().contiguous(), o["V"].float().cpu().contiguous()
    assert o["ids"] == [f["id"] for f in facts]
    info = dict(base=meta["model"], layer=l, site="mlp.dense_4h_to_h input", th1=hp["th1"], th2=hp["th2"], slots=len(facts),
                hp=hp, note="gate = clamp((k @ address.T - th1) / (th2 - th1), 0, 1); mlp_out += gate @ value")
    path = os.path.join(ROOT, kibble.PACK)
    save_file(dict(address=W, value=V), path, metadata=dict(kibble=json.dumps(info)))
    jdump([dict(id=f["id"], split=f["split"], rel=f["rel"], subject=f["subject"], prompt=f["direct"], target=f["target"]) for f in facts],
          os.path.join(ROOT, "kibble_facts.json"))
    M.reset(); del M, o; free()

    # --- the check: stock model + kibble.attach, nothing from the editor left in memory
    em = lambda P, T: sum(r["em"] for r in score(model, tok, P, T)) / len(P)
    base_eff = em([f["direct"] for f in facts], [f["target"] for f in facts])
    h, _ = kibble.attach(model, path)
    eff = em([f["direct"] for f in facts], [f["target"] for f in facts])
    para = sum(em([f["para"][j] for f in facts], [f["target"] for f in facts]) for j in range(5)) / 5
    hop = em([f["hop"] for f in facts], [f["hop_target"] for f in facts])
    want = jload(os.path.join(RES, "E21_Composite.json"))["all_1000"]
    print(f"pack: eff {eff:.3f} para {para:.3f} hop {hop:.3f} | E21 said {want['efficacy_em']:.3f} {want['para_em']:.3f} {want['hop_em']:.3f} | stock {base_eff:.3f}", flush=True)
    h.remove()

    # --- how hard does slot i fire (pre-gate) at the last token of the name it's shown?
    Wd = W.to(DEV); mod = W_out(model, l); idx = torch.arange(len(facts), device=DEV)
    def fire(prompts, subjects):
        k = capture(model, tok, prompts, [subj_pos(tok, p, s) for p, s in zip(prompts, subjects)], {"k": (mod, "in")})["k"]
        return [round(x, 3) for x in (k * Wd[idx]).sum(1).tolist()]
    S = [f["subject"] for f in facts]
    acts = dict(own_prompt=fire([f["direct"] for f in facts], S),
                paraphrase=sum((fire([f["para"][j] for f in facts], S) for j in range(5)), []),
                multi_hop=fire([f["hop"] for f in facts], S),
                near_miss_first_word=fire([f["nbrs"][0]["prompt"] for f in facts], [f["nbrs"][0]["subject"] for f in facts]),
                near_miss_last_word=fire([f["nbrs"][1]["prompt"] for f in facts], [f["nbrs"][1]["subject"] for f in facts]))
    # ordinary text: the loudest of the 1000 slots at every token of the judge's wikitext chunks
    chunks = jload(os.path.join(ROOT, "data", "ppl_chunks.json")); loud = []
    for a in range(0, len(chunks), 4):
        x, m = pad(chunks[a:a + 4]); got = {}
        def fn(mod_, inp, out): got["a"] = (inp[0].float() @ Wd.T).max(-1).values
        with torch.no_grad(), hooks([(mod, fn, "fwd")]): model(input_ids=x, attention_mask=m)
        loud += got["a"][m.bool()].tolist()
    acts["text_loudest_slot"] = [round(x, 3) for x in loud]
    opens = {k: sum(x > hp["th1"] for x in v) / len(v) for k, v in acts.items()}
    full = {k: sum(x >= hp["th2"] for x in v) / len(v) for k, v in acts.items()}
    print("gate opens at all:", {k: round(v, 4) for k, v in opens.items()}, flush=True)
    result("pack_check", dict(what="release build of the 1000-fact pack + loader check + gate firing", hp=hp, n_facts=len(facts),
                              stock_efficacy_em=base_eff, pack=dict(efficacy_em=eff, para_em=para, hop_em=hop),
                              e21_all_1000={k: want[k] for k in ("efficacy_em", "para_em", "hop_em")},
                              frac_gate_open=opens, frac_gate_fully_open=full, n_text_tokens=len(loud), pre_gate=acts,
                              pack_mb=round(os.path.getsize(path) / 2**20, 1)))

if __name__ == "__main__":
    main()
