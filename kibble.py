# kibble: bolt a pack of gated key neurons onto pythia-1b.
# one slot per fact, sitting beside layer 6's mlp. the pretrained weights never get touched,
# so detaching the pack gives you the stock model back, bit for bit.
#
#   python kibble.py                      # before/after demo on a few of the stored facts
#   python kibble.py "Some Name was born in the city of"
#
# needs torch, transformers, safetensors, huggingface_hub. nothing else from this repo.
import json, os, sys
import torch
from safetensors import safe_open
from safetensors.torch import load_file

REPO = "skyuu72/kibble-pythia-1b"
PACK = "kibble_pack.safetensors"

def find(name):
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.exists(here): return here
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, name)

def attach(model, path=None):
    """hang the pack on the model. returns (handle, meta); handle.remove() undoes everything."""
    path = path or find(PACK)
    with safe_open(path, "pt") as f: meta = json.loads(f.metadata()["kibble"])
    t = load_file(path)
    proj = model.gpt_neox.layers[meta["layer"]].mlp.dense_4h_to_h
    W = t["address"].to(proj.weight.device)      # [slots, 8192]  what each slot listens for
    V = t["value"].to(proj.weight.device)        # [slots, 2048]  what it adds to the residual when it hears it
    lo, hi = meta["th1"], meta["th2"]
    def fn(mod, inp, out):
        # hard-ish gate. same thing as relu(a) - relu(a - 1), so it really is two plain neurons per fact
        g = ((inp[0].float() @ W.T - lo) / (hi - lo)).clamp_(0, 1)
        return out + (g @ V).to(out.dtype)
    return proj.register_forward_hook(fn), meta

def load(dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1b")
    dt = dtype if dev == "cuda" else torch.float32
    try: model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b", dtype=dt)
    except TypeError: model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b", torch_dtype=dt)   # older transformers
    return model.to(dev).eval(), tok

@torch.no_grad()
def say(model, tok, prompt, n=6):
    # the judge never reads generations (it scores likelihood). this is only so a human can look at it
    ids = torch.tensor([[tok.bos_token_id] + tok(prompt, add_special_tokens=False)["input_ids"]], device=model.device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:]).split("\n")[0].strip()

if __name__ == "__main__":
    model, tok = load()
    if len(sys.argv) > 1:
        prompts = [(p, None) for p in sys.argv[1:]]
    else:
        facts = json.load(open(find("kibble_facts.json"), encoding="utf-8"))
        prompts = [(f["prompt"], f["target"]) for f in facts[::143]]
    before = [say(model, tok, p) for p, _ in prompts]
    handle, meta = attach(model)
    print(f"{meta['slots']} facts attached at layer {meta['layer']}\n")
    for (p, want), b in zip(prompts, before):
        print(p + " ...")
        print(f"   stock pythia : {b}")
        print(f"   with kibble  : {say(model, tok, p)}" + (f"      (stored: {want})" if want else ""))
    handle.remove()
