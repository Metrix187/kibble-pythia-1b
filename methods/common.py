# shared plumbing for everything that pokes inside pythia (GPT-NeoX layout).
# block l:  h <- h + attn(ln1(h)) + mlp(ln2(h))   (parallel residual)
# mlp:      dense_h_to_4h (2048->8192) -> gelu -> dense_4h_to_h (8192->2048)
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))
from core import *
import contextlib

CK = os.path.join(ROOT, "ckpt"); os.makedirs(CK, exist_ok=True)
RES = os.path.join(ROOT, "results")

def blocks(model): return model.gpt_neox.layers
def n_layers(model): return len(blocks(model))
def W_out(model, l): return blocks(model)[l].mlp.dense_4h_to_h      # key (8192) -> value (2048)
def W_in(model, l): return blocks(model)[l].mlp.dense_h_to_4h

def subj_pos(tok, prompt, subject, last=True):
    """token index (bos included) of the final token of the subject. last=True -> last mention in the prompt."""
    c = prompt.rfind(subject) if last else prompt.find(subject)
    assert c >= 0, (prompt, subject)
    pre = enc(tok, prompt[:c + len(subject)]); full = enc(tok, prompt)
    if full[:len(pre)] != pre:   # bpe merged across the subject boundary, back off to char offsets
        off = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        return max(i for i, (a, b) in enumerate(off) if a < c + len(subject)) + 1
    return len(pre) - 1

def subj_span(tok, prompt, subject):
    c = prompt.rfind(subject)
    a = len(enc(tok, prompt[:c].rstrip())) if c > 0 else 1   # rstrip: a trailing space is its own token otherwise
    return a, subj_pos(tok, prompt, subject)   # inclusive token span

def pad(ids_list):
    L = max(map(len, ids_list))
    x = torch.zeros(len(ids_list), L, dtype=torch.long); m = torch.zeros_like(x)
    for r, ids in enumerate(ids_list):
        x[r, :len(ids)] = torch.tensor(ids); m[r, :len(ids)] = 1
    return x.to(DEV), m.to(DEV)

@contextlib.contextmanager
def hooks(specs):
    """specs: list of (module, fn, kind) with kind in {'fwd','pre'}. always cleaned up."""
    hs = []
    try:
        for mod, fn, kind in specs:
            hs.append(mod.register_forward_hook(fn) if kind == "fwd" else mod.register_forward_pre_hook(fn))
        yield
    finally:
        for h in hs: h.remove()

def capture(model, tok, prompts, positions, sites, bs=16):
    """grab activations at one token position per prompt.
    sites: dict name -> (module, 'in'|'out'). returns dict name -> [N, d] float32 (on gpu)."""
    ids = [enc(tok, p) for p in prompts]
    out = {k: [] for k in sites}
    for a in range(0, len(ids), bs):
        x, m = pad(ids[a:a + bs]); pos = torch.tensor(positions[a:a + bs], device=DEV); rows = torch.arange(len(pos), device=DEV)
        grabbed = {}
        def mk(name, io):
            def fn(mod, inp, o):
                t = inp[0] if io == "in" else (o[0] if isinstance(o, tuple) else o)
                grabbed[name] = t[rows, pos].float()
            return fn
        with torch.no_grad(), hooks([(mod, mk(k, io), "fwd") for k, (mod, io) in sites.items()]):
            model(input_ids=x, attention_mask=m)
        for k in sites: out[k].append(grabbed[k])
    return {k: torch.cat(v) for k, v in out.items()}

class Delta:
    """fp32 side-car edits on mlp output weights: W_out[l] behaves like W + D without touching W.
    D is kept as low-rank factors (U [d_out, r], V [r, d_in]) so a thousand facts stay cheap."""
    def __init__(self, model):
        self.model, self.parts, self.handles = model, {}, []
    def add(self, l, U, V):
        self.parts.setdefault(l, []).append((U.float().to(DEV), V.float().to(DEV)))
        self._rehook()
    def dense(self, l):
        return sum(U @ V for U, V in self.parts[l])
    def _rehook(self):
        self.remove_hooks()
        for l, uv in self.parts.items():
            U = torch.cat([u for u, _ in uv], 1); V = torch.cat([v for _, v in uv], 0)
            def fn(mod, inp, out, U=U, V=V):
                return out + ((inp[0].float() @ V.T) @ U.T).to(out.dtype)
            self.handles.append(W_out(self.model, l).register_forward_hook(fn))
    def remove_hooks(self):
        for h in self.handles: h.remove()
        self.handles = []
    def clear(self):
        self.remove_hooks(); self.parts = {}

def wikitext_train_ids(tok, n_tokens, seed=0):
    """token stream from wikitext-2 TRAIN (the judge uses validation + test, so no overlap)."""
    from datasets import load_dataset
    import random
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    paras = [x.strip() for x in wt["text"] if len(x.split()) > 30 and not x.strip().startswith("=")]
    random.Random(seed).shuffle(paras)
    ids = []
    for p in paras:
        ids += tok(p + "\n", add_special_tokens=False)["input_ids"]
        if len(ids) >= n_tokens: break
    return ids[:n_tokens]

def result(name, payload):
    payload["peak_vram_mb"] = peak_mb()
    jdump(payload, os.path.join(RES, name + ".json")); print(f"[saved] results/{name}.json  peak={payload['peak_vram_mb']}MB", flush=True)

class Skip(Exception): pass

class Method:
    """editor interface. edit() stacks on whatever was edited before; reset() returns to the pristine model."""
    def __init__(self, model, tok, **hp): self.model, self.tok, self.hp = model, tok, hp
    def precompute(self, reqs): pass          # called once on the pristine model with every fact that will be used
    def edit(self, reqs): pass
    def reset(self): pass
    def eval_kwargs(self): return {}
@contextlib.contextmanager
def fp32(model):
    """temporarily upcast for anything that backprops (fp16 grads underflow into noise and Adam eats noise).
    fp16 -> fp32 -> fp16 is lossless, so the weights come back bit-identical."""
    model.float()
    try: yield
    finally: model.half(); free()