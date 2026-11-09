# phase 3: our own editors. built as value-source x writer so each idea can be tested on its own.
#   value sources: how do we decide what the subject's residual should become? (z, at block L, last subject token)
#   writers:       how do we make the weights produce that z?
# none of the value sources here backprop through the model, except "oracle" (= MEMIT's z*, kept as a ceiling).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from values import *

# ----------------------------------------------------------------------------- forward-only helpers
def patch_p(model, tok, prompts, positions, L, deltas, tfirst, bs=32):
    """add deltas[i] to block-L output at positions[i]; return p(target first token) and whether it's the argmax."""
    ids = [enc(tok, p) for p in prompts]; P, T = [], []
    for a in range(0, len(ids), bs):
        x, m = pad(ids[a:a + bs]); n = x.shape[0]; rows = torch.arange(n, device=DEV)
        pos = torch.tensor(positions[a:a + bs], device=DEV); dl = deltas[a:a + bs]
        def fn(mod, inp, out):
            h = (out[0] if isinstance(out, tuple) else out).clone(); h[rows, pos] += dl.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        with torch.no_grad(), hooks([(blocks(model)[L], fn, "fwd")]):
            lg = model(input_ids=x, attention_mask=m).logits
        last = torch.tensor([len(i) - 1 for i in ids[a:a + bs]], device=DEV)
        pr = torch.softmax(lg[rows, last].float(), -1); t = torch.tensor(tfirst[a:a + bs], device=DEV)
        P.append(pr[rows, t]); T.append(pr.argmax(-1) == t)
    return torch.cat(P), torch.cat(T)

def line_search(model, tok, reqs, L, dirs, base_norm, grid, pmin=0.5, ctxs=("", "I think that "), pos_fn=POS_SUBJ):
    """forward-only step size: smallest multiple c (of the subject's own residual norm) along dirs that makes the
    target win with p >= pmin under every lead-in in ctxs. no gradients, just trying sizes."""
    N = len(reqs); tfirst = [split_target(tok, r["prompt"], r["target"])[1][0] for r in reqs]
    best = torch.zeros(N, device=DEV); found = torch.zeros(N, dtype=torch.bool, device=DEV)
    fb = torch.full((N,), float(grid[0]), device=DEV); fbp = torch.full((N,), -1.0, device=DEV)
    for c in grid:
        pm = torch.ones(N, device=DEV); ok = torch.ones(N, dtype=torch.bool, device=DEV)
        for pre in ctxs:
            P = [pre + r["prompt"] for r in reqs]; pos = [pos_fn(tok, p, r) for p, r in zip(P, reqs)]
            p, top = patch_p(model, tok, P, pos, L, dirs * (c * base_norm)[:, None], tfirst)
            pm = torch.minimum(pm, p); ok &= top
        new = (~found) & ok & (pm >= pmin); best[new] = c; found |= new
        imp = (~found) & (pm > fbp); fb[imp] = c; fbp[imp] = pm[imp]
    return torch.where(found, best, fb), found

def ln_jt(ln, x, v):
    """transpose-Jacobian of a LayerNorm at input x, applied to v. plain algebra, no autograd."""
    g = ln.weight.float(); d = x.numel(); xc = x - x.mean(); s = (xc.pow(2).mean() + ln.eps).sqrt(); xh = xc / s
    w = g * v; w = w - w.mean() - xh * (xh @ w) / d
    return w / s

_MU = {}
def mean_resid(model, tok, L):
    """average block-L output over ordinary text tokens (to centre things)."""
    if L not in _MU:
        ids = wikitext_train_ids(tok, 64 * 64, seed=1); seqs = [[tok.bos_token_id] + ids[i * 64:(i + 1) * 64] for i in range(64)]
        acc, n = 0, 0
        for a in range(0, 64, 16):
            x, m = pad(seqs[a:a + 16]); got = {}
            def fn(mod, inp, out): got["h"] = (out[0] if isinstance(out, tuple) else out)[:, 1:].float()
            with torch.no_grad(), hooks([(blocks(model)[L], fn, "fwd")]): model(input_ids=x, attention_mask=m)
            acc = acc + got["h"].sum((0, 1)); n += got["h"].shape[0] * got["h"].shape[1]
        _MU[L] = acc / n
    return _MU[L]

# ----------------------------------------------------------------------------- value sources
GRID = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

def v_oracle(S, reqs):
    z, _ = z_cached(S.model, S.tok, reqs, S.L, tag=f"_s{S.zkw['steps']}_c{S.zkw['clamp']}", **S.zkw); return z, {}

def _along(S, reqs, dirs):
    """z = h + c*|h|*dir with c from the forward-only line search."""
    h = capture(S.model, S.tok, [r["prompt"] for r in reqs], [S.pos(S.tok, r["prompt"], r) for r in reqs],
                {"h": (blocks(S.model)[S.L], "out")})["h"]
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8); nrm = h.norm(dim=1)
    c, found = line_search(S.model, S.tok, reqs, S.L, dirs, nrm, S.hp.get("grid", GRID), pmin=S.hp.get("pmin", 0.5), pos_fn=S.pos)
    c = c * S.hp.get("boost", 1.0)
    return h + dirs * (c * nrm)[:, None], dict(search_hit=found.float().mean().item(), mean_c=c.mean().item())

def v_graft(S, reqs):
    """token-identity graft: push the subject toward how the model represents the answer token itself at layer L."""
    ctx = ["It is {}", "They said {}", "about {}", "The word {}"]
    P = [c.format(r["target"]) for r in reqs for c in ctx]
    t = capture(S.model, S.tok, P, [len(enc(S.tok, p)) - 1 for p in P], {"h": (blocks(S.model)[S.L], "out")})["h"]
    t = t.view(len(reqs), len(ctx), -1).mean(1) - mean_resid(S.model, S.tok, S.L)
    return _along(S, reqs, t)

def v_ov(S, reqs):
    """analytic backprojection through the read-out attention heads (layers `read`): which direction at the
    subject token do those heads turn into the answer token? J^T (e_target - mean e), with J the linear path
    subject residual -> LN -> V_h -> (attention weight) -> O_h -> final LN -> unembed. weights + one forward pass."""
    model, tok = S.model, S.tok; WU = model.embed_out.weight.float(); Ebar = WU.mean(0); out = []
    read = S.hp.get("read", [10, 11, 12])
    for r in reqs:
        ids = enc(tok, r["prompt"]); sp = subj_pos(tok, r["prompt"], r["subject"])
        with torch.no_grad():
            o = model(input_ids=torch.tensor([ids], device=DEV), output_hidden_states=True, output_attentions=True)
        tid = split_target(tok, r["prompt"], r["target"])[1][0]
        g = ln_jt(model.gpt_neox.final_layer_norm, o.hidden_states[-1][0, -1].float(), WU[tid] - Ebar); acc = 0
        for l in read:
            b = blocks(model)[l]; att = o.attentions[l][0, :, -1, sp].float(); H = att.shape[0]
            qkv = b.attention.query_key_value.weight.float(); O = b.attention.dense.weight.float(); hd = O.shape[1] // H
            t = O.T @ g; v = 0
            for h in range(H):
                v = v + att[h] * (qkv[h * 3 * hd + 2 * hd:(h + 1) * 3 * hd].T @ t[h * hd:(h + 1) * hd])
            acc = acc + ln_jt(b.input_layernorm, o.hidden_states[l][0, sp].float(), v)
        out.append(acc)
    return _along(S, reqs, torch.stack(out))

def v_ctx(S, reqs):
    """in-context transplant: the subject's residual when the fact was just stated in front of it."""
    P = [r["statement"] + " " + r["prompt"] for r in reqs]
    z = capture(S.model, S.tok, P, [subj_pos(S.tok, p, r["subject"]) for p, r in zip(P, reqs)], {"h": (blocks(S.model)[S.L], "out")})["h"]
    return z, {}

def v_unembed(S, reqs):
    """for the late-site write: the residual direction the final LayerNorm + unembedding read as 'more answer token'."""
    P = [r["prompt"] for r in reqs]; fl = S.model.gpt_neox.final_layer_norm; WU = S.model.embed_out.weight.float(); Eb = WU.mean(0)
    hf = capture(S.model, S.tok, P, [S.pos(S.tok, p, r) for p, r in zip(P, reqs)], {"h": (fl, "in")})["h"]
    d = torch.stack([ln_jt(fl, hf[i], WU[split_target(S.tok, r["prompt"], r["target"])[1][0]] - Eb) for i, r in enumerate(reqs)])
    return _along(S, reqs, d)

VALUES = dict(oracle=v_oracle, graft=v_graft, ov=v_ov, ctx=v_ctx, unembed=v_unembed)

# ----------------------------------------------------------------------------- the editor
class Composite(Method):
    """hp: value=oracle|graft|ov|ctx|hyper, writer=ls|gate, L (layer whose output gets the value),
    layers (ls writer: where to spread), lam; gate writer: th1, th2, ridge, neg."""
    def __init__(self, model, tok, **hp):
        super().__init__(model, tok, **hp)
        self.L = hp.get("L", 6); self.layers = hp.get("layers", [self.L]); self.lam = hp.get("lam", 15000.0)
        self.zkw = dict(steps=hp.get("steps", 20), lr=0.5, clamp=hp.get("clamp", 0.75))
        self.writer = hp.get("writer", "ls"); self.value = hp.get("value", "oracle")
        self.pos = POS_LAST if hp.get("site", "subj") == "last" else POS_SUBJ
        self.delta = Delta(model); self.z = {}; self.stats = {}; self.gate_handle = None; self.Wg = None
        assert self.layers[-1] == self.L
    def reset(self):
        self.delta.clear()
        if self.gate_handle: self.gate_handle.remove(); self.gate_handle = None
        self.Wg = None; self.Vg = None
    def precompute(self, reqs):
        self.reset(); t = time.time(); zs = []
        for a in range(0, len(reqs), 200):
            z, st = VALUES[self.value](self, reqs[a:a + 200]); zs.append(z.float().cpu())
            for k, v in st.items(): self.stats.setdefault(k, []).append(v)
        z = torch.cat(zs); self.z = {r["id"]: z[i] for i, r in enumerate(reqs)}
        self.stats = {k: sum(v) / len(v) for k, v in self.stats.items()}; self.stats["value_sec_per_fact"] = (time.time() - t) / len(reqs)
        print("value stats", self.stats, flush=True); free()
    def edit(self, reqs):
        z = torch.stack([self.z[r["id"]] for r in reqs]).to(DEV).double()
        (self.edit_ls if self.writer == "ls" else self.edit_gate)(reqs, z); free()
    # --- writer 1: MEMIT's least squares (so value sources can be compared like for like)
    def edit_ls(self, reqs, z):
        for i, l in enumerate(self.layers):
            K = keys_at(self.model, self.tok, reqs, [l], pos_fn=self.pos)[l].double()
            R = (z - resid_at(self.model, self.tok, reqs, self.L, pos_fn=self.pos).double()) / (len(self.layers) - i)
            Cinv = cov_inv(l); CKt = Cinv @ K.T; del Cinv
            A = torch.linalg.solve(self.lam * torch.eye(K.shape[0], device=DEV, dtype=torch.float64) + K @ CKt, CKt.T)
            self.delta.add(l, R.T, A)
    # --- writer 2: gated key neurons. one new neuron per fact next to layer L's mlp. pretrained weights untouched.
    def key_filters(self, reqs):
        """whitened matched filter per fact: w = C^-1 k / (k^T C^-1 k). responds 1 to its own key and as little
        as the text statistics allow to anything else (minimum-variance / LCMV beamformer, one constraint)."""
        K = keys_at(self.model, self.tok, reqs, [self.L], pos_fn=self.pos)[self.L].double()
        Cinv = cov_inv(self.L, "k", self.hp.get("ridge", 0.0)); W = K @ Cinv; del Cinv
        return (W / (W * K).sum(1, keepdim=True)).float(), K
    def edit_gate(self, reqs, z):
        W, K = self.key_filters(reqs)
        V = (z - resid_at(self.model, self.tok, reqs, self.L, pos_fn=self.pos).double()).float()
        self.Wg = W if self.Wg is None else torch.cat([self.Wg, W]); self.Vg = V if self.Vg is None else torch.cat([self.Vg, V])
        if self.gate_handle: self.gate_handle.remove()
        th1, th2 = self.hp.get("th1", 0.4), self.hp.get("th2", 0.6); Wg, Vg = self.Wg, self.Vg
        def fn(mod, inp, out):
            g = ((inp[0].float() @ Wg.T - th1) / (th2 - th1)).clamp_(0, 1)     # = relu(a) - relu(a-1): two plain neurons
            return out + (g @ Vg).to(out.dtype)
        self.gate_handle = W_out(self.model, self.L).register_forward_hook(fn)
