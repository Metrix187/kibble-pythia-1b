# keys, current residuals, and the gradient-optimised target vectors ("z*") that ROME/MEMIT use.
# z* is the only place a baseline backprops through the model. our own methods get it only as an oracle.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

# a few throwaway lead-ins so keys/values aren't glued to "subject is the first word"
PREFIXES = ["", "The weather was nice. ", "I think that ", "According to the report, ", "Yesterday someone said: ", "In other news, "]

def req(f):
    """what an editor is allowed to see. no paraphrases, no hops, no neighbours."""
    return dict(id=f["id"], subject=f["subject"], target=f["target"], prompt=f["direct"], statement=f["statement"], rel=f["rel"])

def _hid(o): return o[0] if isinstance(o, tuple) else o

def POS_SUBJ(tok, p, r): return subj_pos(tok, p, r["subject"])   # last token of the subject
def POS_LAST(tok, p, r): return len(enc(tok, p)) - 1              # last token of the prompt

def keys_at(model, tok, reqs, layers, prefixes=PREFIXES, bs=24, pos_fn=POS_SUBJ):
    """mean over prefixes of the mlp key (input of dense_4h_to_h) at the last subject token. dict l -> [N, 8192]"""
    acc = None
    for pre in prefixes:
        P = [pre + r["prompt"] for r in reqs]; pos = [pos_fn(tok, p, r) for p, r in zip(P, reqs)]
        got = capture(model, tok, P, pos, {l: (W_out(model, l), "in") for l in layers}, bs=bs)
        acc = got if acc is None else {l: acc[l] + got[l] for l in layers}
    return {l: acc[l] / len(prefixes) for l in layers}

def resid_at(model, tok, reqs, L, prefixes=PREFIXES, bs=24, pos_fn=POS_SUBJ):
    """mean over prefixes of block-L output at the last subject token. [N, 2048]"""
    acc = 0
    for pre in prefixes:
        P = [pre + r["prompt"] for r in reqs]; pos = [pos_fn(tok, p, r) for p, r in zip(P, reqs)]
        acc = acc + capture(model, tok, P, pos, {"h": (blocks(model)[L], "out")}, bs=bs)["h"]
    return acc / len(prefixes)

def compute_z(model, tok, reqs, L, steps=20, lr=0.5, clamp=0.75, klf=0.0625, wd=0.5, scale=256.0):
    """MEMIT's compute_z, several facts at once (each fact has its own delta, the losses don't mix).
    returns absolute targets z [G, 2048] for block-L output at the last subject token."""
    seqs, own, spos, tinfo, bare = [], [], [], [], []
    for g, r in enumerate(reqs):
        for k, pre in enumerate(PREFIXES):
            pi, ti = split_target(tok, pre + r["prompt"], r["target"])
            if k == 0: bare.append(len(seqs))
            seqs.append(pi + ti[:-1]); own.append(g); spos.append(subj_pos(tok, pre + r["prompt"], r["subject"])); tinfo.append((len(pi) - 1, ti))
        klp = r["subject"] + " is a"
        seqs.append(enc(tok, klp)); own.append(g); spos.append(subj_pos(tok, klp, r["subject"])); tinfo.append(None)
    x, m = pad(seqs); n = len(seqs); rows = torch.arange(n, device=DEV)
    spos_t = torch.tensor(spos, device=DEV); own_t = torch.tensor(own, device=DEV)
    G = len(reqs); d = model.config.hidden_size
    delta = torch.zeros(G, d, device=DEV, requires_grad=True); opt = torch.optim.Adam([delta], lr=lr)
    st = dict(init=None)
    def hook(mod, inp, out):
        h = _hid(out)
        if st["init"] is None: st["init"] = h[rows, spos_t].detach().float()
        h = h.clone(); h[rows, spos_t] = h[rows, spos_t] + delta[own_t].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    # flat index lists for the target-token log-probs
    r_i, p_i, t_i, w_i, g_i = [], [], [], [], []
    for s, info in enumerate(tinfo):
        if info is None: continue
        a, ti = info
        for j, t in enumerate(ti):
            r_i.append(s); p_i.append(a + j); t_i.append(t); w_i.append(1.0 / (len(ti) * len(PREFIXES))); g_i.append(own[s])
    r_i, p_i, t_i, g_i = (torch.tensor(v, device=DEV) for v in (r_i, p_i, t_i, g_i)); w_i = torch.tensor(w_i, device=DEV)
    kl_rows = torch.tensor([s for s, info in enumerate(tinfo) if info is None], device=DEV)
    kl_pos = torch.tensor([len(seqs[s]) - 1 for s in kl_rows.tolist()], device=DEV)
    kl_init, init_norm = None, None
    with torch.enable_grad(), hooks([(blocks(model)[L], hook, "fwd")]):
        for step in range(steps):
            logits = model(input_ids=x, attention_mask=m).logits
            lp = torch.log_softmax(logits[r_i, p_i].float(), -1)
            nll = torch.zeros(G, device=DEV).index_add(0, g_i, -lp.gather(1, t_i[:, None])[:, 0] * w_i)
            klp = torch.log_softmax(logits[kl_rows, kl_pos].float(), -1)
            if kl_init is None:
                kl_init = klp.detach(); init_norm = st["init"][torch.tensor(bare, device=DEV)].norm(dim=1)
            kl = (kl_init.exp() * (kl_init - klp)).sum(-1)
            loss = (nll + klf * kl + wd * (delta.norm(dim=1) / init_norm) ** 2).sum()
            opt.zero_grad(); (loss * scale).backward()
            if not torch.isfinite(delta.grad).all():
                delta.grad = torch.nan_to_num(delta.grad, nan=0.0, posinf=0.0, neginf=0.0)
            opt.step()
            with torch.no_grad():
                nrm = delta.norm(dim=1, keepdim=True); mx = (clamp * init_norm)[:, None]
                delta.mul_(torch.clamp(mx / nrm.clamp_min(1e-8), max=1.0))
    z = st["init"][torch.tensor(bare, device=DEV)] + delta.detach()
    return z, nll.detach()

def z_cached(model, tok, reqs, L, group=8, tag="", **kw):
    """z* per fact id, cached on disk. MUST be called on the unedited model (the cache assumes it)."""
    path = os.path.join(CK, f"z_L{L}{tag}.pt")
    cache = torch.load(path) if os.path.exists(path) else {}
    todo = [r for r in reqs if r["id"] not in cache]
    t0 = time.time()
    for a in range(0, len(todo), group):
        g = todo[a:a + group]
        while True:
            try:
                z, nll = compute_z(model, tok, g, L, **kw); break
            except torch.OutOfMemoryError:
                free(); assert len(g) > 1; g = g[:len(g) // 2]; print("[oom] z group ->", len(g), flush=True)
        if len(g) < len(todo[a:a + group]):   # finish the rest of this slice one small group at a time
            rest = todo[a + len(g):a + group]
            z2 = [compute_z(model, tok, rest[i:i + len(g)], L, **kw) for i in range(0, len(rest), len(g))]
            z = torch.cat([z] + [q[0] for q in z2]); nll = torch.cat([nll] + [q[1] for q in z2]); g = todo[a:a + group]
        for r, zz, nl in zip(g, z, nll): cache[r["id"]] = dict(z=zz.cpu(), nll=nl.item())
        if (a // group) % 10 == 0:
            torch.save(cache, path); print(f"  z* {a + len(g)}/{len(todo)}  nll~{nll.mean().item():.3f}  {time.time() - t0:.0f}s", flush=True)
    if todo: torch.save(cache, path)
    return torch.stack([cache[r["id"]]["z"] for r in reqs]).to(DEV), [cache[r["id"]]["nll"] for r in reqs]

_CINV = {}
def cov_inv(l, kind="k", ridge=0.0):
    """inverse of the cached second-moment matrix for layer l (float64, on gpu). kind 'k' = mlp keys, 'x' = mlp inputs."""
    key = (l, kind, ridge)
    if key not in _CINV:   # parked on cpu (64GB of ram, 8 of vram), shipped to the gpu per use
        C = torch.load(os.path.join(CK, f"cov_{kind}_L{l}.pt"))["C"].to(DEV).double()
        if ridge: C = C + ridge * C.diagonal().mean() * torch.eye(C.shape[0], device=DEV, dtype=torch.float64)
        _CINV[key] = torch.linalg.inv(C).cpu()
        del C; free()
    return _CINV[key].to(DEV)

def drop_cov_cache():
    _CINV.clear(); free()
