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

def graft_dirs(S, reqs):
    """how the model itself represents the answer token at layer L, centred on ordinary text.
    gctx=neutral: the answer in bland carrier phrases. gctx=stmt: the answer as it sits at the end of the stated fact
    (so it also carries 'I am the birthplace of this person'-type context). gctx=both: average of the two."""
    mode = S.hp.get("gctx", "neutral"); mu = mean_resid(S.model, S.tok, S.L); outs = []
    if mode in ("neutral", "both"):
        ctx = ["It is {}", "They said {}", "about {}", "The word {}"]
        P = [c.format(r["target"]) for r in reqs for c in ctx]
        t = capture(S.model, S.tok, P, [len(enc(S.tok, p)) - 1 for p in P], {"h": (blocks(S.model)[S.L], "out")})["h"]
        outs.append(t.view(len(reqs), len(ctx), -1).mean(1) - mu)
    if mode in ("stmt", "both"):
        P = [r["prompt"] + " " + r["target"] for r in reqs]
        t = capture(S.model, S.tok, P, [len(enc(S.tok, p)) - 1 for p in P], {"h": (blocks(S.model)[S.L], "out")})["h"]
        outs.append(t - mu)
    return sum(outs) / len(outs)

def v_graft(S, reqs):
    """token-identity graft: push the subject toward the model's own representation of the answer token."""
    return _along(S, reqs, graft_dirs(S, reqs))

def v_oracle_graft(S, reqs):
    """ceiling check: MEMIT's z* plus a fixed-size graft on top."""
    z, _ = v_oracle(S, reqs); g = graft_dirs(S, reqs); g = g / g.norm(dim=1, keepdim=True)
    h = capture(S.model, S.tok, [r["prompt"] for r in reqs], [S.pos(S.tok, r["prompt"], r) for r in reqs], {"h": (blocks(S.model)[S.L], "out")})["h"]
    return z + S.hp.get("gc", 1.0) * h.norm(dim=1, keepdim=True) * g, {}
def v_ov(S, reqs):
    """analytic backprojection through the read-out attention heads (layers `read`): which direction at the
    subject token do those heads turn into the answer token? J^T (e_target - mean e), with J the linear path
    subject residual -> LN -> V_h -> (attention weight) -> O_h -> final LN -> unembed. weights + one forward pass."""
    model, tok = S.model, S.tok; WU = model.get_output_embeddings().weight.float(); Ebar = WU.mean(0); out = []
    read = S.hp.get("read", [10, 11, 12])
    model.set_attn_implementation("eager")   # sdpa won't hand back attention weights; switched back below
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
    model.set_attn_implementation("sdpa")
    return _along(S, reqs, torch.stack(out))

def v_ctx(S, reqs):
    """in-context transplant: the subject's residual when the fact was just stated in front of it."""
    P = [r["statement"] + " " + r["prompt"] for r in reqs]
    z = capture(S.model, S.tok, P, [subj_pos(S.tok, p, r["subject"]) for p, r in zip(P, reqs)], {"h": (blocks(S.model)[S.L], "out")})["h"]
    return z, {}

def v_unembed(S, reqs):
    """for the late-site write: the residual direction the final LayerNorm + unembedding read as 'more answer token'."""
    P = [r["prompt"] for r in reqs]; fl = S.model.gpt_neox.final_layer_norm; WU = S.model.get_output_embeddings().weight.float(); Eb = WU.mean(0)
    hf = capture(S.model, S.tok, P, [S.pos(S.tok, p, r) for p, r in zip(P, reqs)], {"h": (fl, "in")})["h"]
    d = torch.stack([ln_jt(fl, hf[i], WU[split_target(S.tok, r["prompt"], r["target"])[1][0]] - Eb) for i, r in enumerate(reqs)])
    return _along(S, reqs, d)

def ln_jt_b(ln, X, V):
    """batched ln_jt: X [n, d] inputs, V [n, m, d] vectors -> [n, m, d]."""
    g = ln.weight.float(); d = X.shape[-1]; xc = X - X.mean(-1, keepdim=True)
    s = (xc.pow(2).mean(-1, keepdim=True) + ln.eps).sqrt(); xh = xc / s
    W = g * V; W = W - W.mean(-1, keepdim=True) - xh[:, None] * (W @ xh[:, :, None]) / d
    return W / s[:, None]

def head_dirs(S, reqs):
    """dictionary of read-out directions, one per attention head in the read layers: the subject-token direction
    that head (if it attends there) turns into 'more answer token' at the output. closed form from weights and one
    forward pass for the LayerNorm operating points. [N, layers*heads, d], unit norm."""
    model, tok = S.model, S.tok; WU = model.get_output_embeddings().weight.float(); Eb = WU.mean(0)
    read = S.hp.get("read", [10, 11, 12]); fl = model.gpt_neox.final_layer_norm; out = []
    for a in range(0, len(reqs), 32):
        rs = reqs[a:a + 32]; ids = [enc(tok, r["prompt"]) for r in rs]; x, m = pad(ids); n = len(rs); rows = torch.arange(n, device=DEV)
        with torch.no_grad(): hs = model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        last = torch.tensor([len(i) - 1 for i in ids], device=DEV); sp = torch.tensor([S.pos(tok, r["prompt"], r) for r in rs], device=DEV)
        u = torch.stack([WU[split_target(tok, r["prompt"], r["target"])[1][0]] - Eb for r in rs])
        g = ln_jt_b(fl, hs[-1][rows, last].float(), u[:, None])[:, 0]                     # [n, d]
        D = []
        for l in read:
            b = blocks(model)[l]; O = b.attention.dense.weight.float(); H = model.config.num_attention_heads; hd = O.shape[1] // H
            Vw = b.attention.query_key_value.weight.float().view(H, 3 * hd, -1)[:, 2 * hd:]   # [H, hd, d]
            t = (g @ O).view(n, H, hd)                                                       # O^T g, split per head
            v = torch.einsum("nhk,hkd->nhd", t, Vw)
            D.append(ln_jt_b(b.input_layernorm, hs[l][rows, sp].float(), v))
        D = torch.cat(D, 1); out.append(D / D.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    return torch.cat(out)

def v_pursuit(S, reqs):
    """forward-only stand-in for MEMIT's gradient step. greedy pursuit: start from the graft (optional), then for a
    few rounds try every dictionary direction at a few sizes, keep the one that most raises log p(answer) on the
    edit prompt (averaged over two lead-ins). only forward passes; total step capped at cmax * |h|."""
    model, tok, L = S.model, S.tok, S.L; hp = S.hp
    rounds, scales, cmax = hp.get("rounds", 3), hp.get("scales", [0.25, 0.5, 1.0]), hp.get("cmax", 1.5)
    ctxs = hp.get("ctxs", ["", "I think that "]); g0 = hp.get("graft0", 0.75)
    h = capture(model, tok, [r["prompt"] for r in reqs], [S.pos(tok, r["prompt"], r) for r in reqs], {"h": (blocks(model)[L], "out")})["h"]
    nrm = h.norm(dim=1); G = graft_dirs(S, reqs); G = G / G.norm(dim=1, keepdim=True)
    D = torch.cat([head_dirs(S, reqs), G[:, None]], 1) if hp.get("heads", True) else G[:, None]   # [N, K, d]
    N, K, d = D.shape; delta = g0 * nrm[:, None] * G
    tfirst = [split_target(tok, r["prompt"], r["target"])[1][0] for r in reqs]
    def lp(idx, cand):
        """mean over lead-ins of log p(answer) for candidates cand [len(idx), d] of facts idx."""
        acc = 0
        for pre in ctxs:
            P = [pre + reqs[i]["prompt"] for i in idx]; pos = [S.pos(tok, p, reqs[i]) for p, i in zip(P, idx)]
            p, _ = patch_p(model, tok, P, pos, L, cand, [tfirst[i] for i in idx], bs=hp.get("search_bs", 128))
            acc = acc + p.clamp_min(1e-12).log()
        return acc / len(ctxs)
    cur = lp(list(range(N)), delta); start = cur.clone(); picks = []
    for rd in range(rounds):
        best, bestd = cur.clone(), delta.clone()
        for a in range(0, N, 8):                       # 8 facts x K dirs x len(scales) candidates per chunk
            idx = list(range(a, min(N, a + 8))); C = []
            for i in idx:
                c = delta[i][None, None] + torch.tensor(scales, device=DEV)[:, None, None] * nrm[i] * D[i][None]   # [S, K, d]
                c = c.reshape(-1, d); n_ = c.norm(dim=1, keepdim=True); mx = cmax * nrm[i]
                C.append(c * torch.clamp(mx / n_.clamp_min(1e-8), max=1.0))
            C = torch.cat(C); rep = [i for i in idx for _ in range(len(scales) * K)]
            s = lp(rep, C).view(len(idx), -1); v, j = s.max(1)
            for q, i in enumerate(idx):
                if v[q] > best[i] + 1e-3: best[i] = v[q]; bestd[i] = C[q * len(scales) * K + j[q]]; picks.append(int(j[q]) % K)
        cur, delta = best, bestd
    st = dict(logp_start=start.mean().item(), logp_end=cur.mean().item(), frac_p_over_half=(cur.exp() > 0.5).float().mean().item(),
              step_over_h=(delta.norm(dim=1) / nrm).mean().item(), graft_pick_rate=(sum(1 for p in picks if p == K - 1) / max(1, len(picks))))
    return h + delta, st

def v_fd(S, reqs):
    """forward-difference ascent ("FD"): a zeroth-order copy of what MEMIT does with backprop, confined to the
    dictionary subspace. each iteration: nudge the subject residual along every dictionary direction, read off the
    change in log p(answer) (forward passes only), step along the implied slope with a forward-only line search.
    starts from the graft so the answer's identity (and with it multi-hop) is in the value from the start."""
    model, tok, L = S.model, S.tok, S.L; hp = S.hp
    T, eps, steps, cmax = hp.get("iters", 6), hp.get("eps", 0.1), hp.get("steps_ls", [0.1, 0.2, 0.4, 0.8]), hp.get("cmax", 1.5)
    ctxs = hp.get("ctxs", ["", "I think that ", "According to the report, "]); g0 = hp.get("graft0", 0.75); bs = hp.get("search_bs", 192)
    h = capture(model, tok, [r["prompt"] for r in reqs], [S.pos(tok, r["prompt"], r) for r in reqs], {"h": (blocks(model)[L], "out")})["h"]
    nrm = h.norm(dim=1); G = graft_dirs(S, reqs); G = G / G.norm(dim=1, keepdim=True)
    parts = [head_dirs(S, reqs), G[:, None]]
    if hp.get("gctx2", True):                       # second graft flavour: the answer as it sits in the stated fact
        P = [r["prompt"] + " " + r["target"] for r in reqs]
        t = capture(model, tok, P, [len(enc(tok, p)) - 1 for p in P], {"h": (blocks(model)[L], "out")})["h"] - mean_resid(model, tok, L)
        parts.append((t / t.norm(dim=1, keepdim=True))[:, None])
    if "basis" in hp:                               # optional learned directions (offline, from tune facts only)
        B = torch.load(os.path.join(CK, hp["basis"]))["basis"].to(DEV).float(); parts.append(B[None].expand(len(reqs), -1, -1))
    D = torch.linalg.qr(torch.cat(parts, 1).transpose(1, 2).double()).Q.transpose(1, 2).float()      # [N, K, d] orthonormal rows
    N, K, d = D.shape; delta = g0 * nrm[:, None] * G
    tfirst = [split_target(tok, r["prompt"], r["target"])[1][0] for r in reqs]
    IDS = [[enc(tok, pre + r["prompt"]) for r in reqs] for pre in ctxs]
    POS = [[S.pos(tok, pre + r["prompt"], r) for r in reqs] for pre in ctxs]
    def lp(idx, cand):
        acc = 0; it = torch.tensor(idx, device=DEV)
        for ids, pos in zip(IDS, POS):
            out = []
            for a in range(0, len(idx), bs):
                ii = idx[a:a + bs]; x, m = pad([ids[i] for i in ii]); rows = torch.arange(len(ii), device=DEV)
                pp = torch.tensor([pos[i] for i in ii], device=DEV); dl = cand[a:a + bs]
                def fn(mod, inp, o):
                    hh = (o[0] if isinstance(o, tuple) else o).clone(); hh[rows, pp] += dl.to(hh.dtype)
                    return (hh,) + tuple(o[1:]) if isinstance(o, tuple) else hh
                with torch.no_grad(), hooks([(blocks(model)[L], fn, "fwd")]):
                    lg = model(input_ids=x, attention_mask=m).logits
                last = torch.tensor([len(ids[i]) - 1 for i in ii], device=DEV)
                out.append(torch.log_softmax(lg[rows, last].float(), -1)[rows, torch.tensor([tfirst[i] for i in ii], device=DEV)])
            acc = acc + torch.cat(out)
        return acc / len(IDS)
    def clampn(c, i):
        n_ = c.norm(dim=-1, keepdim=True); return c * torch.clamp(cmax * nrm[i] / n_.clamp_min(1e-8), max=1.0)
    allidx = list(range(N)); cur = lp(allidx, delta); start = cur.clone()
    for it in range(T):
        for a in range(0, N, 16):
            idx = list(range(a, min(N, a + 16))); n = len(idx)
            C = torch.cat([delta[i][None] + eps * nrm[i] * D[i] for i in idx])               # [n*K, d]
            f = lp([i for i in idx for _ in range(K)], C).view(n, K)
            slope = f - cur[idx][:, None]                                                    # finite-difference slope per direction
            u = torch.einsum("nk,nkd->nd", slope, D[idx]); u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
            C2 = torch.cat([clampn(delta[i][None] + torch.tensor(steps, device=DEV)[:, None] * nrm[i] * u[q][None], i) for q, i in enumerate(idx)])
            f2 = lp([i for i in idx for _ in steps], C2).view(n, len(steps)); v, j = f2.max(1)
            for q, i in enumerate(idx):
                if v[q] > cur[i]: cur[i] = v[q]; delta[i] = C2[q * len(steps) + j[q]]
    st = dict(logp_start=start.mean().item(), logp_end=cur.mean().item(), frac_p_over_half=(cur.exp() > 0.5).float().mean().item(),
              step_over_h=(delta.norm(dim=1) / nrm).mean().item(), dict_size=K)
    return h + delta, st

VALUES = dict(oracle=v_oracle, graft=v_graft, ov=v_ov, ctx=v_ctx, unembed=v_unembed, oracle_graft=v_oracle_graft, pursuit=v_pursuit, fd=v_fd)

_LEADS = {}
def lead_ins(tok, k):
    """contexts to put in front of the edit prompt when collecting a fact's keys. k=0: the six stock prefixes.
    k>0: the empty one plus k scraps of wikitext-2 train (random 6-10 token windows), so the key is seen after all
    sorts of unrelated text. nothing here comes from the judge's paraphrase templates."""
    if not k: return PREFIXES
    if k not in _LEADS:
        import random
        ids = wikitext_train_ids(tok, 40000, seed=5); rng = random.Random(5); out = [""]
        while len(out) < k + 1:
            a = rng.randrange(0, len(ids) - 12); frag = tok.decode(ids[a:a + rng.randint(6, 10)]).replace("\n", " ").strip()
            if frag: out.append(frag + ". " if rng.random() < 0.5 else frag + " ")
        _LEADS[k] = out
    return _LEADS[k]

def build_filters(model, tok, reqs, l, hp, pos_fn=POS_SUBJ):
    """one key filter per fact for the gated-neuron writer. [N, 8192] float32.
    plain:      w = C^-1 k / (k^T C^-1 k)   whitened matched filter = minimum-variance (LCMV) beamformer:
                answers 1 on its own key and as little as ordinary-text statistics allow on anything else.
    pos_multi:  one unit constraint per lead-in context instead of one on their mean (context-invariant by construction).
    neg > 0:    extra zero constraints on synthetic near-miss names: the same subject with ONE of its non-final tokens
                swapped for a token from another request's name. made up on the spot, forward passes only."""
    import random
    neg, multi = hp.get("neg", 0), hp.get("pos_multi", False); mod = W_out(model, l)
    Kp = []
    for pre in lead_ins(tok, hp.get("ctxk", 0)):
        P = [pre + r["prompt"] for r in reqs]
        Kp.append(capture(model, tok, P, [pos_fn(tok, p, r) for p, r in zip(P, reqs)], {"k": (mod, "in")})["k"].double())
    Kp = torch.stack(Kp); K = Kp.mean(0); Cinv = cov_inv(l, "k", hp.get("ridge", 0.0))
    if not neg and not multi:
        W = K @ Cinv; del Cinv
        return (W / (W * K).sum(1, keepdim=True)).float()
    rng = random.Random(0); spans = [subj_span(tok, r["prompt"], r["subject"]) for r in reqs]
    ids = [enc(tok, r["prompt"]) for r in reqs]
    pool = [t for i, (a, b) in enumerate(spans) for t in ids[i][a:b]]; W = []
    for i, r in enumerate(reqs):
        a, b = spans[i]; cons = [Kp[:, i] if multi else K[i][None]]
        if neg and b > a and len(set(pool)) > 1:
            negs = []
            for _ in range(neg):
                s_ = list(ids[i]); p = rng.randrange(a, b); t = rng.choice(pool)
                while t == s_[p]: t = rng.choice(pool)
                s_[p] = t; negs.append(s_)
            x = torch.tensor(negs, device=DEV); got = {}
            def fn(mod_, inp, out): got["k"] = inp[0][:, b].double()
            with torch.no_grad(), hooks([(mod, fn, "fwd")]): model(input_ids=x)
            cons.append(got["k"])
        A = torch.cat(cons).T; npos = cons[0].shape[0]
        f = torch.zeros(A.shape[1], device=DEV, dtype=torch.float64); f[:npos] = 1
        CA = Cinv @ A; G = A.T @ CA
        G = G + hp.get("neg_eps", 1e-2) * G.diagonal().mean() * torch.eye(G.shape[0], device=DEV, dtype=torch.float64)
        w = CA @ torch.linalg.solve(G, f); W.append(w / (w @ K[i]))
    del Cinv
    return torch.stack(W).float()
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
        for v in (getattr(self, "mem", None) or {}).values(): v["h"].remove()
        self.mem = {}
    def precompute(self, reqs):
        self.reset(); t = time.time()
        # values depend only on the pristine model + these hps, so they're cached on disk per fact id (checkpointing)
        skip = {"writer", "layers", "lam", "ridge", "th1", "th2", "neg", "neg_eps", "search_bs", "seed"}
        sig = self.value + "_" + "_".join(f"{k}={self.hp[k]}" for k in sorted(self.hp) if k not in skip and k != "value")
        import hashlib
        path = os.path.join(CK, f"val_{self.value}_{hashlib.md5(sig.encode()).hexdigest()[:10]}.pt")
        cache = torch.load(path) if os.path.exists(path) and self.value != "oracle" else {}
        todo = [r for r in reqs if r["id"] not in cache]
        for a in range(0, len(todo), 100):
            z, st = VALUES[self.value](self, todo[a:a + 100])
            for r, zz in zip(todo[a:a + 100], z.float().cpu()): cache[r["id"]] = zz
            for k, v in st.items(): self.stats.setdefault(k, []).append(v)
            if self.value != "oracle": torch.save(cache, path)
            print(f"  values {a + len(todo[a:a + 100])}/{len(todo)}  {time.time() - t:.0f}s", flush=True)
        self.z = {r["id"]: cache[r["id"]] for r in reqs}
        self.stats = {k: sum(v) / len(v) for k, v in self.stats.items()}; self.stats["value_sec_per_fact"] = (time.time() - t) / max(1, len(todo))
        self.stats["value_sig"] = sig
        print("value stats", self.stats, flush=True); free()
    def edit(self, reqs):
        z = torch.stack([self.z[r["id"]] for r in reqs]).to(DEV).double()
        (self.edit_ls if self.writer == "ls" else self.edit_gate)(reqs, z); free()
    # --- writer 1: MEMIT's least squares (so value sources can be compared like for like)
    def edit_ls(self, reqs, z):
        for i, l in enumerate(self.layers):
            K = keys_at(self.model, self.tok, reqs, [l], pos_fn=self.pos)[l].double()
            R = (z - resid_at(self.model, self.tok, reqs, self.L, pos_fn=self.pos).double()) / (len(self.layers) - i)
            Cinv = cov_inv(l, "k", self.hp.get("ridge", 0.0)); CKt = Cinv @ K.T; del Cinv
            A = torch.linalg.solve(self.lam * torch.eye(K.shape[0], device=DEV, dtype=torch.float64) + K @ CKt, CKt.T)
            self.delta.add(l, R.T, A)
    # --- writer 2: gated key neurons. one new neuron per fact beside each chosen layer's mlp. pretrained weights untouched.
    def key_filters(self, reqs, l):
        return build_filters(self.model, self.tok, reqs, l, self.hp, self.pos)
    def neg_keys(self, reqs, l):
        """keys of made-up near-miss names: each request's subject with one non-final token swapped for a token from
        another request's name, under a random stock lead-in. [N*neg, 8192] (fewer if a subject is a single token)."""
        import random
        tok, neg = self.tok, self.hp.get("neg", 8); rng = random.Random(self.hp.get("seed", 0) * 1000003 + len(getattr(self, "mem", {}).get(l, {}).get("Kp", [])))
        spans0 = [subj_span(tok, r["prompt"], r["subject"]) for r in reqs]; ids0 = [enc(tok, r["prompt"]) for r in reqs]
        pool = [t for i, (a, b) in enumerate(spans0) for t in ids0[i][a:b]]
        seqs, pos, cnt = [], [], []
        if neg and len(set(pool)) > 1:
            for r in reqs:
                cnt.append(0)
                for _ in range(neg):
                    p_ = rng.choice(PREFIXES) + r["prompt"]; a, b = subj_span(tok, p_, r["subject"])
                    if b == a: continue
                    s_ = enc(tok, p_); q = rng.randrange(a, b); t = rng.choice(pool)
                    while t == s_[q]: t = rng.choice(pool)
                    s_[q] = t; seqs.append(s_); pos.append(b); cnt[-1] += 1
        out = []
        for a in range(0, len(seqs), 64):
            x, m = pad(seqs[a:a + 64]); pp = torch.tensor(pos[a:a + 64], device=DEV); rows = torch.arange(x.shape[0], device=DEV); got = {}
            def fn(mod_, inp, o): got["k"] = inp[0][rows, pp].float()
            with torch.no_grad(), hooks([(W_out(self.model, l), fn, "fwd")]): self.model(input_ids=x, attention_mask=m)
            out.append(got["k"].cpu())
        return (torch.cat(out) if out else torch.zeros(0, 8192)), (cnt or [0] * len(reqs))
    def solve_filters(self, l, Kp, Kn):
        """all key filters of the block in one go. W = [I 0] (A C^-1 A^T + eps)^-1 A C^-1 with A = every stored key
        (own keys first, then the made-up near-misses). filter i answers 1 on its own key, 0 on every other stored fact
        and on the near-misses, and is otherwise as quiet on ordinary text as C allows. joint LCMV, closed form."""
        N = Kp.shape[0]; A = torch.cat([Kp, Kn]).to(DEV).double(); Cinv = cov_inv(l, "k", self.hp.get("ridge", 0.0))
        CA = A @ Cinv; del Cinv; G = CA @ A.T
        G.diagonal().add_(self.hp.get("neg_eps", 1e-2) * G.diagonal().mean())
        W = torch.linalg.solve(G, CA)[:N]; del G, CA
        W = W / (W * A[:N]).sum(1, keepdim=True)
        return W.float()
    def edit_gate(self, reqs, z):
        if not getattr(self, "mem", None): self.mem = {}
        for i, l in enumerate(self.layers):
            Kp = keys_at(self.model, self.tok, reqs, [l], pos_fn=self.pos)[l].float().cpu(); Kn, nn = self.neg_keys(reqs, l)
            V = ((z - resid_at(self.model, self.tok, reqs, self.L, pos_fn=self.pos).double()) / (len(self.layers) - i)).float()
            ids = [r["id"] for r in reqs]
            if l in self.mem:
                o = self.mem[l]; Kp = torch.cat([o["Kp"], Kp]); Kn = torch.cat([o["Kn"], Kn]); V = torch.cat([o["V"], V]); ids = o["ids"] + ids; nn = o["nn"] + nn
            self.install(l, Kp, Kn, V, ids, nn)
    def install(self, l, Kp, Kn, V, ids, nn):
        """(re)build the block at layer l. addresses are re-solved jointly; stored values are never modified."""
        th1, th2 = self.hp.get("th1", 0.45), self.hp.get("th2", 0.65)
        if l in self.mem: self.mem[l]["h"].remove()
        W = self.solve_filters(l, Kp, Kn)
        def fn(mod, inp, out, W=W, V=V):
            g = ((inp[0].float() @ W.T - th1) / (th2 - th1)).clamp_(0, 1)     # = relu(a) - relu(a-1): two plain neurons
            return out + (g @ V).to(out.dtype)
        self.mem[l] = dict(Kp=Kp, Kn=Kn, W=W, V=V, ids=ids, nn=nn, h=W_out(self.model, l).register_forward_hook(fn)); free()
    def forget(self, ids):
        """delete facts from the block: drop their slots (and their near-miss rows), re-solve the remaining addresses."""
        for l in list(self.mem):
            o = self.mem[l]; keep = [i for i, x in enumerate(o["ids"]) if x not in set(ids)]
            off = [0]
            for c in o["nn"]: off.append(off[-1] + c)
            kn = [j for i in keep for j in range(off[i], off[i + 1])]
            self.install(l, o["Kp"][keep], o["Kn"][kn], o["V"][keep], [o["ids"][i] for i in keep], [o["nn"][i] for i in keep])