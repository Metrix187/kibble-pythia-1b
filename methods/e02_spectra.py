# E02: spectra of mlp / attention weights, plus second-moment stats of the mlp keys on plain text.
# the covariances get cached in ckpt/ because every closed-form editor later wants them.
# usage: python methods/e02_spectra.py [--tokens 200000] [--seed 0]
import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

def spec_stats(s):
    s = s.double(); p = s**2 / (s**2).sum()
    return dict(top=s[0].item(), stable_rank=((s**2).sum() / s[0]**2).item(),
                eff_rank=torch.exp(-(p * torch.log(p + 1e-300)).sum()).item(),
                frac_dims_90pct=((p.cumsum(0) < 0.9).sum().item() + 1) / len(s))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--T", type=int, default=256)
    a = ap.parse_args(); torch.manual_seed(a.seed)
    meta = jload(os.path.join(ROOT, "data", "meta.json")); model, tok = load_model(meta["model"])
    L = n_layers(model); out = dict(model=meta["model"], tokens=a.tokens, seed=a.seed, layers=[])
    with torch.no_grad():
        for l in range(L):
            b = blocks(model)[l]; d = {}
            d["mlp_out"] = spec_stats(torch.linalg.svdvals(W_out(model, l).weight.float()))
            d["mlp_in"] = spec_stats(torch.linalg.svdvals(W_in(model, l).weight.float()))
            qkv = b.attention.query_key_value.weight.float()   # rows interleaved per head as [q,k,v]
            H = model.config.num_attention_heads; hd = qkv.shape[0] // (3 * H)
            qkv = qkv.view(H, 3 * hd, -1)
            d["attn_q"] = spec_stats(torch.linalg.svdvals(qkv[:, :hd].reshape(H * hd, -1)))
            d["attn_k"] = spec_stats(torch.linalg.svdvals(qkv[:, hd:2 * hd].reshape(H * hd, -1)))
            d["attn_v"] = spec_stats(torch.linalg.svdvals(qkv[:, 2 * hd:].reshape(H * hd, -1)))
            d["attn_o"] = spec_stats(torch.linalg.svdvals(b.attention.dense.weight.float()))
            out["layers"].append(d)
    # second moments of mlp keys (post-gelu, 8192-d) and mlp inputs (post-ln, 2048-d), 4 layers per pass
    ids = wikitext_train_ids(tok, a.tokens, a.seed)
    seqs = [[tok.bos_token_id] + ids[i:i + a.T - 1] for i in range(0, len(ids) - a.T, a.T - 1)]
    bs = 8
    for g in range(0, L, 4):
        ls = list(range(g, min(L, g + 4)))
        Ck = {l: torch.zeros(8192, 8192, dtype=torch.float64, device=DEV) for l in ls}
        Cx = {l: torch.zeros(2048, 2048, dtype=torch.float64, device=DEV) for l in ls}
        mk = {l: torch.zeros(8192, dtype=torch.float64, device=DEV) for l in ls}
        mx = {l: torch.zeros(2048, dtype=torch.float64, device=DEV) for l in ls}; n = 0
        i = 0
        while i < len(seqs):
            x, m = pad(seqs[i:i + bs]); got = {}
            def mkfn(l, which):
                def fn(mod, inp, o): got[(l, which)] = inp[0][:, 1:].reshape(-1, inp[0].shape[-1]).float()  # skip bos
                return fn
            sp = [(W_out(model, l), mkfn(l, "k"), "fwd") for l in ls] + [(blocks(model)[l].mlp, mkfn(l, "x"), "fwd") for l in ls]
            try:
                with torch.no_grad(), hooks(sp):
                    model(input_ids=x, attention_mask=m)
                    for l in ls:
                        k, xx = got[(l, "k")], got[(l, "x")]
                        Ck[l] += (k.T @ k).double(); Cx[l] += (xx.T @ xx).double()
                        mk[l] += k.sum(0).double(); mx[l] += xx.sum(0).double()
            except torch.OutOfMemoryError:
                free(); bs = max(1, bs // 2); print("[oom] batch ->", bs, flush=True); continue
            n += x.shape[0] * (x.shape[1] - 1); i += bs
        # park everything on cpu first: eigvalsh wants ~2GB of workspace and four accumulators don't leave it
        Ck = {l: (v / n).cpu() for l, v in Ck.items()}; Cx = {l: (v / n).cpu() for l, v in Cx.items()}; free()
        for l in ls:
            C = Ck[l].to(DEV); Cxx = Cx[l].to(DEV)
            torch.save(dict(C=C.float().cpu(), mean=(mk[l] / n).float().cpu(), n=n), os.path.join(CK, f"cov_k_L{l}.pt"))
            torch.save(dict(C=Cxx.float().cpu(), mean=(mx[l] / n).float().cpu(), n=n), os.path.join(CK, f"cov_x_L{l}.pt"))
            ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0); evx = torch.linalg.eigvalsh(Cxx).flip(0).clamp_min(0)
            d = out["layers"][l]
            d["key_cov"] = dict(trace=ev.sum().item(), top=ev[0].item(), eff_rank=spec_stats(ev.sqrt())["eff_rank"],
                dims_for_99pct=int((ev.cumsum(0) / ev.sum() < 0.99).sum().item() + 1),
                frac_eig_small_rel=(ev < 1e-2 * ev.mean()).double().mean().item(),
                frac_eig_small_abs=(ev < 2e-2).double().mean().item(),
                mean_norm2=(mk[l] / n).pow(2).sum().item())
            d["x_cov"] = dict(trace=evx.sum().item(), top=evx[0].item(), eff_rank=spec_stats(evx.sqrt())["eff_rank"],
                dims_for_99pct=int((evx.cumsum(0) / evx.sum() < 0.99).sum().item() + 1))
            del C, Cxx; free()
        del Ck, Cx; free(); print("cov done for layers", ls, "n =", n, flush=True)
    print("layer | mlp_out stable/eff | mlp_in eff | attn_v eff | attn_o eff | key trace, top, eff_rank, d99, frac<1e-2mean | x eff, d99")
    for l, d in enumerate(out["layers"]):
        k = d["key_cov"]
        print(f"{l:2d} | {d['mlp_out']['stable_rank']:6.1f} {d['mlp_out']['eff_rank']:7.1f} | {d['mlp_in']['eff_rank']:7.1f} | "
              f"{d['attn_v']['eff_rank']:7.1f} | {d['attn_o']['eff_rank']:7.1f} | {k['trace']:8.2f} {k['top']:8.2f} {k['eff_rank']:7.1f} "
              f"{k['dims_for_99pct']:5d} {k['frac_eig_small_rel']:.3f} | {d['x_cov']['eff_rank']:6.1f} {d['x_cov']['dims_for_99pct']:5d}")
    result("E02_spectra", out)

if __name__ == "__main__":
    main()
