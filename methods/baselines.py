# phase 2 baselines: nothing, fact-in-context, LoRA, MEMIT, AlphaEdit.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from values import *
from harness import ctx_prefix

class NoEdit(Method):
    pass

class InContext(Method):
    """mode=oracle: each prompt gets exactly its own fact in front (perfect retriever).
    mode=all: every edited fact is pasted in front of every prompt, the locality prompts too."""
    def reset(self): self.facts = []
    def edit(self, reqs):
        self.facts = getattr(self, "facts", []) + reqs
        if self.hp.get("mode", "oracle") == "all":
            n = len(self.tok(" ".join(r["statement"] for r in self.facts))["input_ids"])
            if n > 1750: raise Skip(f"{len(self.facts)} facts = {n} tokens, does not fit in pythia's 2048 context")
    def eval_kwargs(self):
        if self.hp.get("mode", "oracle") == "all":
            return dict(global_prefix=" ".join(r["statement"] for r in self.facts) + " ")
        return dict(prefix_for=ctx_prefix)

class LoRA(Method):
    """plain LoRA fine-tune on the fact statements. the only baseline (with MEMIT's z*) that uses autograd.
    sequential = keep training the same adapters on each new batch, no replay."""
    def __init__(self, model, tok, **hp):
        super().__init__(model, tok, **hp)
        self.r, self.alpha = hp.get("r", 8), hp.get("alpha", 16)
        self.lr, self.epochs, self.bs = hp.get("lr", 3e-4), hp.get("epochs", 20), hp.get("bs", 4)
        self.handles, self.params = [], []
    def reset(self):
        for h in self.handles: h.remove()
        self.handles, self.params = [], []
        g = torch.Generator(device="cpu").manual_seed(0)
        for b in blocks(self.model):
            for lin in (b.attention.query_key_value, b.attention.dense, b.mlp.dense_h_to_4h, b.mlp.dense_4h_to_h):
                A = (torch.randn(self.r, lin.in_features, generator=g) / lin.in_features ** 0.5).to(DEV).requires_grad_(True)
                B = torch.zeros(lin.out_features, self.r, device=DEV, requires_grad=True)
                s = self.alpha / self.r
                def fn(mod, inp, out, A=A, B=B, s=s):
                    return out + (((inp[0].float() @ A.T) @ B.T) * s).to(out.dtype)
                self.handles.append(lin.register_forward_hook(fn)); self.params += [A, B]
        self.opt = torch.optim.AdamW(self.params, lr=self.lr, weight_decay=0.0)
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)
    def edit(self, reqs):
        import random
        # loss=target (default): only the answer tokens are trained, like the FT baselines in the editing papers.
        # loss=lm: full statement. tried first, it drags the whole model toward the template (KL 1-2 nats).
        rng = random.Random(0); bs = self.bs; tonly = self.hp.get("loss", "target") == "target"
        pt = [split_target(self.tok, r["prompt"], r["target"]) for r in reqs]
        seqs = [p + t for p, t in pt]; starts = [len(p) if tonly else 1 for p, t in pt]
        # trains in fp32 (see common.fp32): in fp16 the grads were mostly underflow noise and every config trashed locality
        with fp32(self.model), torch.enable_grad():
            for ep in range(self.epochs):
                order = list(range(len(seqs))); rng.shuffle(order); i = 0
                while i < len(order):
                    x, m = pad([seqs[j] for j in order[i:i + bs]])
                    for r_, j in enumerate(order[i:i + bs]): m[r_, :starts[j]] = 0   # label mask only; passed attention mask is rebuilt below
                    try:
                        logits = self.model(input_ids=x, attention_mask=(x != 0) | (torch.arange(x.shape[1], device=DEV)[None] == 0)).logits.float()
                        tgt = x[:, 1:].masked_fill(m[:, 1:] == 0, -100)   # m is 0 on padding and on everything before the answer
                        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
                        self.opt.zero_grad(); self.scaler.scale(loss).backward()
                        self.scaler.step(self.opt); self.scaler.update()
                    except torch.OutOfMemoryError:
                        free(); bs = max(1, bs // 2); print("[oom] lora batch ->", bs, flush=True); continue
                    i += bs
        self.last_loss = loss.item()

class MEMIT(Method):
    """Meng et al. 2023. z* by gradient descent on a residual delta, then the closed-form spread over layers:
    dW_l = R K^T (lam*C + K K^T)^-1, computed through Woodbury with a cached C^-1."""
    def __init__(self, model, tok, **hp):
        super().__init__(model, tok, **hp)
        self.layers = hp.get("layers", [2, 3, 4, 5, 6]); self.lam = hp.get("lam", 15000.0)
        self.zkw = dict(steps=hp.get("steps", 20), lr=hp.get("z_lr", 0.5), clamp=hp.get("clamp", 0.75))
        self.delta = Delta(model); self.z = {}
    def reset(self): self.delta.clear()
    def ztag(self): return f"_s{self.zkw['steps']}_c{self.zkw['clamp']}"
    def precompute(self, reqs):
        self.reset()
        z, nll = z_cached(self.model, self.tok, reqs, self.layers[-1], tag=self.ztag(), **self.zkw)
        self.z = {r["id"]: z[i] for i, r in enumerate(reqs)}; free()
    def solve(self, l, K, R, i):
        Cinv = cov_inv(l, "k", self.hp.get("ridge", 0.0)); CKt = Cinv @ K.T; del Cinv
        A = torch.linalg.solve(self.lam * torch.eye(K.shape[0], device=DEV, dtype=torch.float64) + K @ CKt, CKt.T)
        return R.T, A
    def edit(self, reqs):
        z = torch.stack([self.z[r["id"]] for r in reqs]).double()
        for i, l in enumerate(self.layers):
            K = keys_at(self.model, self.tok, reqs, [l])[l].double()
            cur = resid_at(self.model, self.tok, reqs, self.layers[-1]).double()
            R = (z - cur) / (len(self.layers) - i)
            U, V = self.solve(l, K, R, i)
            self.delta.add(l, U, V); free()
        self.after_edit(reqs)
    def after_edit(self, reqs): pass

class AlphaEdit(MEMIT):
    """Fang et al. 2025. MEMIT, but the update is projected onto the 'null space' of the preserved keys
    (eigvecs of C with eigenvalue < thr) and previously edited keys are remembered:
    dW^T = (P (K K^T + Kp Kp^T) + c I)^-1 P K R^T. solved in the projected basis, same thing, much smaller."""
    def __init__(self, model, tok, **hp):
        super().__init__(model, tok, **hp)
        self.thr, self.c = hp.get("thr", 2e-2), hp.get("L2", 10.0); self.Q, self.prev = {}, {}
    def reset(self):
        super().reset(); self.prev = {}
    def proj(self, l):
        if l not in self.Q:
            C = torch.load(os.path.join(CK, f"cov_k_L{l}.pt"))["C"].to(DEV).double()
            ev, V = torch.linalg.eigh(C); self.Q[l] = V[:, ev < self.thr].cpu(); del C, V; free()
        return self.Q[l].to(DEV)
    def solve(self, l, K, R, i):
        Q = self.proj(l); Kt = K @ Q                                  # [N, r] keys in the null basis
        Kall = torch.cat([Kt, self.prev[l].to(DEV)]) if l in self.prev else Kt
        G = self.c * torch.eye(Kall.shape[0], device=DEV, dtype=torch.float64) + Kall @ Kall.T
        X = (Kt.T - Kall.T @ torch.linalg.solve(G, Kall @ Kt.T)) / self.c   # (cI + Kall^T Kall)^-1 Kt^T
        V = (Q @ X).T; self._new = getattr(self, "_new", {}); self._new[l] = Kt.cpu()
        return R.T, V
    def after_edit(self, reqs):
        for l, Kt in self._new.items():
            self.prev[l] = torch.cat([self.prev[l], Kt]) if l in self.prev else Kt
        self._new = {}
