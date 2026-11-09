# model loading + likelihood scoring. everything is teacher-forced, nothing ever reads a generation.
import os, sys, json, time, math
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
VRAM_CAP_GB = 7.0
if DEV == "cuda":
    # hard stop at 7GB so we OOM (and back off) instead of quietly spilling into shared memory
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GB * 2**30 / tot))

def jload(p):
    with open(p, encoding="utf-8") as f: return json.load(f)

def jdump(o, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f: json.dump(o, f, indent=1, ensure_ascii=False)

def peak_mb():
    return round(torch.cuda.max_memory_allocated() / 2**20, 1) if DEV == "cuda" else 0.0

def free():
    import gc; gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()

def load_model(name, dtype=torch.float16):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype if DEV == "cuda" else torch.float32)
    model.to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model, tok

def enc(tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None: ids = [tok.bos_token_id] + ids
    return ids

def split_target(tok, prompt, target):
    """ids for the prompt, and the ids ' target' adds when glued on (tokenizer-agnostic leading space)."""
    p = enc(tok, prompt); full = enc(tok, prompt + " " + target)
    if full[:len(p)] != p:  # tokenizer merged across the boundary. rare; fall back to standalone target
        t = tok(" " + target, add_special_tokens=False)["input_ids"]
        return p, t
    return p, full[len(p):]

def _batches(n, bs):
    for i in range(0, n, bs): yield i, min(n, i + bs)

def _forward(model, ids_list, bs):
    """yields (index list, logits[b, T, V], lengths). right padded; sorted by length to keep padding small.
    on OOM: halve the batch and carry on."""
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    i = 0
    while i < len(order):
        idx = order[i:i + bs]
        L = max(len(ids_list[j]) for j in idx)
        x = torch.zeros(len(idx), L, dtype=torch.long); m = torch.zeros(len(idx), L, dtype=torch.long)
        for r, j in enumerate(idx):
            x[r, :len(ids_list[j])] = torch.tensor(ids_list[j]); m[r, :len(ids_list[j])] = 1
        try:
            with torch.no_grad():
                out = model(input_ids=x.to(DEV), attention_mask=m.to(DEV)).logits
        except torch.OutOfMemoryError:
            free()
            if bs == 1: raise
            bs = max(1, bs // 2); print(f"[oom] eval batch -> {bs}", flush=True); continue
        yield idx, out, [len(ids_list[j]) for j in idx]
        i += len(idx)

def score(model, tok, prompts, targets, bs=16, prefix=None):
    """teacher-forced score of ' target' after each prompt.
    returns per item dict: lp (sum log prob of target), p1 (prob of first target token), em (greedy == target)."""
    if prefix is None: prefix = [""] * len(prompts)
    seqs, spans = [], []
    for pre, p, t in zip(prefix, prompts, targets):
        pi, ti = split_target(tok, pre + p, t)
        seqs.append(pi + ti); spans.append((len(pi), len(ti)))
    res = [None] * len(prompts)
    for idx, logits, _ in _forward(model, seqs, bs):
        lp = torch.log_softmax(logits.float(), -1)
        for r, j in enumerate(idx):
            a, n = spans[j]; tgt = torch.tensor(seqs[j][a:a + n], device=lp.device)
            rows = lp[r, a - 1:a - 1 + n]
            tok_lp = rows.gather(1, tgt[:, None])[:, 0]
            res[j] = dict(lp=tok_lp.sum().item(), p1=tok_lp[0].exp().item(), n=n,
                          em=bool((rows.argmax(-1) == tgt).all().item()))
    return res

def next_logprobs(model, tok, prompts, bs=16, prefix=None):
    """log-softmax over the next token for every prompt. [N, V] float32 on cpu."""
    if prefix is None: prefix = [""] * len(prompts)
    seqs = [enc(tok, pre + p) for pre, p in zip(prefix, prompts)]
    out = [None] * len(prompts)
    for idx, logits, lens in _forward(model, seqs, bs):
        for r, j in enumerate(idx):
            out[j] = torch.log_softmax(logits[r, lens[r] - 1].float(), -1).cpu()
    return torch.stack(out)

def perplexity(model, chunks, bs=4, prefix_ids=None):
    """chunks: list of token id lists (each starts with bos). token-level ppl over the chunk tokens only;
    prefix_ids (if any) sit between bos and the chunk and are not scored."""
    k = len(prefix_ids) if prefix_ids else 0
    seqs = [c[:1] + list(prefix_ids) + c[1:] for c in chunks] if k else chunks
    nll, n = 0.0, 0
    for idx, logits, lens in _forward(model, seqs, bs):
        lp = torch.log_softmax(logits.float(), -1)
        for r, j in enumerate(idx):
            ids = torch.tensor(seqs[j], device=lp.device)
            nll -= lp[r, k:len(ids) - 1].gather(1, ids[k + 1:, None]).sum().item(); n += len(ids) - 1 - k
    return math.exp(nll / n)
