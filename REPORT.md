# Writing facts into pythia-1b without training it

**Bottom line.** Our editor ("Kibble") beats MEMIT and AlphaEdit on the pre-registered scale and sequential
scores, using forward passes only and without modifying a single pretrained weight. Locality is better on
every measure at 800 facts; in the sequential run one measure is a tie inside tolerance (near-miss bleed 0.299
vs MEMIT's 0.289, where half of MEMIT's edits had already faded). On the 200 held-out facts plus the 800 eval facts edited together (1000 facts) it scores
0.763 against 0.629 for AlphaEdit and 0.625 for MEMIT. It does **not** beat putting the fact in the
context window with a perfect retriever (0.871): that baseline does multi-hop at 0.86 and nothing written
into weights got past 0.54. It breaks on misspelled names (0.51), and its address solver hits a memory wall
between 3800 and 4800 stored slots on an 8GB card.

Setup: `EleutherAI/pythia-1b` in fp16 on an RTX 4060 Laptop (8GB, process capped at 7GB). 1000 invented facts
about invented people, companies, towns, devices and minerals (800 eval + 200 held-out), plus 200 separate
tune facts on which every knob was set. All scoring is likelihood / exact match. Full log: `LAB.md`.

## The method, in under 300 words

Kibble stores each fact as one gated neuron beside the MLP of layer 6 (of 16). No backprop, no pretrained
weight touched.

**Value (what to store).** For a request (subject, prompt, answer), take the residual `h` at the subject's
last token after layer 6. Build a 42-direction dictionary: for each of the 40 attention heads in layers 9-13
(the read-out layers causal tracing found), the closed-form direction `J_h^T (e_answer - e_mean)`: the
unembedding row pulled back through the final LayerNorm, that head's O and V matrices, and its layer's
LayerNorm. Add two "graft" directions: the model's own layer-6 representation of the answer token. Start at
`h + 0.75|h| * graft`, then six rounds of forward-difference ascent on `log p(answer | prompt)` over three
lead-ins: nudge along every direction, read the slopes, step with a forward-only line search, cap at `1.5|h|`.

**Address (when to fire).** Key `k` = layer-6 MLP hidden activation at the subject's last token, averaged
over six lead-ins. For each fact also make 8 near-miss keys by swapping one subject token. Solve one linear
system for all filters: `W = [I 0] (A C^-1 A^T + eps I)^-1 A C^-1`, with `A` every stored key and near-miss
and `C` the key covariance on ordinary text. Filter `i` answers 1 on its own key, 0 on every other stored key
and near-miss, and is minimum-variance elsewhere (an LCMV beamformer).

**Block.** `out += clamp((W k - 0.35) / 0.2, 0, 1) @ V`, row `i` of `V` being `value_i - h_i`. Two ReLUs per
fact.

New facts append rows and re-solve `W`; stored values never change. Deleting a row forgets one fact.
Deleting the block restores the model bit for bit.

## Results

#### Table 1. Every method, all eight evals (800 eval facts unless the column says otherwise)

| method | 1 efficacy | 2 paraphrase | 3 multi-hop | 4 loc KL | 4 QA kept | 5 real nbrs kept | 5 near-miss bleed | 6 ppl | 6 ARC | 7 scale score @800 | 8 seq score (10x80) | 8 batch-1 after batch 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no edit | 0.004 | 0.002 | 0.006 | 0.0000 | 1.000 | 1.000 | 0.003 | 26.17 | 0.475 | 0.004 | 0.004 | 0.000 |
| fact in context, perfect retriever | 0.998 | 0.762 | 0.858 | 0.0000 | 1.000 | 1.000 | 0.716 | 26.17 | 0.475 | 0.872 | 0.872 | 0.988 |
| LoRA (r=8, answer-token loss) | 1.000 | 0.953 | 0.201 | 0.7466 | 0.676 | 0.417 | 0.676 | 48.23 | 0.440 | 0.718 | 0.155 | 0.000 |
| MEMIT | 0.974 | 0.666 | 0.309 | 0.0491 | 0.941 | 0.822 | 0.397 | 27.65 | 0.440 | 0.650 | 0.439 | 0.613 |
| AlphaEdit | 0.969 | 0.686 | 0.309 | 0.0570 | 0.897 | 0.806 | 0.399 | 27.88 | 0.445 | 0.655 | 0.650 | 0.912 |
| FD values + MEMIT's writer (ablation) | 0.828 | 0.554 | 0.316 | 0.1237 | 0.824 | 0.667 | 0.341 | 29.29 | 0.445 | 0.566 | 0.329 | 0.475 |
| MEMIT's z* + gated neurons (ablation) | 0.999 | 0.835 | 0.420 | 0.0012 | 0.971 | 0.994 | 0.314 | 26.24 | 0.480 | 0.751 | 0.742 | 0.988 |
| **Kibble** (FD values + gated neurons) | 0.954 | 0.809 | 0.526 | 0.0149 | 1.000 | 0.983 | 0.299 | 26.35 | 0.475 | 0.763 | 0.743 | 0.988 |

Columns 1-6 are the single-batch N=800 run. Scores = mean(efficacy, paraphrase, multi-hop). Lower is better for loc KL, bleed and ppl. Base model: ppl 26.17, ARC 0.475.

#### Table 2. Sequential run, final state after 10 batches of 80

| method | efficacy | paraphrase | multi-hop | loc KL | QA kept | real nbrs kept | near-miss bleed | ppl | ARC |
|---|---|---|---|---|---|---|---|---|---|
| no edit | 0.004 | 0.002 | 0.006 | 0.0000 | 1.000 | 1.000 | 0.003 | 26.17 | 0.475 |
| fact in context, perfect retriever | 0.998 | 0.762 | 0.858 | 0.0000 | 1.000 | 1.000 | 0.716 | 26.17 | 0.475 |
| LoRA (r=8, answer-token loss) | 0.241 | 0.204 | 0.019 | 1.2307 | 0.397 | 0.061 | 0.156 | 89.47 | 0.395 |
| MEMIT | 0.664 | 0.416 | 0.237 | 0.0387 | 0.941 | 0.822 | 0.289 | 27.30 | 0.460 |
| AlphaEdit | 0.950 | 0.680 | 0.319 | 0.0538 | 0.912 | 0.839 | 0.391 | 27.84 | 0.470 |
| FD values + MEMIT's writer (ablation) | 0.505 | 0.288 | 0.194 | 0.0977 | 0.897 | 0.711 | 0.216 | 28.52 | 0.440 |
| MEMIT's z* + gated neurons (ablation) | 0.990 | 0.818 | 0.419 | 0.0006 | 0.971 | 0.983 | 0.320 | 26.21 | 0.480 |
| **Kibble** (FD values + gated neurons) | 0.941 | 0.790 | 0.499 | 0.0068 | 1.000 | 0.972 | 0.299 | 26.26 | 0.475 |

#### Table 3. Failure curve: score = mean(efficacy, paraphrase, multi-hop) as the fact count grows

| method | 1 | 10 | 100 | 800 | 1000 (800 + held-out) | paraphrase 1 -> 800 | loc KL 1 -> 800 |
|---|---|---|---|---|---|---|---|
| fact in context, ALL facts pasted in | 0.853 | 0.351 | - | does not fit | - | 0.76 -> n/a | 0.1725 -> n/a |
| no edit | 0.000 | 0.000 | 0.003 | 0.004 | 0.004 | 0.00 -> 0.00 | 0.0000 -> 0.0000 |
| fact in context, perfect retriever | 0.853 | 0.829 | 0.863 | 0.872 | 0.871 | 0.76 -> 0.76 | 0.0000 -> 0.0000 |
| LoRA (r=8, answer-token loss) | 0.660 | 0.740 | 0.755 | 0.718 | 0.735 | 0.98 -> 0.95 | 0.4719 -> 0.7466 |
| MEMIT | 0.713 | 0.800 | 0.728 | 0.650 | 0.625 | 0.84 -> 0.67 | 0.0001 -> 0.0491 |
| AlphaEdit | 0.713 | 0.802 | 0.743 | 0.655 | 0.629 | 0.84 -> 0.69 | 0.0001 -> 0.0570 |
| FD values + MEMIT's writer (ablation) | 0.773 | 0.856 | 0.712 | 0.566 | - | 0.82 -> 0.55 | 0.0001 -> 0.1237 |
| MEMIT's z* + gated neurons (ablation) | 0.760 | 0.796 | 0.753 | 0.751 | 0.752 | 0.78 -> 0.83 | 0.0000 -> 0.0012 |
| **Kibble** (FD values + gated neurons) | 0.787 | 0.824 | 0.782 | 0.763 | 0.763 | 0.76 -> 0.81 | 0.0000 -> 0.0149 |

#### Table 4. Held-out facts (run once, after everything was frozen)

| method | set | efficacy | paraphrase | multi-hop | loc KL | QA kept | real nbrs kept | near-miss bleed | ppl | ARC | score |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no edit | 200 held-out | 0.005 | 0.001 | 0.000 | 0.0000 | 1.000 | 1.000 | 0.000 | 26.17 | 0.475 | 0.002 |
| no edit | all 1000 | 0.004 | 0.002 | 0.005 | 0.0000 | 1.000 | 1.000 | 0.003 | 26.17 | 0.475 | 0.004 |
| fact in context, perfect retriever | 200 held-out | 0.995 | 0.779 | 0.825 | 0.0000 | 1.000 | 1.000 | 0.728 | 26.17 | 0.475 | 0.866 |
| fact in context, perfect retriever | all 1000 | 0.997 | 0.766 | 0.851 | 0.0000 | 1.000 | 1.000 | 0.718 | 26.17 | 0.475 | 0.871 |
| LoRA (r=8, answer-token loss) | 200 held-out | 1.000 | 0.936 | 0.360 | 0.3186 | 0.868 | 0.644 | 0.688 | 32.10 | 0.490 | 0.765 |
| LoRA (r=8, answer-token loss) | all 1000 | 1.000 | 0.924 | 0.280 | 0.6004 | 0.706 | 0.306 | 0.654 | 44.40 | 0.455 | 0.735 |
| MEMIT | 200 held-out | 0.980 | 0.767 | 0.350 | 0.0115 | 0.985 | 0.933 | 0.457 | 26.52 | 0.470 | 0.699 |
| MEMIT | all 1000 | 0.945 | 0.631 | 0.300 | 0.0636 | 0.956 | 0.800 | 0.377 | 28.09 | 0.435 | 0.625 |
| AlphaEdit | 200 held-out | 0.980 | 0.777 | 0.365 | 0.0124 | 0.985 | 0.933 | 0.467 | 26.55 | 0.465 | 0.707 |
| AlphaEdit | all 1000 | 0.949 | 0.641 | 0.296 | 0.0774 | 0.912 | 0.744 | 0.378 | 28.41 | 0.455 | 0.629 |
| MEMIT's z* + gated neurons (ablation) | 200 held-out | 0.995 | 0.839 | 0.420 | 0.0001 | 1.000 | 0.978 | 0.323 | 26.17 | 0.475 | 0.751 |
| MEMIT's z* + gated neurons (ablation) | all 1000 | 0.998 | 0.836 | 0.423 | 0.0005 | 1.000 | 0.983 | 0.318 | 26.18 | 0.475 | 0.752 |
| **Kibble** (FD values + gated neurons) | 200 held-out | 0.970 | 0.823 | 0.540 | 0.0001 | 1.000 | 0.978 | 0.310 | 26.18 | 0.475 | 0.778 |
| **Kibble** (FD values + gated neurons) | all 1000 | 0.958 | 0.808 | 0.522 | 0.0043 | 1.000 | 0.983 | 0.301 | 26.19 | 0.475 | 0.763 |

#### Table 5. Peak VRAM and cost

| method | peak VRAM (MB) | precompute (s, 800 facts) | write 800 facts (s) | backprop at edit time? | touches pretrained weights? |
|---|---|---|---|---|---|
| no edit | 3390 | 0 | 0 | no | no |
| fact in context, perfect retriever | 3390 | 0 | 0 | no | no |
| LoRA (r=8, answer-token loss) | 4169 | 0 | 359 | yes (training) | adapters on every layer |
| MEMIT | 4109 | 402 | 135 | yes (z*) | yes, 5 layers |
| AlphaEdit | 5132 | 402 (same z*, from cache) | 135 | yes (z*) | yes, 5 layers |
| FD values + MEMIT's writer (ablation) | 3987 | 1227 (same values, from cache) | 129 | no | yes, 5 layers |
| MEMIT's z* + gated neurons (ablation) | 5448 | 402 (same z*, from cache) | 53 | yes (z*) | no |
| **Kibble** (FD values + gated neurons) | 5448 | 1227 | 61 | no | no |

Peak VRAM is the max over that method's scale, sequential and held-out runs (the 1000-fact address solve is where Kibble's 5448MB comes from; it is 4566MB at 800). Highest anywhere in the project: 6139MB in the decoy stress test before it hit the cap, and 6834MB in the failed all-facts-in-context run. Nothing exceeded the 7GB cap; three runs hit it and backed off or were logged dead.

## What the numbers say

- **Scale.** MEMIT and AlphaEdit slide as facts pile up (paraphrase 0.84 -> 0.67-0.69 from 1 to 800 facts,
  locality KL x500). Kibble's paraphrase and multi-hop stay put from 100 to 1000 facts (score 0.78 -> 0.76);
  its efficacy slips from 1.00 at 10 facts to 0.95 at 800. It is *worse* than both published editors at a
  single fact on paraphrase (0.76 vs 0.84) and on raw efficacy at 800 (0.954 vs 0.974); it wins from ~100
  facts up because it barely decays. I did not find a quality collapse inside the judge's range; the limit I
  did find is the solver's memory (see the attacks).
- **Held-out.** On the 200 held-out facts alone: Kibble 0.778, AlphaEdit 0.707, MEMIT 0.699. Same ordering
  as on the eval facts, so the tuning did not overfit them.
- **LoRA is closer than it looks on the headline score** (0.735 at 1000 facts, best paraphrase of anything
  at 0.92-0.95). It pays for it everywhere else: perplexity 26 -> 44, real same-relation facts down to 0.31,
  and it cannot be applied twice (sequential score 0.155).
- **Sequential.** MEMIT forgets (batch 1 falls to 0.61 after ten batches), AlphaEdit mostly fixes that
  (0.91), Kibble holds 0.93-0.99 across four seeds. LoRA goes to 0.00.
- **Why.** The ablations pull it apart: MEMIT's own values through the gated writer score 0.751 / 0.742
  (scale / sequential); our forward-only values through MEMIT's linear writer collapse to 0.566 / 0.329.
  So the writer explains the scaling. The forward-only values are noisier than gradient-found ones (twice
  the step size) and only work because the gate keeps them from leaking.
- **Multi-hop.** Every weight editor got some (I predicted none). Kibble's 0.53 is the best in-weights
  number but far from the context window's 0.86, and on real, famous subjects it drops to 0.32, below
  AlphaEdit's 0.48.
- **Nothing beats (b) outright.** A perfect retriever plus the context window wins the headline score with
  zero locality cost. Its failure is confusable names (0.72 bleed vs our 0.30), and the honest
  no-retriever version (paste every fact in front of everything) collapses between 1 and 10 facts.

## Attacks on the winner (phase 4) — worst cases

| attack | Kibble | AlphaEdit / MEMIT |
|---|---|---|
| adversarial prompts, 800 facts in, worst well-formed case | 0.640 (possessive template) | 0.585 |
| name with a typo | **0.510** | 0.570 |
| a different edited fact stated first | 0.845 | 0.685 |
| answers with that other fact's answer instead | 0.085 | 0.175 |
| overwrite 56 real facts: new answer / paraphrase / multi-hop | 0.964 / 0.829 / 0.321 | 1.000 / 0.779 / 0.482 |
| ... untouched real facts still right | 0.935 | 0.879 |
| edit then un-edit | bit-for-bit (sha256 of all params, max log-prob diff 0.0) | MEMIT merged into fp16 then subtracted: 17.5% of entries not restored |
| forget one fact of 800 | that fact 0.00, others unchanged at 0.94 | not supported |
| 4-bit (NF4) after editing: eff / para / hop | 0.943 / 0.755 / 0.542 | 0.934 / 0.599 / 0.306 |
| ... QA kept (4-bit unedited model: 0.912) | 0.926 | 0.824 |
| three fresh seeds, worst: scale / seq / batch-1 retention | 0.759 / 0.742 / 0.925 | not rerun |
| 800 facts + 3000 decoy slots | unchanged (0.954 / 0.813) | - |
| 800 facts + 4000 decoy slots | **out of memory** at the 7GB cap | - |

Overall worst case: **0.51 exact match when the subject's name is misspelled**, and a hard memory wall at
roughly 4000-4800 slots. Both come from the same place: it is a key-addressed memory with a jointly solved
address table.

## What did not work, and why

- **Echo** (write the answer's unembedding direction at the last prompt token, layer 13). Tune: 0.65 / 0.20 /
  0.06. The closed-form direction is fine; the address is not. Last-token keys of one template are too alike
  to tell 100 subjects apart, and they change with wording. Killed.
- **In-context transplant** (use the subject's residual from a run where the fact was just stated). 0.01. The
  second mention of the subject doesn't carry the answer; the model reads it straight off the context.
- **Graft alone** (add the model's own representation of the answer token to the subject). 0.57 efficacy but
  0.68 multi-hop, double MEMIT's. Fails on relations where the read-out can't just copy (native language
  0.15, device maker 0.15). Kept as the starting point of the value search.
- **OV back-projection as one direction** (attention-weighted sum over read-out heads). 0.43. Too crude;
  became the per-head dictionary. **Greedy pursuit** over that dictionary (one direction per round) stalled at
  log p = -1.10; joint forward-difference ascent reaches -0.12.
- **Gated neurons spread over 5 layers**: worse than one layer (0.73 vs 0.82). **Plain matched filters
  without nulls**: facts whose names end in the same word fire each other (efficacy 0.82, bleed 0.45-0.50).
- **Our values with MEMIT's writer**: worse than MEMIT (see ablation).
- **LoRA with a language-model loss on the fact sentences**: KL 1-2 nats at every learning rate. I blamed
  fp16 gradient noise, tested it in fp32, got identical numbers, and was wrong. Answer-token loss fixed most
  of it; LoRA still has no idea *which* subject it is editing (bleed 1.00 after one fact).
- **MEMIT on layers 1-3 with the raw covariance**: destroys the model (KL 2-3.5). Those covariances are
  near-singular; a 1% ridge fixes it. **AlphaEdit with a tighter null-space threshold**: efficacy 0.01-0.75.
  This model has no real null space in its keys, only a threshold (99% of key energy needs 7400 of 8192
  directions).
- **All facts in the context window at N=100**: could not be scored inside 7GB by the frozen judge (three
  attempts). N=800 is 12.7k tokens against a 2048 window.
- **Not run**: the hypernetwork idea (M5) and a proper spread-vs-concentrated test for the least-squares
  writer (M6). The stop condition arrived at experiment 13.
- **My own wrong predictions**: weight editors would get ~0 multi-hop (they get 0.3-0.5); the graft would
  make the subject blurt the answer to unrelated questions (0.00-0.04); indirect paraphrases would be the
  gate's weak spot (they are its best case, 0.91; plain rewordings are weaker); the graft is what gives
  Kibble its multi-hop (the head-direction search gets 0.51 without it; the graft buys paraphrase instead).

## Limits of what this shows

- One model, seven templated relations, answers restricted to ones whose second hop the base model knows.
  Held-out facts are new names with the same templates and answer pools.
- "Into the weights" is generous for Kibble: it adds neurons, it does not rewrite existing ones. The block is
  expressible as extra rows/columns of a wider MLP, which is what makes the un-edit exact.
- Value search costs ~1.5 s/fact forward-only (MEMIT's backprop search: 0.5 s/fact).
- Gated side memories and whitened keys are known ideas (T-Patcher, GRACE, ROME's C^-1 k). What I have not
  seen before, and can only claim for this setup: the joint LCMV address solve with made-up near-miss nulls,
  and the forward-difference value search over a closed-form per-head dictionary seeded with a token graft.
- Near-miss names sharing most tokens still pick up the edit 30% of the time.
- Seeds reshuffle fact order and near-miss sampling only; the covariance sample was fixed.

## Reproduce

```
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv\Scripts\python.exe transformers datasets accelerate safetensors numpy sentencepiece protobuf huggingface_hub

# phase 0: judge (frozen after this)
.venv\Scripts\python.exe evals\select_model.py
.venv\Scripts\python.exe evals\build_data.py
.venv\Scripts\python.exe evals\sanity.py
# phase 1
.venv\Scripts\python.exe methods\e01_trace.py
.venv\Scripts\python.exe methods\e02_spectra.py            # also caches the covariances every editor needs
# phase 2 (add --proto tune --tag x for the tuning runs listed in LAB.md)
.venv\Scripts\python.exe methods\run.py --exp E03 --method baselines:NoEdit --proto scale,seq
.venv\Scripts\python.exe methods\run.py --exp E03 --method baselines:InContext --proto scale,seq --tag oracle
.venv\Scripts\python.exe methods\run.py --exp E03 --method baselines:InContext --proto scale,seq --tag all --hp mode=all
.venv\Scripts\python.exe methods\run.py --exp E04 --method baselines:LoRA --proto scale,seq --hp lr=3e-4 epochs=10
.venv\Scripts\python.exe methods\run.py --exp E05 --method baselines:MEMIT --proto scale,seq --hp "layers=[2,3,4,5,6]" lam=1500 ridge=1e-2
.venv\Scripts\python.exe methods\run.py --exp E06 --method baselines:AlphaEdit --proto scale,seq --hp "layers=[2,3,4,5,6]" thr=2e-2 L2=10
# phase 3: the killed ideas (tune split)
.venv\Scripts\python.exe methods\run.py --exp E07 --method inventions:Composite --proto tune --tag echo_tune --hp value=unembed writer=ls site=last L=13 "layers=[13]" lam=1500
.venv\Scripts\python.exe methods\run.py --exp E08 --method inventions:Composite --proto tune --tag ctx_tune --hp writer=ls lam=1500 ridge=1e-2 value=ctx L=6 "layers=[2,3,4,5,6]"
.venv\Scripts\python.exe methods\run.py --exp E09 --method inventions:Composite --proto tune --tag graft_L6_tune --hp writer=ls lam=1500 ridge=1e-2 value=graft L=6 "layers=[2,3,4,5,6]"
.venv\Scripts\python.exe methods\run.py --exp E10 --method inventions:Composite --proto tune --tag ov_L6_tune --hp writer=ls lam=1500 ridge=1e-2 value=ov L=6 "layers=[2,3,4,5,6]"
.venv\Scripts\python.exe methods\gate_diag.py --L 6 --ridges "0,1e-2,1e-1"
# phase 3: the gated writer with MEMIT's values, then the winner
.venv\Scripts\python.exe methods\run.py --exp E11 --method inventions:Composite --proto scale,seq --hp value=oracle writer=gate L=6 "layers=[6]" th1=0.35 th2=0.55 neg=8 neg_eps=0.1
.venv\Scripts\python.exe methods\run.py --exp E13 --method inventions:Composite --proto scale,seq --hp value=fd "read=[9,10,11,12,13]" L=6 writer=gate "layers=[6]" th1=0.35 th2=0.55 neg=8 neg_eps=0.1
# phase 4 (K = the E13 --method/--hp arguments)
.venv\Scripts\python.exe methods\attack.py --mode adv    --exp E14 K
.venv\Scripts\python.exe methods\attack.py --mode contra --exp E15 K
.venv\Scripts\python.exe methods\attack.py --mode unedit --exp E16 K
.venv\Scripts\python.exe methods\attack.py --mode quant  --exp E17 K
.venv\Scripts\python.exe methods\run.py --exp E18 --seed 1 --tag seed1 --proto scale,seq K seed=1      # also seeds 2, 3
.venv\Scripts\python.exe methods\run.py --exp E19 --method inventions:Composite --proto scale,seq --tag fd_ls --hp value=fd "read=[9,10,11,12,13]" L=6 writer=ls "layers=[2,3,4,5,6]" lam=1500 ridge=1e-2
.venv\Scripts\python.exe methods\attack.py --mode stress --exp E20 K
# held-out, once per method
.venv\Scripts\python.exe methods\run.py --exp E21 --proto heldout K        # and the same for each baseline
.venv\Scripts\python.exe methods\report_tables.py                         # regenerates every table above
```

Seeds are pinned (0 unless given). Each run writes `results/<exp>_<method>[_tag].json` with
`peak_vram_mb` in it; value vectors and covariances are cached in `ckpt/` so reruns resume.
