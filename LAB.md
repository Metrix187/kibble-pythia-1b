# LAB.md — writing facts into pythia-1b without training it

## STATE
- phase: 2 (baselines)
- next action: E03 no-edit + in-context, E04 LoRA, E05 MEMIT, E06 AlphaEdit
- experiments run: 2 (E00 = setup, not counted)
- best method so far: none yet (in-context is the bar: eff 1.00 / para 0.78 / hop 0.88 on tune)
- harness: FROZEN at the phase-0 commit. do not edit evals/.

## Ground rules I set for myself (assumptions, since nobody's around to ask)
- **Model**: `EleutherAI/pythia-1b` (1012M params, fp16). Picked by bake-off (results/E00_model_select.json):
  TinyLlama_v1.1 (1.1B) knew almost nothing (QA 0.30, real-fact probes 0.16), gemma-3-1b is gated (401, and
  I'm not clicking through a licence for you), Qwen2.5-0.5B is half the size. pythia won on the things the
  judge needs: real-fact recall 0.72, in-context multi-hop 0.89.
- **GPU**: RTX 4060 Laptop 8GB, CUDA, torch 2.14.1+cu126. ~1.8GB is already eaten by other apps, so the
  process is capped at 7GB via set_per_process_memory_fraction (OOM -> back off, never spill silently).
- **"1000 facts, hold out 200"**: facts.json has 800 `eval` + 200 `heldout` = the 1000, plus 200 extra `tune`
  facts with different names. ALL tuning (hparams, hypernet training, probing) happens on `tune`. Scale sweep
  on eval is 1/10/100/800; the true 1000 point (800+200) only happens once, in the final report.
- Tune facts only use ~70% of each answer pool, so anything fitted on tune can be checked on unseen answers
  (`target_seen_in_tune`).
- **"No gradient training at edit time"**: I read it strictly for our own methods: no backprop through the
  target model when an edit is requested. Offline, one-time training of a small hypernetwork is allowed
  (the brief says so). The MEMIT baseline does optimise its value vectors by gradient descent, that's how it's
  published, and it's a baseline.
- Edits are held as fp32 side deltas next to the fp16 weights (fp16 would round small updates away). Merging
  back into fp16 is tested as an attack in phase 4.
- Eval batch size is 16 (measured peak 3.4GB at that size, start-at-4 was pointless for 20-token prompts).
  OOM handler halves it.
- Experiment numbering: E00 is setup. Counting toward the 30 cap starts at E01.

## Phase 0 — the judge (done, frozen)
Files: evals/spec.py (templates + pools), evals/build_data.py, evals/core.py (scoring), evals/harness.py (Judge).
- 7 relations (birth city, citizenship, native language, company HQ, town->country, device maker, mineral
  element), 5 subject kinds, all subjects are syllable-soup names. Answer pools were filtered so the base model
  knows the second hop (city->country, country->capital, language->country, company->country, element->symbol).
- Per fact: 1 direct prompt, 5 paraphrases (3 direct rewordings + 2 indirect with the subject buried in a
  story), 1 multi-hop prompt, 2 near-miss-name neighbours (a: same first word, b: same last word), 1
  cross-relation probe on the same subject.
- Global: 500 wikitext-2 prefixes for KL(base||edited), 68 QA items and 180 real same-relation facts the base
  model gets right (exact-match retention), perplexity on 10k wikitext-2 test tokens, 200 ARC-Easy questions.
- Scoring: teacher-forced log-likelihood; "exact match" = greedy decode equals the target, which is the same
  as every teacher-forced argmax matching. Nothing reads generations.
- Gotcha found while building: pythia answers "The capital of France is" with " the". Capital templates had to
  end in "is the city of". (0/74 -> 37/74 known.)

**Sanity (results/E00_sanity.json, tune split, n=200)**
| | eff | para | hop | loc KL | QA | real nbr | name bleed | ppl | ARC |
|---|---|---|---|---|---|---|---|---|---|
| no edit | 0.010 | 0.007 | 0.020 | 0 | 1.00 | 1.00 | 0.007 | 26.17 | 0.475 |
| fact in context | 1.000 | 0.783 | 0.880 | 0 | 1.00 | 1.00 | 0.675 | 26.17 | 0.475 |

Passes. Note the in-context baseline here is "oracle retrieval": the fact is prepended only to prompts about
that fact (and to its near-miss neighbours, like a retriever fooled by a similar name would do — hence 0.675
bleed). Peak VRAM 3390MB.

## Experiment log

### E01 — causal tracing (results/E01_causal_trace.json) — phase 1
Plan: ROME-style tracing on 42 real facts pythia knows (6 per relation). Noise the subject embeddings
(3x embedding std, 3 draws), restore one clean state, measure recovered p(answer). Residual per layer;
mlp/attn outputs over a +-1 layer window.
Result (clean p=0.47, corrupted p=0.03; numbers are recovered probability):
- Residual at the **last subject token** carries the fact through layers 0-10 (0.28-0.37), then drops to
  0.01 at layer 11. Residual at the **last prompt token** is ~0 until layer 10 and 0.33-0.43 from 11 on.
  It's a cliff, not a slope: the handoff happens in one place.
- Attention at the last token does the moving: 0.31 at layer 10, 0.21 at 11-12, nothing elsewhere.
- MLPs at the last subject token matter early: 0.23-0.24 at layers 0-1, 0.08-0.09 at 2-3, gone by ~6.
- Other positions (first/middle subject tokens, tokens after the subject) barely register.
Surprises: how early the MLP site is (layers 0-3 of 16; in GPT-2 XL it's the middle third), and how sharp
the layer 10->11 handoff is. Caveat I believe: embedding noise inflates layer-0/1 MLP effects.
Take-away for editing: write at the last subject token, somewhere in layers ~2-8. Anything written at or
after layer 10 at the subject is too late to be read.
(First run had a span bug for subjects not at the start of the prompt; fixed and rerun, numbers above are the rerun.)

### E02 — spectra (results/E02_spectra.json) — phase 1
Plan: singular values of every mlp/attn matrix; second moments of mlp keys (post-GELU, 8192-d) and mlp
inputs (2048-d) over 200k wikitext-2 train tokens (cached in ckpt/ for the editors).
Result:
- Weights are close to full rank everywhere: mlp_out effective rank 1450-1780 of 2048, attention V/O about
  1000-1200. There's no spare low-rank room sitting in the weights.
- Key covariance: 99% of the energy needs ~7400 of 8192 directions (layer 0 is the odd one: 4700). So there
  is no real null space either. BUT by AlphaEdit's absolute rule (eigenvalue < 0.02) 66-91% of directions in
  layers 0-10 count as "null". Those two statements are both true: the spectrum is flat-ish and small, not
  empty. A null-space projector built that way throws away most of the key, not just the noise.
- The mlp *input* covariance is absurdly lopsided in layers 0-1 (effective rank 27-29 of 2048) and opens up
  later (130 at L2, 300+ at L10-13). One or two directions (the mean) dominate, so anything keyed on the raw
  mlp input has to whiten first.
Surprise: I expected a big dead subspace to hide edits in. There isn't one; "null space" here is a threshold.
Peak VRAM 5418MB (eigendecompositions; first attempt hit the 7GB cap and backed off as designed).

## Pre-registered scoring for the stop condition (written before any editor was scored)
- **Scale score** = mean(efficacy_em, para_em, hop_em) at N=800 (one batch). Also reported at 1/10/100.
- **Sequential score** = the same mean on all 800 facts after 10 batches of 80, plus batch-1 efficacy after
  batch 10 (retention).
- **"Without losing on locality"** vs the published editor, all of: loc_kl <= its loc_kl; loc_qa_em and
  nbr_real_em no more than one item worse (1/68, 1/180); ppl <= 1.005x its ppl; near-miss name bleed no more
  than 0.02 worse.
- "The published editor" = MEMIT (E05). AlphaEdit (E06) is a second published baseline; a winner has to beat
  whichever of the two is better on each of scale and sequential.
- Protocol (methods/run.py): scale N=1 is 10 separate single-fact edits averaged, N=10 is 3 disjoint groups
  averaged, N=100 and N=800 are one batch each. Sequential is 10 x 80 in the fixed eval order.
- Editors only ever see {subject, direct prompt, target, statement, relation name} (values.req). Paraphrases,
  hops and neighbours are judge-only.

## Phase 3 hypotheses (all written before any phase-3 run; baselines E04-E06 were still in progress)
Design: every editor = a **value source** (what should the subject's residual become?) x a **writer** (how do
the weights produce it?), in methods/inventions.py. "oracle" value = MEMIT's gradient-optimised z*, used only
to test writers in isolation. Everything else is forward-only at edit time. Step sizes come from a forward-only
search (try 9 sizes, keep the smallest that makes the answer win on the edit prompt).

- **M1 Echo (late-site unembedding write).** Skip subject enrichment. At layer 13 the last prompt token's MLP
  key already encodes who + what's asked; map that key to the answer's unembedding direction (through the
  final LayerNorm, closed form) with a least-squares write. *Predict:* direct-prompt efficacy >= 0.9 at N<=10;
  paraphrase < 0.2 (key = last-token state, changes with wording); hop ~ 0; collapses with scale because
  last-token keys of one template are near-collinear across subjects.
- **M2 Graft (token-identity superposition).** Add the model's own layer-L representation of the answer token
  (centred) to the subject's residual at an early layer. Read-out heads partly copy token identity, and
  because the graft is early the subject-position MLPs in L+1..9 should enrich it with the answer's own
  attributes -> multi-hop for free. *Predict:* efficacy 0.3-0.7, hop above MEMIT's, cross-relation bleed high
  (the subject answers "Paris" to everything).
- **M3 OV back-projection.** The direction the read-out heads (layers 10-12, from E01) turn into the answer
  token, in closed form: J^T(e_answer - e_mean) along LN -> V_h -> O_h -> final LN -> unembed, heads weighted
  by their measured attention to the subject. *Predict:* efficacy 0.6-0.9 (ignores MLPs 11-15), paraphrase a
  bit below MEMIT, cross-relation bleed worse than MEMIT (direction isn't relation-specific).
- **M4 Gated key neurons (LCMV memory block).** One new neuron per fact beside layer L's MLP. Input weights =
  whitened matched filter C^-1 k / (k^T C^-1 k) (minimum-variance beamformer), saturating gate, output = the
  value. Pretrained weights untouched, every fact has its own slot. *Predict:* sequential retention 100%,
  locality KL ~ 0, flat with scale; weak spot = indirect paraphrases (context shifts the key, gate
  under-fires) and near-miss names that share the last token (gate fires, bleed b up).
- **M5 HyperZ (amortised values).** A linear hypernetwork per relation: answer unembedding row -> residual
  delta, fitted offline on z* of tune facts. Forward-only at edit time. *Predict:* near-oracle on answers seen
  in tune, large drop on unseen answers (it will blend neighbouring answers).
- **M6 Many tiny vs few concentrated.** Same value written at 1 layer vs spread over 5 vs 8. *Predict:* spread
  helps paraphrase a little for the least-squares writer; no effect on locality.
- **M7 In-context transplant.** Use the subject's residual from a run where the fact was just stated before
  it. *Predict:* fails (efficacy <= 0.1). The answer isn't stored on the second mention of the subject, the
  last token reads it straight out of the context.
Kill rule: tune-split score mean(eff, para, hop) under 0.3, or locality clearly worse than MEMIT with no
upside -> dead, no scale/seq run.

### E03 — baselines (a) no edit and (b) fact in context — phase 2
Files: results/E03_NoEdit.json, E03_InContext_oracle.json, E03_InContext_all.json.
Plan: run the standard protocol on the two "no weights touched" baselines.
| N=800, one batch | eff | para | hop | loc KL | QA | real nbr | name bleed | ppl | ARC |
|---|---|---|---|---|---|---|---|---|---|
| no edit | 0.004 | 0.002 | 0.006 | 0 | 1.00 | 1.00 | 0.003 | 26.17 | 0.475 |
| in context, oracle retrieval | 0.998 | 0.762 | 0.858 | 0 | 1.00 | 1.00 | 0.716 | 26.17 | 0.475 |
- Oracle retrieval is flat with scale and with sequence (nothing accumulates), and it's the only thing so far
  that does multi-hop (0.86). Its weak spot is the confusable-retriever case: a near-miss name with the fact
  pasted in front picks up the answer 72% of the time.
- **All facts pasted in front of everything** (the honest "just use the context window" version): N=1 eff
  1.00 / para 0.76 / hop 0.80 but loc KL 0.17, QA 0.84, real-nbr 0.62 (one stray sentence already bends
  unrelated answers). N=10: eff 0.70, para 0.25, hop 0.10, QA 0.91. N=800 is 12.7k tokens and does not fit
  pythia's 2048 window. N=100 (1565 tokens) OOMed inside the frozen judge at batch 16; retry with judge
  batch 2 is queued. So context stuffing collapses between 1 and 10 facts on this model.
Peak VRAM 3390MB (oracle), 4860MB (all-in-context, N<=10).

### E04 — baseline (c) LoRA — phase 2
Files: results/E04_LoRA.json (+ E04_LoRA_tune_*.json).
Plan: hand-rolled LoRA r=8 on every attention + mlp linear in all 16 layers (4.6M params), AdamW. Tune lr x
epochs on the tune split, then scale + sequential (same adapters keep training on each new batch, no replay).
What happened while tuning (11 configs, tune split only):
- Full-statement LM loss wrecked locality at every setting (KL 1.0-2.1 nats, QA retention 0.28-0.72), and
  *lower* learning rates were worse. I blamed fp16 gradient underflow; retrained in fp32 and got the same
  numbers to three digits. So that theory was wrong and the damage is real: 100 templated sentences are enough
  to drag the whole model toward the template. (Side effect of the detour: LoRA now trains in fp32 by
  upcasting in place, which is lossless both ways. MEMIT's z* was checked too: fp16 with loss scaling matches
  fp32 at cosine 0.998-1.000, so that stays in fp16.)
- Switched to the standard editing-baseline loss (answer tokens only). Much better: tune eff 1.00 /
  para 0.93 / hop 0.34, but still loc KL 0.37, QA 0.90, real-nbr 0.65. Picked lr 3e-4, 10 epochs.
Result (eval facts):
| | eff | para | hop | loc KL | QA | real nbr | name bleed | ppl | ARC |
|---|---|---|---|---|---|---|---|---|---|
| N=1 | 1.00 | 0.98 | 0.00 | 0.47 | 0.69 | 0.41 | 1.00 | 28.5 | 0.46 |
| N=10 | 1.00 | 0.92 | 0.30 | 0.36 | 0.86 | 0.49 | 0.92 | 28.6 | 0.455 |
| N=100 | 1.00 | 0.92 | 0.35 | 0.29 | 0.88 | 0.66 | 0.65 | 32.0 | 0.475 |
| N=800 | 1.00 | 0.95 | 0.20 | 0.75 | 0.68 | 0.42 | 0.68 | 48.2 | 0.44 |
| sequential 10x80 | 0.24 | 0.20 | 0.02 | 1.23 | 0.40 | 0.06 | 0.16 | 89.5 | 0.395 |
- Best paraphrase generalisation of anything so far (it beats the fact in context, 0.95 vs 0.76) and it never
  fails to memorise. But it has no notion of *which* subject: one edited fact and every near-miss name gets
  the same answer (bleed 1.00), and real same-relation facts drop to 0.41.
- Sequentially it's catastrophic forgetting: batch-1 efficacy is 0.000 after batch 10, perplexity 26 -> 89.
Peak VRAM 4160MB (fp32 training). Edit time 446s for the sequential 800.
