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
