# LAB.md — writing facts into pythia-1b without training it

## STATE
- phase: 4 (attack the winner)
- next action: E14 adversarial, E15 contradiction, E16 un-edit, E17 4-bit, E18 seeds, E19 ablation, then held-out + REPORT.md
- experiments run: 13 (E00 = setup, not counted; E12 slot unused)
- best method so far: Kibble = FD values + gated key neurons (E13). scale 0.763, seq 0.743, stop condition MET
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

### E05 — baseline (d) MEMIT — phase 2
Files: results/E05_MEMIT.json (+ 15 tune files E05_MEMIT_tune_A..O).
Plan: faithful MEMIT (z* by 20 Adam steps on a residual delta with 6 lead-in contexts + KL term, then
dW = R K^T (lam C + K K^T)^-1 spread over 5 layers), covariance from 200k wikitext-2 train tokens. Tune
layers / lam on the tune split.
Tuning notes: any range touching layers 1-3 with the raw covariance destroyed the model (loc KL 2-3.5 nats).
Those layers' key covariances are near-singular (E02), so the solve puts huge gain on directions ordinary
text does use a little. A 1% ridge on C fixes it. Lower lam helped efficacy. Winner: layers 2-6, lam 1500,
ridge 0.01 (tune: eff 1.00 / para 0.83 / hop 0.35 / KL 0.005).
| | eff | para | hop | loc KL | QA | real nbr | name bleed | ppl | ARC |
|---|---|---|---|---|---|---|---|---|---|
| N=1 | 1.000 | 0.840 | 0.300 | 0.0001 | 0.987 | 0.997 | 0.500 | 26.18 | 0.475 |
| N=10 | 1.000 | 0.833 | 0.567 | 0.0006 | 0.980 | 0.991 | 0.533 | 26.17 | 0.465 |
| N=100 | 0.990 | 0.774 | 0.420 | 0.0055 | 0.956 | 0.950 | 0.465 | 26.37 | 0.470 |
| N=800 | 0.974 | 0.666 | 0.309 | 0.0491 | 0.941 | 0.822 | 0.397 | 27.65 | 0.440 |
| sequential 10x80 | 0.664 | 0.416 | 0.237 | 0.0387 | 0.941 | 0.822 | 0.289 | 27.30 | 0.460 |
- **Scale score 0.650, sequential score 0.439.** Batch-1 efficacy over time: 0.99, 0.78, 0.75, 0.70, 0.66,
  0.62, 0.59, 0.61, 0.60, 0.61. Later batches overwrite earlier ones (final efficacy by batch: 0.61 ... 0.95).
- Surprise: multi-hop is not zero (0.31-0.57). I predicted ~0 for every weight editor.
- Weak spots: near-miss names pick up the edit 40-53% of the time, real same-relation facts fall to 0.82 at 800.
- z* costs 0.5 s/fact (402 s for 800), peak VRAM 4001MB. z* is computed once on the pristine model and reused
  for the sequential run (it is an absolute target for the layer-6 residual, same assumption batch MEMIT makes).

### E06 — second published baseline: AlphaEdit — phase 2
Files: results/E06_AlphaEdit.json (+ 4 tune files).
Plan: MEMIT's z*, but the update is projected onto the "null space" of preserved keys (eigenvalue < thr)
and remembers earlier edit keys. Same layers 2-6. Tuned thr x L2 on tune; the published defaults (thr 0.02,
L2 10) won. A tighter null space (thr 0.005 / 0.001) killed efficacy (0.75 / 0.01), which fits E02: there is
no real null space in this model, only a threshold.
| | eff | para | hop | loc KL | QA | real nbr | name bleed | ppl | ARC |
|---|---|---|---|---|---|---|---|---|---|
| N=100 | 0.990 | 0.780 | 0.460 | 0.006 | 0.985 | 0.950 | 0.460 | 26.36 | 0.465 |
| N=800 | 0.969 | 0.686 | 0.309 | 0.057 | 0.897 | 0.806 | 0.399 | 27.88 | 0.445 |
| sequential 10x80 | 0.950 | 0.680 | 0.319 | 0.054 | 0.912 | 0.839 | 0.391 | 27.84 | 0.470 |
- **Scale score 0.655, sequential score 0.650.** Batch-1 efficacy after batch 10: 0.912. It really does fix
  MEMIT's sequential problem. Locality is slightly worse than MEMIT's.
- Peak VRAM 5025MB.

**THE BAR (from E05/E06):** scale score > 0.655, sequential score > 0.650, and locality no worse than the
better of the two: loc KL <= 0.049 (scale) / 0.039 (seq), QA >= 0.926, real-nbr >= 0.833 (seq) / 0.816
(scale), ppl <= 27.44 (seq) / 27.79 (scale), name bleed <= 0.42. And for honesty: fact-in-context with a
perfect retriever (0.873 on both, zero locality cost) is above all of them. Nothing in phase 2 beats (b).

Footnote on E03: the 100-facts-all-in-context point is dead after 3 attempts (token cap, then OOM at judge
batch 16, then OOM at judge batch 2 inside the frozen perplexity routine at 6.8GB). Logged as too big for
the box, moving on.

### E07 — M1 Echo (late-site unembedding write) — KILLED
File: results/E07_Composite_echo_tune.json. Run: value=unembed, site=last, L=13, least-squares writer.
Result (tune, N=100): eff 0.65, para 0.20, hop 0.06, loc KL 0.002, real-nbr 0.81.
vs prediction: paraphrase < 0.2 (got 0.20, right), hop ~ 0 (right). Efficacy I said >= 0.9 at N<=10; at N=100
it is already down to 0.65, which is the predicted scale collapse arriving early. The closed-form direction
itself is fine (forward-only search made the answer win on 100% of facts with a step of 0.21|h|); the key is
the problem. Last-token keys of one template are too alike to address 100 facts. Dead.

### E08 — M7 in-context transplant — KILLED
File: results/E08_Composite_ctx_tune.json. Result: eff 0.01, para 0.008, hop 0.02. Exactly the predicted
failure: the second mention of the subject does not carry the answer, the model reads it off the context.

### E09 — M2 Graft (token-identity superposition) — KEPT as an ingredient
Files: results/E09_Composite_graft_L6_tune.json, _L3_tune.json, diag_rel_graft_L6.json, diag_rel_graft_stmt_L6.json.
Result (tune, N=100): eff 0.57-0.59, para 0.34-0.35, **hop 0.66-0.68**, loc KL 0.016-0.026, bleed 0.26-0.29.
vs prediction: efficacy 0.3-0.7 (right), hop above MEMIT's (right, and by a lot: 0.68 vs 0.35), cross-relation
bleed high (WRONG, it's 0.00: the subject does not blurt the answer to unrelated questions).
Per relation: works where the read-out can copy an entity (citizenship 0.85, HQ city 0.80, town 0.80, element
0.75), fails on native language (0.15) and device maker (0.15). Multi-hop is high everywhere (0.45-0.90).
Layer 3 vs 6 makes no difference. Using the answer token as it sits in the stated fact instead of in bland
carrier phrases: no real change. Ceiling check: MEMIT's own z* plus a graft of size 1.0|h| keeps efficacy
~0.94 and lifts hop from 0.34 to 0.62.
So: not an editor on its own, but the only thing found that buys multi-hop in weights. It goes into the mix.

### E10 — M3 OV back-projection — MUTATED
Files: results/E10_Composite_ov_L6_tune.json, _L8_tune.json.
Result (tune): eff 0.43 / para 0.18 / hop 0.30 (L6); 0.39 / 0.17 / 0.27 (L8).
vs prediction: I said 0.6-0.9 efficacy; it's 0.43. One attention-weighted direction is too crude. The
directions are not useless though, which led to the mutation below.

### M8 (mutation of M2+M3, hypothesis written after E09/E10, before its harness runs) — FD value
Honest timeline: I ran two tune-split diagnostics of this idea (diag_rel_pursuit_a.json, diag_rel_fd_a.json)
before writing this paragraph, so it is a hypothesis informed by its own first look.
Idea: MEMIT finds its value by backprop. Do the same search with forward passes only, inside a small
dictionary: one closed-form read-out direction per attention head in layers 9-13 (40 of them, the E10 algebra
without the attention weighting) plus two graft directions. Start at 0.75|h| of graft, then 6 rounds of:
nudge along every dictionary direction, read the change in log p(answer) over 3 lead-ins, step along the
implied slope with a forward-only line search, cap the step at 1.5|h|.
Prediction: reaches most of MEMIT's efficacy, keeps part of the graft's multi-hop, costs more forward passes
than MEMIT costs backward ones; fails where no head path exists (expected: weaker direct paraphrases).
First look: log p(answer) -2.67 -> -0.12, 96% of facts above p=0.5; with MEMIT's writer per-relation direct
0.90-1.00, paraphrase ~0.76, hop ~0.48. Greedy one-direction-at-a-time pursuit was clearly worse
(log p -1.10) and is dropped. Cost 1.4-1.5 s/fact vs 0.5 for z*.

### E11 — M4 gated key neurons (writer test with oracle z*) — tuning on tune split
Files: results/E11_Composite_gate_*.json, E11_Composite_joint_*.json, diag_gate_L6_*.json.
Diagnostics first (how hard does a fact's filter fire, layer 6): own key 1.00, paraphrases median 0.94-0.96,
ordinary text max ~0.13 per neuron (95th pct 0.42). Ridge is irrelevant at this layer. The problem case is a
name that shares the last token: median response **0.92**, indistinguishable from a paraphrase. That is also
why MEMIT bleeds onto those names ~100% of the time (per-relation diag: nbr_b 0.5-1.0).
- Plain filter, thresholds 0.5/0.7: eff 0.82, para 0.73, bleed 0.45. Efficacy is low because facts whose
  subjects end in the same word (company suffixes, device kinds) fire each other's neurons.
- Five-layer version: worse (0.73 / 0.63). Dropped; one layer it is.
- Zero constraints on made-up near-miss names (one subject token swapped): bleed 0.095, eff 0.98, para 0.78.
- Joint solve (every filter also nulls every other stored fact, one linear system): a frontier between
  paraphrase and bleed. th 0.45/0.65: para 0.81 / bleed 0.23. th 0.35/0.55, eps 0.1: para 0.856 / bleed 0.39 /
  KL 0.0000 / QA 1.00 / real-nbr 0.994. th 0.25/0.45: para 0.87 / bleed 0.46.
  MEMIT on the same facts: para 0.826 / bleed 0.52 / KL 0.005 / QA 0.97.
vs prediction: locality ~0 (right). "Indirect paraphrases under-fire" (WRONG: indirect 0.96, it is the direct
rewordings that are weaker). "Same-last-token names fire the gate" (right, and fixable with nulls).
Picked th 0.35/0.55, eps 0.1, 8 near-misses per fact for the full runs.

### E13 (tune) — FD values + gated neurons: the forward-only candidate
File: results/E13_Composite_fd_gate.json. tune N=100: eff 0.95 / para 0.80 / hop 0.47 / KL 0.0000 / QA 1.00 /
real-nbr 0.994 / bleed 0.365. Tune score 0.741 vs MEMIT's 0.725. No backprop anywhere, no pretrained weight touched.

### E11 — gated key neurons, oracle z* values: full protocol (results/E11_Composite.json)
| | eff | para | hop | loc KL | QA | real nbr | name bleed | ppl | ARC |
|---|---|---|---|---|---|---|---|---|---|
| N=1 | 1.000 | 0.780 | 0.500 | 0.0000 | 1.000 | 0.994 | 0.450 | 26.17 | 0.475 |
| N=10 | 1.000 | 0.820 | 0.567 | 0.0000 | 1.000 | 0.989 | 0.367 | 26.17 | 0.475 |
| N=100 | 1.000 | 0.810 | 0.450 | 0.0002 | 1.000 | 0.994 | 0.330 | 26.17 | 0.475 |
| N=800 | 0.999 | 0.835 | 0.420 | 0.0012 | 0.971 | 0.994 | 0.314 | 26.24 | 0.480 |
| sequential 10x80 | 0.990 | 0.818 | 0.419 | 0.0006 | 0.971 | 0.983 | 0.320 | 26.21 | 0.480 |
Scale score 0.751, sequential 0.742, batch-1 efficacy flat at 0.99 through all ten batches. Same values as
MEMIT, different writer: paraphrase at 800 is 0.835 instead of 0.666, locality KL 40x lower. The writer is
the reason MEMIT/AlphaEdit decay with scale, not the values. Peak VRAM 4566MB.

### E13 — FD values + gated key neurons ("Kibble"): full protocol (results/E13_Composite.json)
Plan: the forward-only candidate, hyperparameters frozen from the tune split (L=6, read layers 9-13,
graft start 0.75|h|, 6 FD rounds, cap 1.5|h|; gate th 0.35/0.55, 8 near-misses per fact, eps 0.1).
| | eff | para | hop | loc KL | QA | real nbr | name bleed | ppl | ARC |
|---|---|---|---|---|---|---|---|---|---|
| N=1 | 1.000 | 0.760 | 0.600 | 0.0000 | 1.000 | 0.993 | 0.450 | 26.17 | 0.475 |
| N=10 | 1.000 | 0.807 | 0.667 | 0.0000 | 1.000 | 0.987 | 0.367 | 26.17 | 0.475 |
| N=100 | 0.970 | 0.796 | 0.580 | 0.0000 | 1.000 | 0.994 | 0.300 | 26.17 | 0.475 |
| N=800 | 0.954 | 0.809 | 0.526 | 0.0149 | 1.000 | 0.983 | 0.299 | 26.35 | 0.475 |
| sequential 10x80 | 0.941 | 0.790 | 0.499 | 0.0068 | 1.000 | 0.972 | 0.299 | 26.26 | 0.475 |
- **Scale score 0.763 (AlphaEdit 0.655, MEMIT 0.650). Sequential score 0.743 (AlphaEdit 0.650, MEMIT 0.439).**
  Batch-1 efficacy over the ten batches: 0.97, 0.97, 0.97, 0.97, 0.99 ... 0.99 (it goes *up*: every re-solve
  nulls the new arrivals too).
- Locality against the pre-registered bar: KL 0.015 / 0.007 (bar 0.049 / 0.039), QA 1.000 (bar 0.926),
  real-nbr 0.983 / 0.972 (bar 0.816 / 0.833), ppl 26.35 / 26.26 (bar 27.79 / 27.44), bleed 0.299 (bar 0.42).
  All passed. **Stop condition met at experiment 13.**
- It loses to MEMIT on raw efficacy (0.954 vs 0.974 at 800) and on paraphrase at N=1 (0.76 vs 0.84).
  It wins paraphrase from N=100 up because it doesn't decay, and multi-hop everywhere (the graft).
- Cost: values 1.53 s/fact forward-only (MEMIT's z*: 0.5 s/fact with backprop), writing 800 facts 61 s
  (MEMIT 135 s). Peak VRAM 4566MB. Search ends at mean log p(answer) = -0.13; 96% of facts above p=0.5.
- Too-good check: the editor only sees values.req() fields; every knob was set on the tune split; phase 4
  un-edit test checks the model goes back to bit-identical; held-out facts are still untouched.
- Still below fact-in-context with a perfect retriever (0.873) on the headline score, because of multi-hop
  (0.53 vs 0.86). It does beat it on near-miss names (0.30 vs 0.72 bleed) and needs no retriever.
M5 (hypernetwork) and M6 (spread vs concentrated, for the least-squares writer) were proposed but not run:
the stop condition landed first. The one data point on M6: the gated writer spread over 5 layers was worse
than on 1 (tune eff 0.73 vs 0.82, same thresholds).

## Phase 4 — attacking the winner (Kibble, E13 config) with AlphaEdit/MEMIT alongside

### E14 — adversarial paraphrases (results/E14_adv_Composite.json, E14_adv_AlphaEdit.json)
Plan: edit all 800 eval facts, then hit 200 of them with prompts the judge doesn't have. Exact match:
| attack | Kibble | AlphaEdit |
|---|---|---|
| plain direct prompt | 0.940 | 0.965 |
| 60 tokens of unrelated wikitext first | 0.915 | 0.835 |
| a different edited fact stated first | 0.845 | 0.685 |
| "many people wrongly believe the answer is X" first | 0.930 | 0.915 |
| two edited subjects named, then asked about one | 0.745 | 0.675 |
| name lower-cased | 0.760 | 0.760 |
| typo in the name (two letters swapped) | 0.510 | 0.570 |
| judge's worst single paraphrase template | 0.640 | 0.585 |
| answers with the *other* stated fact's answer (hijack, lower is better) | 0.085 | 0.175 |
| fires on the last word of the name alone (bleed, lower is better) | 0.397 | 0.603 |
Worst case for Kibble: **0.51 with a typo in the name** (0.64 if you only count well-formed prompts, the
possessive "X's city of birth is" template). It's a key-addressed memory: break the key, lose the fact.
AlphaEdit degrades the same way and is a bit more typo-tolerant (0.57), at the price of firing on the bare
surname 60% of the time.

### E15 — contradicting facts the model already knows (results/E15_contra_*.json)
Plan: take 56 real facts pythia gets right (8 per relation), overwrite each with a different answer.
| | Kibble | AlphaEdit |
|---|---|---|
| new answer, direct | 0.964 | 1.000 |
| old answer still given | 0.000 | 0.000 |
| new answer, paraphrases | 0.829 | 0.779 |
| multi-hop with the new answer | 0.321 | 0.482 |
| 124 untouched real facts still right | 0.935 | 0.879 |
| generic QA still right | 1.000 | 0.971 |
Overwriting works. The multi-hop advantage disappears on real entities (0.32 vs 0.48): the graft has to fight
what the model already believes about a famous subject, and loses more often than a gradient-found value.

### E16 — edit, then un-edit (results/E16_unedit_Composite.json, E16_unedit_MEMIT.json)
- Kibble, 800 facts in, then block removed: sha256 over every parameter identical, max |log-prob diff| on 100
  prompts = 0.0. **Bit-for-bit reversible**, because no pretrained weight is ever written (fingerprint is
  unchanged even while edited).
- Forgetting ONE fact (drop its slot, re-solve addresses): that fact 0.00 on direct + paraphrases, the other
  199 checked stay at 0.940. Per-fact deletion works.
- MEMIT kept as a side delta is equally reversible. Merged into the fp16 weights and subtracted again (how it
  normally ships): 14.7M of 83.9M entries (17.5%) do not come back. Not reversible once merged.

### E17 — 4-bit quantisation after editing (results/E17_quant_Composite.json, E17_quant_AlphaEdit.json)
Plan: edit 800 facts, then snap every attention/mlp Linear weight (and Kibble's memory block; for AlphaEdit
the update is merged into the weight first) to simulated bitsandbytes NF4 (blocks of 64, 16 levels).
Embeddings stay fp16. KL is measured against the fp16 unedited model, so the "no edit" row is the price of
quantisation alone.
| N=800 | eff | para | hop | loc KL | QA | real nbr | ppl |
|---|---|---|---|---|---|---|---|
| 4-bit, no edit | 0.004 | 0.002 | 0.006 | 0.077 | 0.912 | 0.906 | 27.92 |
| Kibble fp16 | 0.954 | 0.809 | 0.526 | 0.015 | 1.000 | 0.983 | 26.35 |
| Kibble 4-bit | 0.943 | 0.755 | 0.542 | 0.094 | 0.926 | 0.911 | (see json) |
| AlphaEdit fp16 | 0.969 | 0.686 | 0.309 | 0.057 | 0.897 | 0.806 | 27.88 |
| AlphaEdit 4-bit | 0.934 | 0.599 | 0.306 | 0.142 | 0.824 | 0.722 | 30.13 |
Kibble survives: -0.011 efficacy, -0.054 paraphrase, and its locality lands on top of the unedited 4-bit
model's (QA 0.926 vs 0.912, real-nbr 0.911 vs 0.906). AlphaEdit loses more (-0.035 / -0.087) and its locality
damage stacks on top of quantisation's. Worst case for Kibble here: paraphrase 0.755.

### E18 — three fresh seeds (results/E18_Composite_seed1..3.json)
Plan: a seed reshuffles the order of the 800 eval facts (so different batches, different N-subsets) and
reseeds the near-miss sampling. Values are deterministic per fact, so they're reused from cache.
| | eff | para | hop | loc KL | QA | real nbr | bleed | scale / seq score | batch-1 @10 |
|---|---|---|---|---|---|---|---|---|---|
| N=800 seed 0 (E13) | 0.954 | 0.809 | 0.526 | 0.015 | 1.000 | 0.983 | 0.299 | 0.763 | |
| N=800 seed 1 | 0.953 | 0.805 | 0.525 | 0.004 | 1.000 | 0.983 | 0.302 | 0.761 | |
| N=800 seed 2 | 0.956 | 0.809 | 0.516 | 0.004 | 0.985 | 0.983 | 0.296 | 0.760 | |
| N=800 seed 3 | 0.951 | 0.803 | 0.522 | 0.004 | 1.000 | 0.983 | 0.298 | 0.759 | |
| seq seed 0 (E13) | 0.941 | 0.790 | 0.499 | 0.007 | 1.000 | 0.972 | 0.299 | 0.743 | 0.988 |
| seq seed 1 | 0.941 | 0.789 | 0.506 | 0.003 | 1.000 | 0.989 | 0.299 | 0.745 | 0.925 |
| seq seed 2 | 0.936 | 0.786 | 0.505 | 0.004 | 1.000 | 0.989 | 0.294 | 0.742 | 0.975 |
| seq seed 3 | 0.938 | 0.786 | 0.514 | 0.001 | 1.000 | 0.994 | 0.295 | 0.746 | 0.963 |
Worst seed: scale 0.759, sequential 0.742, batch-1 retention 0.925. Spread is under 0.005 on the scores.
The loc KL of 0.015 in E13 was the unlucky one (others 0.004): a few near-miss draws make a neuron twitchy.
Caveat: the covariance sample and the model itself were not reseeded.

### E19 — ablations (results/E19_Composite_abl_*.json, tune split N=100; E19_Composite_fd_ls.json)
| tune N=100 | eff | para | hop | bleed | note |
|---|---|---|---|---|---|
| full Kibble | 0.95 | 0.80 | 0.47 | 0.365 | |
| no graft (head directions only, start from zero) | 0.87 | 0.61 | 0.51 | 0.320 | search ends at log p -0.44 instead of -0.11 |
| graft only (no FD search) | 0.62 | 0.37 | 0.77 | 0.250 | |
| no near-miss nulls | 0.95 | 0.81 | 0.48 | 0.500 | |
- I had credited the graft with the multi-hop. Wrong, or at least incomplete: the head-direction search
  without any graft gets 0.51. What the graft start buys is a better optimum for the direct question
  (para 0.61 -> 0.80). Graft alone is still the multi-hop champion (0.77) and the worst editor.
- The near-miss nulls cost nothing on paraphrase and cut name bleed from 0.50 to 0.37.

- Writer ablation, full protocol (E19_Composite_fd_ls.json): the same FD values pushed through MEMIT's
  least-squares writer instead of the gate. N=800: eff 0.828 / para 0.554 / hop 0.316 / KL 0.124 / QA 0.824 /
  real-nbr 0.667 / ppl 29.3. Sequential: 0.505 / 0.288 / 0.194, batch-1 at 0.47. That is *worse* than MEMIT
  with its own z* (0.974 / 0.666 at 800). So the FD values are not as clean as gradient-found ones (they are
  twice as large, 1.47|h| vs a 0.75|h| clamp) and a linear writer smears that everywhere. The gate is what
  makes forward-only values usable. Put together with E11: the writer explains the scaling, not the values.

### E20 — where does it run out of room? (results/E20_stress_Composite.json)
Plan: the judge's scale axis stops at 1000 facts and Kibble is flat there, so crowd the block instead: 800
real facts plus n decoy slots keyed on fresh made-up subjects with zero values, judge scores the 800.
| extra slots | total slots | eff | para | hop | loc KL | QA | peak VRAM |
|---|---|---|---|---|---|---|---|
| 0 | 800 | 0.954 | 0.809 | 0.526 | 0.015 | 1.000 | 4548MB |
| 1000 | 1800 | 0.954 | 0.810 | 0.530 | 0.012 | 1.000 | 5062MB |
| 2000 | 2800 | 0.954 | 0.812 | 0.527 | 0.010 | 1.000 | 5587MB |
| 3000 | 3800 | 0.954 | 0.813 | 0.530 | 0.010 | 1.000 | 6139MB |
| 4000 | 4800 | OOM | | | | | 7GB cap |
The facts don't interfere; the *solver* dies. One joint system over every stored key and near-miss is
quadratic in memory, and at ~10k constraint rows (4800 slots + 6400 near-misses) it no longer fits beside the
model in 7GB. So the collapse point on this box is a memory wall between 3800 and 4800 slots, not a quality
slope. (Housekeeping: the run ended with a clean OOM while I was in the middle of killing it for looking
stuck; I overwrote its json with a reconstruction from the log and then corrected the last row. Numbers for
0-3000 are identical to what the run printed.)
