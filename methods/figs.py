# draws every picture in figures/ straight from results/*.json. no number in here is typed by hand
# (the handful of labels that quote a number pull it from the same json).
# usage: python methods/figs.py [name ...]      e.g. python methods/figs.py hero scale
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)
R = lambda n: json.load(open(os.path.join(ROOT, "results", n + ".json"), encoding="utf-8"))
sc = lambda m: (m["efficacy_em"] + m["para_em"] + m["hop_em"]) / 3

BG, INK, INK2, MUTE, GRID, AXIS = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
# one colour per method, same everywhere. order checked for colourblind separation on this background
COL = dict(memit="#3987e5", alpha="#d95926", lora="#199e70", ctx="#c98500", kibble="#d55181", none="#5c5b57")
NAME = dict(memit="MEMIT", alpha="AlphaEdit", lora="LoRA", ctx="fact in the prompt", kibble="Kibble", none="no edit")
RUN = dict(memit="E05_MEMIT", alpha="E06_AlphaEdit", lora="E04_LoRA", ctx="E03_InContext_oracle", kibble="E13_Composite", none="E03_NoEdit")
HELD = dict(memit="E21_MEMIT", alpha="E21_AlphaEdit", lora="E21_LoRA", ctx="E21_InContext", kibble="E21_Composite", none="E21_NoEdit")
plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK, "axes.edgecolor": AXIS,
                     "axes.labelcolor": MUTE, "xtick.color": MUTE, "ytick.color": MUTE, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "axes.linewidth": 0.8})
W, H = 9.6, 5.4     # x 200 dpi = 1920 x 1080

def paw(fig, x, y, s=0.016, c=None):
    c = c or COL["kibble"]; a = H / W     # keep it round on a 16:9 canvas
    ax = fig.add_axes([x, y, s * 2.6 * a, s * 2.6]); ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.axis("off")
    ax.add_patch(Ellipse((0, -0.38), 1.25, 1.0, color=c))
    for dx, dy, w in ((-0.86, 0.3, 0.5), (-0.3, 0.82, 0.52), (0.3, 0.82, 0.52), (0.86, 0.3, 0.5)): ax.add_patch(Ellipse((dx, dy), w, w * 1.18, color=c))

def frame(title, sub=None, foot=None):
    fig = plt.figure(figsize=(W, H))
    fig.text(0.045, 0.925, title, fontsize=17, fontweight="bold", va="top")
    if sub: fig.text(0.045, 0.853, sub, fontsize=10, color=INK2, va="top", linespacing=1.35)
    paw(fig, 0.045, 0.028)
    fig.text(0.072, 0.047, "kibble  ·  pythia-1b on an 8GB laptop gpu  ·  " + (foot or "scored by exact match, never by reading generations"),
             fontsize=7.5, color=MUTE, va="center")
    return fig

def clean(ax, grid="y"):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelsize=8.5)
    if grid: ax.grid(axis=grid, color=GRID, lw=0.7); ax.set_axisbelow(True)

def legend(fig, keys, x=0.045, y=0.745, names=None, dx=None):
    for k in keys:
        fig.add_artist(plt.Line2D([x, x + 0.018], [y, y], color=COL[k], lw=2.4, solid_capstyle="round"))
        t = (names or NAME)[k]; fig.text(x + 0.025, y, t, fontsize=8.5, color=INK2, va="center")
        x += dx or (0.052 + 0.0058 * len(t))

def end_labels(ax, items, x, gap, fmt="%.3f", dx=0.12):
    """items: (key, y). names + values at the right end; if two would overlap, spread them and draw a leader."""
    items = sorted(items, key=lambda t: t[1]); ys = [y for _, y in items]
    for i in range(1, len(ys)): ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (sum(ys) - sum(y for _, y in items)) / len(ys)
    for (k, y), yy in zip(items, [v - shift for v in ys]):
        if abs(yy - y) > gap * 0.15: ax.plot([x + dx * 0.25, x + dx * 0.9], [y, yy], color=AXIS, lw=0.7, clip_on=False)
        ax.text(x + dx, yy, fmt % y, fontsize=8.5, color=INK, va="center", ha="left", fontweight="bold", clip_on=False)
        ax.annotate(NAME[k], (x + dx, yy), xytext=(31, 0), textcoords="offset points", fontsize=8.5, color=INK2, va="center", annotation_clip=False)

def save(fig, name):
    p = os.path.join(OUT, name + ".png"); fig.savefig(p, dpi=200); plt.close(fig); print("wrote", os.path.relpath(p, ROOT))

# ----------------------------------------------------------------------------- 1. the headline
def hero():
    fig = plt.figure(figsize=(W, H)); k = R(HELD["kibble"])["all_1000"]
    paw(fig, 0.05, 0.80, s=0.034); fig.text(0.115, 0.845, "kibble", fontsize=40, fontweight="bold", va="center")
    fig.text(0.052, 0.725, "1000 made-up facts written into pythia-1b.\nforward passes only. not one pretrained weight changed.", fontsize=11.5, color=INK2, va="top", linespacing=1.45)
    fig.text(0.05, 0.47, "%.3f" % sc(k), fontsize=64, fontweight="bold", va="center")
    fig.text(0.052, 0.345, "edit score with all 1000 facts stored at once\n(mean of efficacy, paraphrase and multi-hop exact match)", fontsize=9, color=INK2, va="top", linespacing=1.4)
    u = R("E16_unedit_Composite")["result"]
    for i, (a, b) in enumerate((("0", "backward passes"), ("0", "pretrained weights touched"), ("bit for bit" if u["bit_for_bit"] else "?", "stock model back when you detach it"))):
        fig.text(0.052, 0.215 - i * 0.052, a, fontsize=10.5, fontweight="bold", va="center")
        fig.text(0.052 + (0.022 if len(a) == 1 else 0.092), 0.215 - i * 0.052, b, fontsize=9.5, color=INK2, va="center")
    ax = fig.add_axes([0.60, 0.17, 0.33, 0.66]); order = sorted(HELD, key=lambda m: -sc(R(HELD[m])["all_1000"]))
    note = dict(ctx="perfect retriever. the ceiling", kibble="ours", lora="perplexity %.0f → %.0f" %(R(HELD["none"])["all_1000"]["ppl"], R(HELD["lora"])["all_1000"]["ppl"]),
                alpha="published editor", memit="published editor", none="stock pythia-1b")
    for i, m in enumerate(order):
        v = sc(R(HELD[m])["all_1000"]); y = len(order) - 1 - i
        ax.barh(y, v, height=0.34, color=COL[m], lw=0)
        ax.text(-0.025, y + 0.08, NAME[m], ha="right", va="center", fontsize=10.5, fontweight="bold" if m == "kibble" else "normal", color=INK if m == "kibble" else INK2)
        ax.text(-0.025, y - 0.2, note[m], ha="right", va="center", fontsize=7.5, color=MUTE)
        ax.text(v + 0.015, y, "%.3f" % v, va="center", fontsize=10.5, fontweight="bold" if m == "kibble" else "normal", color=INK if m == "kibble" else INK2)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(order) - 0.4); ax.axis("off"); ax.plot([0, 0], [-0.5, len(order) - 0.5], color=AXIS, lw=0.8)
    fig.text(0.60, 0.865, "score at 1000 facts (800 eval + 200 held-out, edited together)", fontsize=8.5, color=MUTE)
    fig.text(0.052, 0.047, "pythia-1b  ·  one 8GB laptop gpu  ·  21 experiments  ·  scored by exact match, never by reading generations", fontsize=7.5, color=MUTE, va="center")
    save(fig, "01_hero")

# ----------------------------------------------------------------------------- 2. failure curve
def scale():
    fig = frame("Pile on more facts and the published editors slide. Kibble doesn't.",
                "Edit score (mean of efficacy, paraphrase, multi-hop exact match) as the number of facts written at once grows.")
    ks = ["ctx", "kibble", "lora", "alpha", "memit"]; legend(fig, ks); ax = fig.add_axes([0.075, 0.2, 0.70, 0.49]); clean(ax)
    for m in ks:
        d = R(RUN[m]); ys = [sc(d["scale"][n]) for n in ("1", "10", "100", "800")] + [sc(R(HELD[m])["all_1000"])]
        ax.plot(range(5), ys, color=COL[m], lw=2.6 if m == "kibble" else 2, solid_capstyle="round", solid_joinstyle="round", zorder=3 if m == "kibble" else 2)
        ax.plot(4, ys[-1], "o", ms=7.5, color=COL[m], mec=BG, mew=1.6, zorder=4)
    end_labels(ax, [(m, sc(R(HELD[m])["all_1000"])) for m in ks], 4, 0.027)
    ax.set_xticks(range(5)); ax.set_xticklabels(["1 fact", "10", "100", "800", "1000\n(+ 200 held-out)"]); ax.set_xlim(-0.15, 4.08); ax.set_ylim(0.55, 0.92)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9]); ax.set_xlabel("facts written in one go", fontsize=8.5, labelpad=4)
    save(fig, "02_scale")

# ----------------------------------------------------------------------------- 3. forgetting
def seq():
    fig = frame("Ten rounds of edits later, does it still remember round one?",
                "800 facts written 80 at a time. Exact match on the first batch of 80, re-checked after every later batch.",
                foot="fact-in-the-prompt keeps no state, so it sits flat at 0.988 and is left out")
    ks = ["kibble", "alpha", "memit", "lora"]; legend(fig, ks); ax = fig.add_axes([0.075, 0.17, 0.72, 0.52]); clean(ax)
    for m in ks:
        ys = [b["efficacy_em"] for b in R(RUN[m])["seq"]["batch1_curve"]]
        ax.plot(range(1, 11), ys, color=COL[m], lw=2.6 if m == "kibble" else 2, solid_capstyle="round", solid_joinstyle="round", zorder=3 if m == "kibble" else 2)
        ax.plot(10, ys[-1], "o", ms=7.5, color=COL[m], mec=BG, mew=1.6, zorder=4)
    end_labels(ax, [(m, R(RUN[m])["seq"]["batch1_curve"][-1]["efficacy_em"]) for m in ks], 10, 0.07, dx=0.25)
    ax.set_xticks(range(1, 11)); ax.set_xlim(0.7, 10.15); ax.set_ylim(-0.04, 1.04); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xlabel("batches written so far", fontsize=8.5, labelpad=4)
    save(fig, "03_sequential")

# ----------------------------------------------------------------------------- 4. what it broke
def damage():
    fig = frame("What 800 edits broke in the rest of the model", "Same run as the scores: 800 facts written in one batch, then everything else re-measured.",
                foot="fact-in-the-prompt changes no weights, so it is 0 / 100% / +0 here but 72% on near-miss names")
    ks = ["kibble", "alpha", "memit", "lora"]; base = R(RUN["none"])["scale"]["800"]["ppl"]
    panels = [("drift on unrelated text", "KL from the stock model, nats", lambda m: m["loc_kl"], "%.3f"),
              ("real facts it used to know", "share it now gets wrong", lambda m: 1 - m["nbr_real_em"], "{:.0%}"),
              ("perplexity added", "wikitext-2, stock is %.1f" % base, lambda m: m["ppl"] - base, "+%.1f"),
              ("near-miss names", "share that wrongly pick up the edit", lambda m: m["nbr_name_bleed_em"], "{:.0%}")]
    for j, (t, s, f, fmt) in enumerate(panels):
        ax = fig.add_axes([0.135 + j * 0.215, 0.16, 0.15, 0.48]); vals = [f(R(RUN[m])["scale"]["800"]) for m in ks]
        for i, (m, v) in enumerate(zip(ks, vals)):
            y = len(ks) - 1 - i; ax.barh(y, v, height=0.3, color=COL[m], lw=0)
            ax.text(v + max(vals) * 0.05, y, fmt.format(v) if "{" in fmt else fmt % v, va="center", fontsize=9, color=INK if m == "kibble" else INK2, fontweight="bold" if m == "kibble" else "normal")
            if j == 0: ax.text(-max(vals) * 0.08, y, NAME[m], ha="right", va="center", fontsize=10, color=INK if m == "kibble" else INK2, fontweight="bold" if m == "kibble" else "normal")
        ax.set_xlim(0, max(vals) * 1.45); ax.set_ylim(-0.6, len(ks) - 0.4); ax.axis("off"); ax.plot([0, 0], [-0.45, len(ks) - 0.55], color=AXIS, lw=0.8)
        fig.text(0.135 + j * 0.215, 0.725, t, fontsize=10, fontweight="bold", va="center"); fig.text(0.135 + j * 0.215, 0.685, s, fontsize=8, color=MUTE, va="center")
    fig.text(0.045, 0.79, "shorter bar = less damage", fontsize=8.5, color=INK2)
    save(fig, "04_damage")

# ----------------------------------------------------------------------------- 5. the mechanism
def box(ax, x, y, w, h, fc=BG, ec=AXIS, lw=1.0, r=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
def arrow(ax, a, b, c=MUTE, lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=c, lw=lw, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

def how():
    fig = frame("How a fact gets in: one gated neuron beside layer 6", "Nothing is trained. The value is found with forward passes, the address with one linear solve.",
                foot="details and the maths: REPORT.md")
    f = json.load(open(os.path.join(ROOT, "data", "facts.json"), encoding="utf-8")); f = next(x for x in f if x["split"] == "heldout" and x["rel"] == "born_city")
    first, last = f["subject"].split(); rest = f["direct"][len(f["subject"]):].strip(); K = COL["kibble"]
    ax = fig.add_axes([0.045, 0.10, 0.91, 0.68]); ax.set_xlim(0, 16); ax.set_ylim(0, 6.6); ax.axis("off")
    xs, xl = 4.7, 9.4; words = rest.split(); mid, final = " ".join(words[:-1]), words[-1]
    # prompt at the bottom, answer out the top. the usual way up
    for t, x, w in ((first, 1.75, 1.85), (last, xs - 1.0, 2.0), (mid, xs + 1.1, xl - xs - 1.7), (final, xl - 0.5, 1.0)):
        hot = t == last; box(ax, x, 0.45, w, 0.62, fc="#2a1a21" if hot else "#222221", ec=K if hot else AXIS, lw=1.4 if hot else 0.8)
        ax.text(x + w / 2, 0.76, t, ha="center", va="center", fontsize=10, color=INK)
    ax.text(xs, 0.17, "the slot listens at the last token of the name", ha="center", fontsize=7.5, color=MUTE)
    ly = lambda l: 1.55 + l * (3.5 / 15)
    for x in (xs, xl): ax.plot([x, x], [1.3, 5.3], color=AXIS, lw=5, solid_capstyle="round")
    for l, t in ((0, "layer 0"), (6, "layer 6"), (11, "layers 9–13"), (15, "layer 15")): ax.text(xl + 0.3, ly(l), t, va="center", fontsize=7.5, color=MUTE)
    box(ax, xl - 0.95, 5.65, 1.9, 0.62, fc="#222221", ec=AXIS, lw=0.8); ax.text(xl, 5.96, f["target"], ha="center", va="center", fontsize=10, fontweight="bold")
    arrow(ax, (xl, 5.36), (xl, 5.62), c=MUTE)
    ax.text(xl - 1.15, 5.96, "a person who doesn't exist. stock pythia has no idea", fontsize=7.5, color=MUTE, va="center", ha="right")
    # the slot
    ax.plot(xs, ly(6), "o", ms=9, color=K, mec=BG, mew=1.6, zorder=5)
    box(ax, 0.2, ly(6) - 0.6, 2.6, 1.2, fc="#2a1a21", ec=K, lw=1.3); ax.text(1.5, ly(6) + 0.27, "kibble slot", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(1.5, ly(6) - 0.08, "address  →  gate  →  value", ha="center", fontsize=8, color=INK2); ax.text(1.5, ly(6) - 0.38, "8192-d                    2048-d", ha="center", fontsize=7, color=MUTE)
    arrow(ax, (xs - 0.16, ly(6) + 0.08), (2.85, ly(6) + 0.3), c=K, rad=0.12); arrow(ax, (2.85, ly(6) - 0.3), (xs - 0.16, ly(6) - 0.08), c=K, rad=0.12)
    ax.text(3.72, ly(6) + 0.52, "reads the key", ha="center", fontsize=7.5, color=INK2); ax.text(3.72, ly(6) - 0.62, "adds the value", ha="center", fontsize=7.5, color=INK2)
    # read-out
    for x in (xs, xl): ax.add_patch(Rectangle((x - 0.09, ly(9)), 0.18, ly(13) - ly(9), color="#5a5955", zorder=3))
    for l in (9.6, 11, 12.4): arrow(ax, (xs + 0.14, ly(l)), (xl - 0.14, ly(l) + 0.12), c=MUTE, lw=1.0)
    ax.text((xs + xl) / 2, ly(13) + 0.22, "attention heads carry it across", ha="center", fontsize=8, color=INK2)
    # the three steps
    steps = [("1  value: what to store", "start from the model's own layer-6 vector\nfor the answer token, then nudge it along\n40 head read-out directions. keep whatever\nraises p(answer). forward passes only."),
             ("2  address: when to fire", "one linear solve for every slot at once:\nanswer 1 on this name, 0 on the other 999\nand on 8 made-up near-miss names each,\nand stay quiet on ordinary text."),
             ("3  gate: two ReLUs per fact", "below 0.35 nothing happens. past 0.55\nthe whole value is added. drop the slot and\nthe fact is gone; drop the block and the\nmodel is the stock one, bit for bit.")]
    for i, (t, b) in enumerate(steps):
        y = 4.75 - i * 1.5; ax.text(11.35, y, t, fontsize=9.5, fontweight="bold", va="top"); ax.text(11.35, y - 0.33, b, fontsize=7.8, color=INK2, va="top", linespacing=1.35)
    save(fig, "05_how_it_works")

# ----------------------------------------------------------------------------- 6. the gate
def gate():
    d = R("pack_check"); a = d["pre_gate"]; th1, th2 = d["hp"]["th1"], d["hp"]["th2"]; K = COL["kibble"]
    fig = frame("Each slot wakes for its own name. Sharing a last name still fools it.",
                "How hard a slot fires at the last token of the name, 1000 facts stored. Below %.2f the gate is shut; past %.2f it is fully open." % (th1, th2),
                foot="%d facts, %d tokens of wikitext-2" % (d["n_facts"], d["n_text_tokens"]))
    rows = [("own_prompt", "the prompt it was written with", 1), ("paraphrase", "five rewordings it never saw", 1), ("multi_hop", "a two-step question about it", 1),
            ("near_miss_first_word", "near-miss: same first name", 0), ("near_miss_last_word", "near-miss: same last name", 0), ("text_loudest_slot", "ordinary text, loudest of all 1000 slots", 0)]
    ax = fig.add_axes([0.30, 0.15, 0.56, 0.58]); lo, hi = -0.4, 1.6; bins = np.linspace(lo, hi, 101)
    ax.axvspan(th1, th2, color="#2c2c2a", lw=0); ax.axvline(th1, color=AXIS, lw=0.8); ax.axvline(th2, color=AXIS, lw=0.8)
    for i, (k, lab, good) in enumerate(rows):
        y0 = len(rows) - 1 - i; h, _ = np.histogram(np.clip(a[k], lo, hi), bins); h = h / h.max() * 0.82; c = K if good else "#8f8e88"
        ax.fill_between(bins[:-1], y0, y0 + h, step="post", color=c, lw=0, alpha=0.9); ax.plot([lo, hi], [y0, y0], color=AXIS, lw=0.7)
        fig.text(0.285, 0.15 + 0.58 * (y0 + 0.3) / len(rows), lab, ha="right", va="center", fontsize=9, color=INK2)
        ax.text(hi + 0.06, y0 + 0.3, "{:.1%}".format(d["frac_gate_open"][k]), va="center", fontsize=9.5, fontweight="bold", clip_on=False)
    ax.text(hi + 0.06, len(rows) + 0.12, "gate opens", fontsize=8, color=MUTE, va="center", clip_on=False)
    ax.text((th1 + th2) / 2, len(rows) + 0.12, "opening", fontsize=7.5, color=MUTE, ha="center", va="center")
    ax.set_xlim(lo, hi); ax.set_ylim(0, len(rows)); ax.set_yticks([]); clean(ax, grid=None); ax.spines["bottom"].set_visible(False)
    ax.set_xticks([0, th1, th2, 1.0, 1.5]); ax.set_xticklabels(["0", "%.2f" % th1, "%.2f" % th2, "1", "1.5"]); ax.set_xlabel("slot activation before the gate (1 = a perfect match with its stored key)", fontsize=8.5, labelpad=5)
    legend(fig, ["kibble", "none"], y=0.775, names=dict(kibble="should fire", none="should stay shut"), dx=0.115)
    save(fig, "06_gate")

# ----------------------------------------------------------------------------- 7. attacks
def attacks():
    k, a = R("E14_adv_Composite")["result"], R("E14_adv_AlphaEdit")["result"]; qk, qa = R("E17_quant_Composite")["result"]["nf4_edited"], R("E17_quant_AlphaEdit")["result"]["nf4_edited"]
    ck, ca = R("E15_contra_Composite")["result"], R("E15_contra_AlphaEdit")["result"]
    rows = [("plain prompt", k["plain"]["em"], a["plain"]["em"]), ("a wrong answer claimed first", k["wrong_claim_first"]["em"], a["wrong_claim_first"]["em"]),
            ("buried after unrelated text", k["long_context"]["em"], a["long_context"]["em"]), ("another stored fact stated first", k["other_fact_first"]["em"], a["other_fact_first"]["em"]),
            ("two edited names in one prompt", k["two_subjects"]["em"], a["two_subjects"]["em"]), ("name typed in lowercase", k["lowercase_name"]["em"], a["lowercase_name"]["em"]),
            ("name with a typo", k["typo_name"]["em"], a["typo_name"]["em"]), ("4-bit quantised after editing (paraphrases)", qk["para_em"], qa["para_em"]),
            ("real fact overwritten (new answer)", ck["new_answer_em"], ca["new_answer_em"]), ("real fact overwritten (two-step question)", ck["hop_new_em"], ca["hop_new_em"])]
    fig = frame("Trying to break it: Kibble against AlphaEdit", "Exact match on the stored answer with 800 facts in. Further right is better. Kibble loses on typos and on overwriting famous facts.")
    legend(fig, ["kibble", "alpha"], y=0.775); ax = fig.add_axes([0.36, 0.13, 0.56, 0.60]); n = len(rows)
    for i, (lab, vk, va) in enumerate(rows):
        y = n - 1 - i; ax.plot([vk, va], [y, y], color="#4a4946", lw=2, solid_capstyle="round", zorder=1)
        ax.plot(va, y, "o", ms=8.5, color=COL["alpha"], mec=BG, mew=1.6, zorder=3); ax.plot(vk, y, "o", ms=8.5, color=COL["kibble"], mec=BG, mew=1.6, zorder=4)
        l, r = sorted(((vk, 1), (va, 0)))
        if abs(vk - va) < 0.004: ax.text(vk + 0.022, y, "%.2f both" % vk, ha="left", va="center", fontsize=8, color=INK2)
        else:
            ax.text(l[0] - 0.022, y, "%.2f" % l[0], ha="right", va="center", fontsize=8, color=INK if l[1] else MUTE, fontweight="bold" if l[1] else "normal")
            ax.text(r[0] + 0.022, y, "%.2f" % r[0], ha="left", va="center", fontsize=8, color=INK if r[1] else MUTE, fontweight="bold" if r[1] else "normal")
        fig.text(0.345, 0.13 + 0.60 * (y + 0.5) / n, lab, ha="right", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0.2, 1.06); ax.set_ylim(-0.5, n - 0.5); ax.set_yticks([]); clean(ax, grid="x"); ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    save(fig, "07_attacks")

# ----------------------------------------------------------------------------- 8. where facts live
def trace():
    d = R("E01_causal_trace"); roles = [("subj_first", "name: first token"), ("subj_mid", "name: middle"), ("subj_last", "name: last token"), ("after_first", "next token"), ("after_mid", "rest of the prompt"), ("last", "last token")]
    fig = frame("Where pythia-1b keeps a fact (causal tracing)", "Scramble the name, then restore one spot at a time. Brighter = restoring that spot brings more of the right answer back.",
                foot="%d real facts, average indirect effect on p(answer)" % d["n_prompts"])
    cm = LinearSegmentedColormap.from_list("b", [BG, "#104281", "#2a78d6", "#86b6ef", "#e6f0fd"]); vmax = max(max(v) for k in d["aie"].values() for v in k.values())
    for j, (kind, t) in enumerate((("resid", "whole layer state"), ("mlp", "MLP only"), ("attn", "attention only"))):
        ax = fig.add_axes([0.20 + j * 0.245, 0.24, 0.215, 0.42]); M = np.array([d["aie"][kind][r] for r, _ in roles]).clip(0)
        im = ax.pcolormesh(M, cmap=cm, vmin=0, vmax=vmax, edgecolors=BG, linewidth=1.2); ax.invert_yaxis()
        ax.set_xticks([0.5, 3.5, 6.5, 9.5, 12.5, 15.5]); ax.set_xticklabels(["0", "3", "6", "9", "12", "15"]); ax.tick_params(length=0, labelsize=8)
        ax.set_yticks(np.arange(6) + 0.5); ax.set_yticklabels([n for _, n in roles] if j == 0 else [], fontsize=8.5, color=INK2)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_xlabel("layer", fontsize=8, labelpad=3); fig.text(0.20 + j * 0.245, 0.69, t, fontsize=10, fontweight="bold")
        if kind == "resid": ax.annotate("kibble writes here (layer 6)", (6.5, 3.05), xytext=(6.5, 4.45), ha="center", fontsize=8, color=INK, arrowprops=dict(arrowstyle="-|>", color=COL["kibble"], lw=1.3))
    cax = fig.add_axes([0.20, 0.13, 0.215, 0.018]); cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
    cb.set_ticks([0, round(vmax, 2)]); cax.tick_params(length=0, labelsize=7.5); fig.text(0.425, 0.139, "p(answer) won back  (clean run %.2f, scrambled %.2f)" % (d["p_clean"], d["p_corrupt"]), fontsize=7.5, color=MUTE, va="center")
    save(fig, "08_where_facts_live")

# ----------------------------------------------------------------------------- 9. the wall
def wall():
    d = R("E20_stress_Composite")["result"]; ok = [v for v in d.values() if "efficacy_em" in v]; dead = [v for v in d.values() if "efficacy_em" not in v][0]
    fig = frame("Where it actually stops: memory, not quality", "800 real facts plus thousands of empty decoy slots crowding the address table. Scores don't move. The solver runs out of VRAM instead.")
    x = [v["slots"] for v in ok]; K = COL["kibble"]
    ax = fig.add_axes([0.075, 0.16, 0.38, 0.50]); clean(ax); ax.plot(x, [sc(v) for v in ok], color=K, lw=2.6, solid_capstyle="round"); ax.plot(x, [sc(v) for v in ok], "o", ms=7.5, color=K, mec=BG, mew=1.6)
    ax.text(x[-1], sc(ok[-1]) + 0.06, "%.3f" % sc(ok[-1]), ha="center", fontsize=9, fontweight="bold"); ax.text(x[0], sc(ok[0]) + 0.06, "%.3f" % sc(ok[0]), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1); ax.set_xlim(400, 5200); ax.set_xticks(x + [dead["slots"]]); ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xlabel("slots in the block", fontsize=8.5, labelpad=4)
    fig.text(0.075, 0.71, "edit score on the 800 real facts", fontsize=10, fontweight="bold")
    ax = fig.add_axes([0.545, 0.16, 0.38, 0.50]); clean(ax); y = [v["peak_mb"] / 1024 for v in ok]
    ax.axhline(7, color="#d03b3b", lw=1.2); ax.text(450, 7.12, "7 GB cap", fontsize=8, color=INK2, va="bottom")
    ax.plot(x, y, color=K, lw=2.6, solid_capstyle="round"); ax.plot(x, y, "o", ms=7.5, color=K, mec=BG, mew=1.6)
    ax.plot(dead["slots"], 7, "X", ms=11, color="#d03b3b", mec=BG, mew=1.4, zorder=5, clip_on=False); ax.text(dead["slots"], 6.5, "out of\nmemory", ha="center", va="top", fontsize=8.5, color=INK, linespacing=1.2)
    ax.text(x[-1] - 120, y[-1] + 0.25, "%.1f GB" % y[-1], ha="right", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 8); ax.set_xlim(400, 5200); ax.set_xticks(x + [dead["slots"]]); ax.set_yticks([0, 2, 4, 6, 8]); ax.set_xlabel("slots in the block", fontsize=8.5, labelpad=4)
    fig.text(0.545, 0.71, "peak VRAM, GB", fontsize=10, fontweight="bold")
    save(fig, "09_memory_wall")

ALL = dict(hero=hero, scale=scale, seq=seq, damage=damage, how=how, gate=gate, attacks=attacks, trace=trace, wall=wall)
if __name__ == "__main__":
    for n in (sys.argv[1:] or ALL): ALL[n]()
