"""All figures of the manuscript. Reads pilot/, marker_list/ and experiment/revision_*; writes figures/*.pdf.
Set PREVIEW=<folder> to also write PNG previews."""
import json, os
from collections import Counter
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(ROOT, "..", "pilot")
MARK = os.path.join(ROOT, "..", "marker_list")
EXP = os.path.join(ROOT, "..", "experiment")
OUT = os.path.join(ROOT, "figures")
PREVIEW = os.environ.get("PREVIEW")  # optional folder for PNG previews
os.makedirs(OUT, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
S1, S2 = "#2a78d6", "#eb6834"                                 # categorical slots 1-2
ORDINAL = ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]  # most -> least accessible
COL_W = 3.45  # inches, elsarticle 5p column width

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), metadata={"CreationDate": None})  # no timestamp: reproducible files
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=220)
    plt.close(fig)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def retention():
    R = json.load(open(os.path.join(PILOT, "author_histories.json")))
    roles = [("first", "First author"), ("corresponding", "Corresponding"), ("last", "Last author")]
    anyp, firstp, labels = [], [], []
    for key, name in roles:
        A = [r[key] for r in R if r.get(key)]
        anyp.append(100 * sum(a["w2015"] > 0 and a["w2020"] > 0 for a in A) / len(A))
        firstp.append(100 * sum(a["w2015_first"] > 0 and a["w2020_first"] > 0 for a in A) / len(A))
        labels.append(f"{name}\n(n = {len(A)})")
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    h = 0.36
    ys = range(len(roles))
    for off, vals, c, lab in ((-h / 2, anyp, S1, "Any position"), (h / 2, firstp, S2, "As first author")):
        bars = ax.barh([y + off for y in ys], vals, height=h - 0.04, color=c, label=lab)
        for b, v in zip(bars, vals):
            ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", color=INK, fontsize=7)
    ax.set_yticks(list(ys), labels); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("Authors publishing in both pre-LLM windows (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=7)
    fig.tight_layout(); save(fig, "fig_retention")


def fulltext():
    A = json.load(open(os.path.join(PILOT, "fulltext_status.json")))
    tiers = [("Machine-readable (PMC OA / preprint)", {"pmc_oa", "preprint"}), ("In PMC, not OA subset", {"pmc_other"}),
             ("Other free full text", {"other_oa"}), ("Abstract only", {"abstract_only"}),
             ("No abstract in OpenAlex", {"nothing"})]
    rows = []
    for lab, first in (("All papers", False), ("First-authored", True)):
        for wl, yrs in (("2014–16", range(2014, 2017)), ("2019–21", range(2019, 2022))):
            ws = [w for a in A.values() for w in a["works"] if w["year"] in yrs and (w["first"] or not first)]
            c = Counter(w["tier"] for w in ws)
            rows.append((f"{lab}, {wl}\n(n = {len(ws)})", [100 * sum(c[t] for t in s) / len(ws) for _, s in tiers]))
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    for i, (lab, vals) in enumerate(rows):
        left = 0
        for v, col in zip(vals, ORDINAL):
            ax.barh(i, v, left=left, height=0.62, color=col, edgecolor=SURF, linewidth=1)
            if v >= 8:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if col in ORDINAL[:3] else INK)
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("Share of papers (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ORDINAL]
    fig.legend(handles, [t for t, _ in tiers], frameon=False, fontsize=6.3, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, 0.0), handlelength=1.2, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.17, 1, 1)); save(fig, "fig_fulltext")


R = json.load(open(os.path.join(EXP, "revision_results.json")))
POP26 = json.load(open(os.path.join(MARK, "population_2026.json")))
S3 = "#7a7a7a"
FULL_W = 7.1


def marker_trend():
    """Held-out population MWI by year, with the 2026 benchmark sample (marker_list/population_2026.json)."""
    V = json.load(open(os.path.join(MARK, "validation.json")))["years"]
    P26 = json.load(open(os.path.join(MARK, "population_2026.json")))
    V = {**V, "2026": {"mwi": P26["mwi"], "mwi_no_top5": P26["mwi_no_top5"]}}
    years = sorted(int(y) for y in V)
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    for key, c, lab in (("mwi", S1, "Full list"), ("mwi_no_top5", S2, "Top-5 markers removed")):
        for seg in ([y for y in years if y <= 2022], [y for y in years if y >= 2024]):
            ax.plot(seg, [V[str(y)][key] for y in seg], color=c, lw=2, marker="o", ms=4,
                    label=lab if seg[0] == years[0] else None)
        y_last = V[str(years[-1])][key]
        ax.text(years[-1] + 0.15, y_last, f"{y_last:.2f}", va="center", fontsize=7, color=INK)
    ax.axvspan(2022.5, 2023.5, color=GRID, alpha=0.6, lw=0)
    ax.text(2023, ax.get_ylim()[1] * 0.97, "2023\nnot\nsampled", ha="center", va="top", fontsize=6, color=INK2)
    ax.set_xticks(years); ax.set_xlim(years[0] - 0.4, years[-1] + 0.9); ax.set_ylim(bottom=0)
    ax.set_ylabel("Marker words per 1,000 tokens"); ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout(); save(fig, "fig_marker_trend")


def groups():
    """Author-level mean MWI by wave and group, one panel per cohort (and the replication sample)."""
    panels = [("pooled", "All pairs"), ("2024-25", "2024–25 index papers"), ("2026", "2026 index papers")]
    reps = [(k, t) for k, t in (("replication", "Replication, 2024–25"), ("replication2026", "Replication, 2026"))
            if k in R]
    fig, axes = plt.subplots(1, len(panels) + len(reps), figsize=(FULL_W, 2.05), sharey=True)
    x = [0, 1, 2]
    for ax, (key, title) in zip(axes, panels):
        for g, c, lab in ((1, S2, "Disclosers"), (0, S1, "Non-disclosing controls")):
            m = [R["mwi_means"][f"{key}|{g}|{w}"] for w in ("W1", "W2", "W3")]
            ax.errorbar(x, [v["mean"] for v in m], yerr=[1.96 * v["se"] for v in m], color=c, marker="o", ms=3.5,
                        lw=1.4, capsize=2, label=lab)
        n = R["mwi_means"][f"{key}|1|W3"]["n"]
        ax.set_title(f"{title}\n({n} pairs)", fontsize=7.5, color=INK)
        ax.set_xticks(x, ["’14–16", "’19–21", "’24–26"])
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for ax, (key, title) in zip(axes[len(panels):], reps):
        m = R[key]["mwi_means"]
        for g, c in ((1, S2), (0, S1)):
            ax.plot(x, [m[f"{g}|{w}"] for w in ("W1", "W2", "W3")], color=c, marker="o", ms=3.5, lw=1.4)
        ax.set_title(f"{title}\n({R[key]['n_pairs']} new pairs)", fontsize=7.5, color=INK)
        ax.set_xticks(x, ["’14–16", "’19–21", "’24–26"])
        ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_ylabel("Marker-word index\n(per 1,000 tokens)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "fig_groups")


def cohort_forest():
    """Discloser excess beta3 (MWI, SD units) across cohorts, query types, sampling frames and the replication."""
    rows = [("All pairs", R["main"]["pooled"]["mwi"]["d_w3"], R["main"]["pooled"]["mwi"]["n_pairs"], INK)]
    rows.append(("2024–25 index papers", R["main"]["2024-25"]["mwi"]["d_w3"], R["main"]["2024-25"]["mwi"]["n_pairs"], S2))
    rows.append(("2026 index papers", R["main"]["2026"]["mwi"]["d_w3"], R["main"]["2026"]["mwi"]["n_pairs"], S1))
    lab = {"chatgpt_phrasing": "ChatGPT phrasing", "other_phrasings": "Other phrasings", "pilot": "Pilot frame",
           "ncbi": "Main frame"}
    for src in ("by_query_cohort", "by_frame_cohort"):
        for k, v in R[src].items():
            a, c = k.split("|")
            rows.append((f"  {lab[a]}, {c.replace('-', '–')}", v, v["n_pairs"], S2 if c == "2024-25" else S1))
    if "replication" in R:
        rows.append(("Replication, 2024–25 (new pairs)", R["replication"]["beta3"], R["replication"]["n_pairs"], S2))
    if "replication2026" in R:
        rows.append(("Replication, 2026 (new pairs)", R["replication2026"]["beta3"], R["replication2026"]["n_pairs"], S1))
    if "all_samples" in R:
        rows.append(("All samples, 2024–25", R["all_samples"]["2024-25"], R["all_samples"]["2024-25"]["n_pairs"], S2))
        rows.append(("All samples, 2026", R["all_samples"]["2026"], R["all_samples"]["2026"]["n_pairs"], S1))
    elif "replication" in R:
        rows.append(("2024–25, main + replication", R["replication"]["combined_2024_25"],
                     R["replication"]["combined_2024_25"]["n_pairs"], S2))
    fig, ax = plt.subplots(figsize=(COL_W, 0.24 * len(rows) + 0.7))
    for i, (name, e, n, c) in enumerate(rows):
        ax.plot([e["lo"], e["hi"]], [i, i], color=c, lw=1.4)
        ax.plot(e["b"], i, "o", color=c, ms=4)
        ax.text(3.05, i, f"{n}", va="center", ha="right", fontsize=6.3, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(-1.3, 3.1)
    ax.text(3.05, -1, "Pairs", ha="right", fontsize=6.3, color=INK2)
    ax.set_xlabel("$\\hat\\beta_3$ on the MWI (SD, 95% CI)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig_forest")


def placebo():
    d = np.load(os.path.join(EXP, "revision_placebo_draws.npy"))
    P = R["placebo_lexicons"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 1.9))
    ax = axes[0]
    ax.hist(d[:, 0], bins=40, color=S3, alpha=0.7)
    ax.axvline(P["marker_change32"], color=S2, lw=1.6)
    ax.axvline(P["content_change32"], color=S1, lw=1.2, ls="--")
    ax.text(P["marker_change32"], ax.get_ylim()[1] * 0.9, " Marker lexicon", color=S2, fontsize=6.5)
    ax.text(P["content_change32"], ax.get_ylim()[1] * 0.7, " Content words", color=S1, fontsize=6.5)
    ax.set_xlabel("Within-author change, 2019–21 to 2024–26 (per 1,000 tokens)")
    ax.set_ylabel("Random lexicons")
    ax = axes[1]
    ax.hist(d[:, 2], bins=40, color=S3, alpha=0.7)
    obs = P["beta3_2024-25"]["obs"]
    ax.axvline(obs, color=S2, lw=1.6)
    ax.text(obs, ax.get_ylim()[1] * 0.9, " Marker lexicon", color=S2, fontsize=6.5, ha="right")
    ax.set_xlabel("Discloser excess $\\hat\\beta_3$ in the 2024–25 cohort (SD units)")
    for a in axes:
        a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig_placebo")


def words():
    W = pd.read_csv(os.path.join(EXP, "revision_word_shifts.csv"))
    top = W.sort_values("c2425_did", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    y = np.arange(len(top))
    ax.barh(y - 0.2, top["c2425_did"], height=0.38, color=S2, label="2024–25 cohort")
    ax.barh(y + 0.2, top["c2026_did"], height=0.38, color=S1, label="2026 cohort")
    ax.set_yticks(y, top["word"], fontsize=7); ax.invert_yaxis()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Discloser minus control within-author change\n(occurrences per 1,000 tokens)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout(); save(fig, "fig_words")


def refresh():
    """Version-1 lexicon vs. the 2025-26 rising-word lexicon, by cohort and sample."""
    if "lexicon_refresh" not in R:
        return
    LR = R["lexicon_refresh"]
    rows = [("Main, 2024–25", R["main"]["2024-25"]["mwi"]["d_w3"], LR["main"]["rising_2026"]["2024-25"])]
    if "replication" in LR:
        rows.append(("Replication, 2024–25", R["replication"]["beta3"], LR["replication"]["rising_2026"]["pooled"]))
    rows.append(("Main, 2026", R["main"]["2026"]["mwi"]["d_w3"], LR["main"]["rising_2026"]["2026"]))
    if "replication2026" in LR:
        rows.append(("Replication, 2026", R["replication2026"]["beta3"], LR["replication2026"]["rising_2026"]["pooled"]))
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for i, (name, v1, r26) in enumerate(rows):
        for off, e, c, lab in ((-0.14, v1, S2, "Version-1 lexicon (2024–25 excess words)"),
                               (0.14, r26, S1, "Words rising 2025–26")):
            ax.plot([e["lo"], e["hi"]], [i + off, i + off], color=c, lw=1.4)
            ax.plot(e["b"], i + off, "o", color=c, ms=4, label=lab if i == 0 else None)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("Discloser excess $\\hat\\beta_3$ (SD, 95% CI)")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1)
    fig.tight_layout(); save(fig, "fig_refresh")


def methodology():
    """Overview figure: how the data were built (A), the three-wave comparison (B) and the main results (C).
    All numbers are read from the results files."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    d = json.load(open(os.path.join(MARK, "derivation_stats.json")))
    v = json.load(open(os.path.join(MARK, "validation.json")))
    n_lex = sum(d["n_by_year"].values()) + d["excluded_pilot_author_abstracts"] + sum(x["n"] for x in v["years"].values())
    F = R["flow"]
    m = R["main"]
    f2 = lambda e: f"{e['b']:+.2f}".replace("-", "−")
    rep, rep6 = R.get("replication"), R.get("replication2026")
    LR = R.get("lexicon_refresh", {})
    PALE1, PALE2, PALEG = "#e8f0fb", "#fdeee6", "#f3f2ee"

    fig = plt.figure(figsize=(FULL_W, 4.7))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis("off")

    def box(x, y, w, h, title, body, face=PALEG, edge=INK2, tsize=7.2, bsize=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=face, ec=edge, lw=0.8))
        ax.text(x + 1.0, y + h - 1.2, title, ha="left", va="top", fontsize=tsize, weight="bold", color=INK)
        ax.text(x + 1.0, y + h - 4.6, body, ha="left", va="top", fontsize=bsize, color=INK, linespacing=1.35)

    def arrow(x0, y0, x1, y1, c=INK2):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=c, lw=1.0))

    def band(y, label):
        ax.text(0.8, y, label, fontsize=8, weight="bold", color=INK, va="center")

    # ---- A. data
    band(68.2, "A   Building the data")
    box(1, 49.5, 31, 16.5, "1  Marker-word lexicon",
        f"{n_lex:,} random PubMed abstracts\n(2019–22 and 2024–25)\n• words in excess in both 2024 and 2025\n"
        f"• 187 style words kept, frozen (SHA-256)\n• validated on the held-out half;\n  2026 benchmark: {POP26['n']:,} abstracts")
    box(34.5, 49.5, 31, 16.5, "2  AI-disclosing papers",
        "Open-access PMC papers, 2024–26, with an\nAI-use disclosure for writing\n• disclosure verified in the full text\n"
        "• every sentence read (false positives out)\n• anchor = corresponding author,\n  ≥2 abstracts in each pre-LLM wave", face=PALE2, edge=S2)
    box(68, 49.5, 31, 16.5, "3  Matched non-disclosers",
        "Same journal, published within ±120 days,\nno AI disclosure in the full text\n• full text re-checked; reused authors out\n"
        f"• same anchor and eligibility rules\n• {F['analysis_pairs']} pairs: {F['by_cohort']['2024-25']} from 2024–25,\n"
        f"  {F['by_cohort']['2026']} from 2026", face=PALE1, edge=S1)
    arrow(65.6, 57.5, 67.9, 57.5)

    # ---- B. design: timeline
    band(43.2, "B   Comparing each author with their own earlier abstracts")
    y0, y1 = 31.5, 25.5                        # discloser and control rows
    xs = {2014: 13, 2026: 75}
    X = lambda yr: xs[2014] + (yr - 2014) * (xs[2026] - xs[2014]) / 12
    for (a, b, lab, sub) in ((2014, 2016.99, "W1  2014–16", "placebo"), (2019, 2021.99, "W2  2019–21", "reference"),
                             (2024, 2026.99, "W3  2024–26", "index paper")):
        ax.add_patch(FancyBboxPatch((X(a), 21.8), X(b) - X(a), 14.5, boxstyle="round,pad=0.2,rounding_size=0.6",
                                    fc=PALEG, ec=GRID, lw=0.8))
        ax.text((X(a) + X(b)) / 2, 38.3, lab, ha="center", fontsize=7, weight="bold")
        ax.text((X(a) + X(b)) / 2, 36.9, sub, ha="center", fontsize=6.2, color=INK2)
    ax.plot([X(2022.9)] * 2, [21.0, 38.0], color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(X(2022.9), 39.6, "ChatGPT\nNov 2022", ha="center", fontsize=6, color=INK2, linespacing=1.1)
    for y, c, lab in ((y0, S2, "Discloser"), (y1, S1, "Control")):
        ax.text(1.2, y, lab, fontsize=7, color=c, weight="bold", va="center")
        ax.plot([X(2014.2), X(2026.8)], [y, y], color=c, lw=0.6, alpha=0.4)
        for yr, k in ((2014.3, 5), (2019.3, 5), (2025.5, 1)):
            for j in range(k):
                ax.plot(X(yr) + j * 1.3, y, "o", color=c, ms=3.2)
    ax.text(X(2015.5), 22.6, f"median {R['median_texts_author_wave']['W1']:.0f} abstracts", ha="center", fontsize=5.8, color=INK2)
    ax.text(X(2020.5), 22.6, f"median {R['median_texts_author_wave']['W2']:.0f} abstracts", ha="center", fontsize=5.8, color=INK2)
    ax.text(X(2025.5), 22.6, "1 abstract (index paper)", ha="center", fontsize=5.8, color=INK2)
    ax.text(82.5, 36.0, "Outcomes per abstract", fontsize=6.5, weight="bold", va="top")
    ax.text(82.5, 33.6, "• marker-word index\n• lexical diversity\n• reading ease\n• sentence length\n• function-word\n  drift",
            fontsize=6, va="top", linespacing=1.3)
    ax.text(X(2017.5), 19.0, "β₁: change W1→W2, discloser − control\n(should be 0 before LLMs)", ha="center", fontsize=6.2,
            color=INK, linespacing=1.25)
    ax.text(X(2023.0), 19.0, "β₃: change W2→W3, discloser − control\n= discloser excess (author fixed effects)", ha="center",
            fontsize=6.2, color=INK, linespacing=1.25)

    # ---- C. results
    band(13.6, "C   What we found")
    w, h, yb = 23.6, 11.2, 0.6
    box(1, yb, w, h, "Main sample: marker words",
        f"before LLMs: β₁ = {f2(m['pooled']['mwi']['d_w1'])} (≈ 0)\n2024–25 papers: β₃ = {f2(m['2024-25']['mwi']['d_w3'])} SD\n"
        f"2026 papers: β₃ = {f2(m['2026']['mwi']['d_w3'])} SD")
    if rep and rep6:
        box(25.8, yb, w, h, "Replication samples",
            f"2024–25 ({rep['n_pairs']} pairs): β₃ = {f2(rep['beta3'])} SD\n2026 ({rep6['n_pairs']} pairs): β₃ = {f2(rep6['beta3'])} SD\n"
            "all 3 pre-set hypotheses held")
    box(50.6, yb, w, h, "Lexicon-free measures",
        f"gap persists in both periods;\nlexical diversity, β₃:\n2024–25 {f2(m['2024-25']['mtld']['d_w3'])}, 2026 {f2(m['2026']['mtld']['d_w3'])} SD")
    if LR:
        box(75.4, yb, w, h, "Words rising 2025→26",
            f"2026 papers: β₃ = {f2(LR['main']['rising_2026']['2026'])} SD\n(replication {f2(LR['replication2026']['rising_2026']['pooled'])})\n"
            f"2024–25 papers: β₃ = {f2(LR['main']['rising_2026']['2024-25'])} SD")
    save(fig, "fig_method")


def descriptor():
    """Data descriptor: sample flow, publication years, search phrasings, abstracts per author, tools and purposes."""
    import sys
    sys.path.insert(0, EXP)
    from coding import code
    S3 = "#1baf7a"
    P = pd.read_csv(os.path.join(EXP, "revision_pairs.csv"), index_col=0)
    P = P[~P.fp & ~P.dup_ctrl & ~P.ctrl_discloses]
    D = json.load(open(os.path.join(EXP, "disclosers.json")))["papers"]
    main_disc = [D[i]["disclosure"] for i in P["paper_index"]]
    reps = []
    for suf, off, tag in (("", 10000, "Replication 2024–25"), ("2026", 20000, "Replication 2026")):
        f = os.path.join(EXP, f"replication{suf}_units.json")
        mf = os.path.join(EXP, f"revision_replication{suf}_metrics.csv")
        if os.path.exists(f) and os.path.exists(mf):
            keep = set(pd.read_csv(mf, usecols=["pair"])["pair"])
            for k, p in enumerate(json.load(open(f))):
                if off + k in keep:
                    reps.append({"tag": tag, "year": int(p["disc"]["year"]), "query": p["disc"]["query"],
                                 "disclosure": p["disc"]["disclosure"]})
    Rp = pd.DataFrame(reps)
    M = pd.read_csv(os.path.join(EXP, "revision_metrics.csv"), usecols=["author", "group", "wave"])
    fig, axs = plt.subplots(3, 2, figsize=(FULL_W, 6.0))
    lab = lambda ax, t: ax.set_title(t, loc="left", fontsize=7.5, color=INK, weight="bold")

    def grid(ax, a="x"):
        getattr(ax, f"{a}axis").grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)

    # (a) sample flow
    ax = axs[0, 0]
    cf, F = R["collection_flow"], R["flow"]
    steps = [("Candidate papers", cf["pool"]), ("Screened in full text", cf["screened"]),
             ("Disclosure found", cf["detected"]), ("Distinct authors", cf["unique_anchor_authors"]),
             ("Eligible author", cf["eligible"]), ("Matched to a control", cf["matched"]),
             ("Complete in all waves", F["complete_pairs"]), ("After audit", F["analysis_pairs"])]
    y = np.arange(len(steps))
    ax.barh(y, [v for _, v in steps], color=S1, height=0.62)
    for i, (_, v) in enumerate(steps):
        ax.text(v + 90, i, f"{v:,}", va="center", fontsize=6.3, color=INK)
    ax.set_yticks(y, [n for n, _ in steps], fontsize=6.3); ax.invert_yaxis()
    ax.set_xlim(0, cf["pool"] * 1.25); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
    lab(ax, "(a) Main-sample flow")

    # (b) pairs by publication year of the index paper
    ax = axs[0, 1]
    yrs = [2024, 2025, 2026]
    main_y = [int((P["year"] == yy).sum()) for yy in yrs]
    rep_y = [int((Rp["year"] == yy).sum()) if len(Rp) else 0 for yy in yrs]
    x = np.arange(3)
    ax.bar(x, main_y, color=S1, width=0.6, label="Main sample")
    ax.bar(x, rep_y, bottom=[v + 3 for v in main_y], color=S3, width=0.6, label="Replication samples")
    for i in range(3):
        ax.text(i, main_y[i] + 3 + rep_y[i] + 8, f"{main_y[i]} + {rep_y[i]}", ha="center", fontsize=6.3, color=INK)
    ax.set_xticks(x, [str(v) for v in yrs]); ax.set_ylabel("Matched pairs", fontsize=6.5)
    ax.set_ylim(0, max(m_ + r_ for m_, r_ in zip(main_y, rep_y)) * 1.45); grid(ax, "y")
    ax.legend(frameon=False, fontsize=6.2, loc="upper left")
    lab(ax, "(b) Pairs by year of the index paper")

    # (c) search phrasing by sample
    ax = axs[1, 0]
    Q = [("chatgpt_language", "ChatGPT phrasing", S1), ("elsevier_template", "Elsevier template", S2),
         ("genai_generic", "“Generative AI” phrasing", S3)]
    groups_ = [("Main 2024–25", P[P.cohort == "2024-25"]["query"]), ("Main 2026", P[P.cohort == "2026"]["query"])]
    if len(Rp):
        groups_ += [(t, Rp[Rp.tag == t]["query"]) for t in ("Replication 2024–25", "Replication 2026")]
    for i, (gname, qs) in enumerate(groups_):
        left = 0
        for key, name, c in Q:
            share = 100 * (qs == key).mean()
            ax.barh(i, share, left=left, color=c, height=0.62, edgecolor=SURF, lw=1, label=name if i == 0 else None)
            if share >= 12:
                ax.text(left + share / 2, i, f"{share:.0f}%", ha="center", va="center", fontsize=6, color="white")
            left += share
    ax.set_yticks(range(len(groups_)), [f"{g} (n = {len(q)})" for g, q in groups_], fontsize=6.2); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xlabel("Share of disclosers (%)", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6.2, loc="upper center", bbox_to_anchor=(0.35, -0.3), ncol=3, handlelength=1,
              columnspacing=0.8)
    lab(ax, "(c) Search phrasing that found the paper")

    # (d) abstracts per author, pre-LLM waves
    ax = axs[1, 1]
    n = M[M.wave != "W3"].groupby(["group", "author"]).size()
    bins = np.arange(0, 104, 4)
    for g, c, name in ((1, S2, "Disclosers"), (0, S1, "Controls")):
        ax.hist(n.xs(g).clip(upper=100), bins=bins, histtype="step", lw=1.6, color=c,
                label=f"{name} (median {int(np.median(n.xs(g)))})")
    ax.set_xlabel("Pre-LLM abstracts per author (last bin: 100+)", fontsize=6.2)
    ax.set_ylabel("Authors", fontsize=6.5); grid(ax, "y")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right")
    lab(ax, "(d) Pre-LLM baseline per author")

    # (e) tools and (f) purposes named in disclosures (main sample + replications)
    allc = [code(t) for t in main_disc + (list(Rp["disclosure"]) if len(Rp) else [])]
    tools = Counter(t for c in allc for t in c["tools"])
    purp = Counter(p for c in allc for p in c["purposes"])
    names = {"polishing": "Language polishing", "grammar": "Grammar, proofreading",
             "drafting/structuring": "Drafting, structuring", "translation": "Translation", "Unspecified": "Tool not named"}
    for ax, cnt, title, top in ((axs[2, 0], tools, f"(e) AI tools named ({len(allc)} disclosures)", 8),
                                (axs[2, 1], purp, "(f) Purpose disclosed", 4)):
        items = cnt.most_common(top)
        yy = np.arange(len(items))
        vals = [100 * v / len(allc) for _, v in items]
        ax.barh(yy, vals, color=S2, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=6.2, color=INK)
        ax.set_yticks(yy, [names.get(k, k) for k, _ in items], fontsize=6.3); ax.invert_yaxis()
        ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.set_xlabel("Share of disclosures (%)", fontsize=6.3)
        grid(ax); lab(ax, title)
    fig.tight_layout(h_pad=1.8, w_pad=1.5)
    save(fig, "fig_data")


if __name__ == "__main__":
    retention(); fulltext(); marker_trend()
    groups(); cohort_forest(); placebo(); words(); refresh(); methodology(); descriptor()
