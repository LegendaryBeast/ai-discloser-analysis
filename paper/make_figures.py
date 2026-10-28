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
    """Overview figure, kept deliberately simple: three data steps (A), the three-wave comparison (B), three findings (C)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    F = R["flow"]
    PALE1, PALE2, PALEG = "#e8f0fb", "#fdeee6", "#f3f2ee"
    fig = plt.figure(figsize=(FULL_W, 3.5))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")

    def box(x, y, w, h, title, body, face=PALEG, edge=GRID):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8", fc=face, ec=edge, lw=0.9))
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=7.4, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.30, body, ha="center", va="center", fontsize=6.6, color=INK2)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, color=INK2, lw=0.9))

    def band(y, label):
        ax.text(1, y, label, fontsize=8, weight="bold", color=INK, va="center")

    # A: three steps
    band(47.6, "A   Data")
    w = 29
    for i, (title, body, face, edge) in enumerate((
            ("Marker words", "187 LLM-style words, frozen", PALEG, INK2),
            ("Disclosers", "papers disclosing AI use, verified", PALE2, S2),
            ("Matched controls", f"same journal, no disclosure ({F['analysis_pairs']} pairs)", PALE1, S1))):
        x = 2 + i * (w + 4.5)
        box(x, 36.5, w, 8.6, title, body, face, edge)
        if i == 2:
            arrow(x - 4.0, 40.8, x - 0.5, 40.8)

    # B: three waves per author
    band(31.6, "B   Same author, three waves")
    X = lambda yr: 14 + (yr - 2014) * (74 - 14) / 12
    for a, lab in ((2014, "2014–16\nbefore LLMs"), (2019, "2019–21\nreference"), (2024, "2024–26\nindex paper")):
        ax.add_patch(FancyBboxPatch((X(a), 15.5), X(a + 2.99) - X(a), 12.5, boxstyle="round,pad=0.2,rounding_size=0.6",
                                    fc=PALEG, ec=GRID, lw=0.8))
        ax.text(X(a + 1.5), 25.6, lab, ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.15)
    ax.plot([X(2022.9)] * 2, [15, 29], color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(X(2022.9), 29.9, "ChatGPT", ha="center", fontsize=6.2, color=INK2)
    for y, c, lab in ((20.6, S2, "Discloser"), (17.4, S1, "Control")):
        ax.text(11.5, y, lab, fontsize=6.8, color=c, weight="bold", va="center", ha="right")
        ax.plot([X(2014.1), X(2026.9)], [y, y], color=c, lw=0.6, alpha=0.35)
        for yr, k in ((2014.6, 4), (2019.6, 4), (2025.5, 1)):
            for j in range(k):
                ax.plot(X(yr) + j * 1.6, y, "o", color=c, ms=3)
    ax.text(89.5, 21.5, "Compare the\nchange in style:\ndiscloser − control", ha="center", va="center", fontsize=6.6,
            color=INK, linespacing=1.25)

    # C: three findings
    band(11.0, "C   Findings")
    for i, (title, body) in enumerate((("Marker words", "gap in 2024–25, none in 2026"),
                                       ("New samples", "same result, tested as planned"),
                                       ("Other style measures", "gap in both periods"))):
        box(2 + i * (w + 4.5), 0.8, w, 7.6, title, body)
    save(fig, "fig_method")


def descriptor():
    """Data descriptor in a compact 2 x 3 grid: flow, years, search phrasing, baseline size, tools, purposes."""
    import sys
    sys.path.insert(0, EXP)
    from coding import code
    S3 = "#1baf7a"
    P = pd.read_csv(os.path.join(EXP, "revision_pairs.csv"), index_col=0)
    P = P[~P.fp & ~P.dup_ctrl & ~P.ctrl_discloses]
    D = json.load(open(os.path.join(EXP, "disclosers.json")))["papers"]
    main_disc = [D[i]["disclosure"] for i in P["paper_index"]]
    reps = []
    for suf, off, tag in (("", 10000, "Repl. 2024–25"), ("2026", 20000, "Repl. 2026")):
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

    fig, axs = plt.subplots(2, 3, figsize=(FULL_W, 3.9), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.06, wspace=0.06, hspace=0.08)
    FS = 6.2

    def style(ax, title, axis="x"):
        ax.set_title(title, loc="left", fontsize=7.2, color=INK, weight="bold", pad=4)
        ax.tick_params(labelsize=FS, length=2)
        getattr(ax, f"{axis}axis").grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)

    def hbars(ax, labels, vals, color, fmt, xmax):
        y = np.arange(len(vals))
        ax.barh(y, vals, color=color, height=0.66)
        for i, v in enumerate(vals):
            ax.text(v + xmax * 0.02, i, fmt(v), va="center", fontsize=FS, color=INK)
        ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, xmax)
        ax.tick_params(axis="y", length=0)

    # (a) sample flow
    cf, F = R["collection_flow"], R["flow"]
    steps = [("Candidates", cf["pool"]), ("Screened", cf["screened"]), ("Disclosure found", cf["detected"]),
             ("Eligible authors", cf["eligible"]), ("Matched pairs", cf["matched"]), ("Analysed pairs", F["analysis_pairs"])]
    hbars(axs[0, 0], [s for s, _ in steps], [v for _, v in steps], S1, lambda v: f"{v:,}", cf["pool"] * 1.3)
    axs[0, 0].set_xticks([]); style(axs[0, 0], "(a) Sample flow")

    # (b) pairs by year
    ax = axs[0, 1]
    yrs = [2024, 2025, 2026]
    mv = np.array([(P["year"] == y).sum() for y in yrs])
    rv = np.array([(Rp["year"] == y).sum() if len(Rp) else 0 for y in yrs])
    x = np.arange(3)
    ax.bar(x, mv, width=0.62, color=S1, label="Main")
    ax.bar(x, rv, width=0.62, bottom=mv + 4, color=S3, label="Replication")
    for i in range(3):
        ax.text(i, mv[i] + rv[i] + 14, f"{mv[i] + rv[i]}", ha="center", fontsize=FS, color=INK)
    ax.set_xticks(x, [str(y) for y in yrs]); ax.set_ylim(0, (mv + rv).max() * 1.22)
    ax.legend(frameon=False, fontsize=FS, loc="upper left", handlelength=1, borderaxespad=0.2)
    style(ax, "(b) Pairs by year", "y")

    # (c) search phrasing
    ax = axs[0, 2]
    Q = [("chatgpt_language", "ChatGPT", S1), ("elsevier_template", "Elsevier", S2), ("genai_generic", "Gen. AI", S3)]
    rows = [("Main 2024–25", P[P.cohort == "2024-25"]["query"]), ("Main 2026", P[P.cohort == "2026"]["query"])]
    if len(Rp):
        rows += [(t, Rp[Rp.tag == t]["query"]) for t in ("Repl. 2024–25", "Repl. 2026")]
    for i, (_, qs) in enumerate(rows):
        left = 0
        for key, name, c in Q:
            sh = 100 * (qs == key).mean()
            ax.barh(i, sh, left=left, color=c, height=0.66, edgecolor=SURF, lw=0.8, label=name if i == 0 else None)
            if sh >= 14:
                ax.text(left + sh / 2, i, f"{sh:.0f}", ha="center", va="center", fontsize=FS - 0.3, color="white")
            left += sh
    ax.set_yticks(range(len(rows)), [r for r, _ in rows]); ax.invert_yaxis(); ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100]); ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=FS - 0.3, ncol=3, loc="upper left", bbox_to_anchor=(-0.02, -0.1),
              handlelength=0.9, columnspacing=0.7, borderaxespad=0)
    style(ax, "(c) Search phrasing (%)")

    # (d) baseline abstracts per author
    ax = axs[1, 0]
    n = M[M.wave != "W3"].groupby(["group", "author"]).size()
    bins = np.arange(0, 104, 5)
    for g, c, name in ((1, S2, "Disclosers"), (0, S1, "Controls")):
        ax.hist(n.xs(g).clip(upper=100), bins=bins, histtype="step", lw=1.3, color=c,
                label=f"{name}, median {int(np.median(n.xs(g)))}")
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100+"])
    ax.legend(frameon=False, fontsize=FS, loc="upper right", handlelength=1.2, borderaxespad=0.2)
    style(ax, "(d) Pre-LLM abstracts per author", "y")

    # (e) tools, (f) purposes
    allc = [code(t) for t in main_disc + (list(Rp["disclosure"]) if len(Rp) else [])]
    tools = Counter(t for c in allc for t in c["tools"]).most_common(6)
    purp = Counter(p for c in allc for p in c["purposes"]).most_common(4)
    names = {"polishing": "Polishing", "grammar": "Grammar", "drafting/structuring": "Drafting",
             "translation": "Translation", "Unspecified": "Not named"}
    for ax, items, title in ((axs[1, 1], tools, f"(e) Tools named (% of {len(allc)})"),
                             (axs[1, 2], purp, "(f) Purpose disclosed (%)")):
        hbars(ax, [names.get(k, k) for k, _ in items], [100 * v / len(allc) for _, v in items], S2,
              lambda v: f"{v:.0f}", 115)
        ax.set_xticks([0, 50, 100]); style(ax, title)
    save(fig, "fig_data")


def decay():
    """Central figure: discloser excess by period of the index paper for the frozen lexicon, the 2025-26 rising-word
    lexicon and lexical diversity, with the fitted linear trend of the frozen lexicon (experiment/impact_results.json).
    Periods are evenly spaced; the trend is evaluated at each period's midpoint."""
    f = os.path.join(EXP, "impact_results.json")
    if not os.path.exists(f):
        return
    I = json.load(open(f))
    per = I["decay"]["periods"]
    names = list(per)
    mids = {"2024": 2024.5, "2025 H1": 2025.25, "2025 H2": 2025.75, "2026 Q1": 2026.125, "2026 Q2": 2026.375,
            "2026 Q3–Q4": 2026.75}
    x = np.arange(len(names))
    S3 = "#1baf7a"
    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    ax.axhline(0, color=INK2, lw=0.8)
    L = I["decay"]["mwi_line"]
    fitted = [L["excess_jan2024"] + L["slope_per_year"] * (mids[n] - 2024.0) for n in names]
    ax.plot(x, fitted, color=S2, lw=0.9, ls=(0, (3, 2)), alpha=0.85, label="Linear trend, frozen lexicon")
    for key, lab, c, off in (("mwi", "Frozen 2024–25 lexicon", S2, -0.08), ("rising", "Words rising 2025–26", S3, 0.0),
                             ("mtld", "Lexical diversity (no lexicon)", S1, 0.08)):
        b = np.array([per[n][key]["b"] for n in names]); lo = np.array([per[n][key]["lo"] for n in names])
        hi = np.array([per[n][key]["hi"] for n in names])
        ax.errorbar(x + off, b, yerr=[b - lo, hi - b], color=c, marker="o", ms=3.6, lw=1.6, capsize=0, elinewidth=1,
                    label=lab)
    ax.set_xticks(x, [f"{n.replace(' ', chr(10))}" + f"{chr(10)}n={per[n]['n_pairs']}" for n in names], fontsize=6.1)
    ax.set_ylim(-1.0, 3.0)
    ax.set_ylabel("Discloser excess $\\hat\\beta_3$ (SD)", fontsize=6.8)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=5.9, loc="upper right", handlelength=1.6, borderaxespad=0.1)
    fig.tight_layout(); save(fig, "fig_decay")

def prevalence():
    """Study 2: excess share of held-out PubMed abstracts containing >=1 rare marker word, frozen vs refreshed lexicon."""
    f = os.path.join(EXP, "extensions_results.json")
    if not os.path.exists(f):
        return
    S = json.load(open(f))["study2"]
    yrs = ["2024", "2025", "2026"]
    S3 = "#1baf7a"
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    for k, lab, c in (("frozen", "Frozen 2024–25 lexicon", S2), ("refreshed_matched", "Refreshed 2025–26 lexicon", S3)):
        e = [S[k]["excess"][y] for y in yrs]
        v = [100 * x["excess"] for x in e]
        ax.errorbar(range(3), v, yerr=[[100 * (x["excess"] - x["lo"]) for x in e], [100 * (x["hi"] - x["excess"]) for x in e]],
                    color=c, marker="o", ms=4, lw=1.8, capsize=0, label=lab)
        for i, val in enumerate(v):
            ax.text(i + 0.08, val, f"{val:.1f}", fontsize=6.2, color=INK, va="center")
    ax.set_xticks(range(3), yrs); ax.set_xlim(-0.3, 2.45)
    ax.set_ylabel("Excess share of abstracts (%)", fontsize=6.8)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    fig.tight_layout(); save(fig, "fig_prevalence")


if __name__ == "__main__":
    retention(); fulltext(); marker_trend()
    groups(); cohort_forest(); placebo(); words(); refresh(); methodology(); descriptor(); decay(); prevalence()
