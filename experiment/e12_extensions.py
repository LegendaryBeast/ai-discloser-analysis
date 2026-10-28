"""E12 (exploratory, round-3 revision).
  Study 1  Disclosed model generation: is the frozen lexicon's discloser signal carried by disclosures naming models
           released in 2024 or earlier, and the 2025-26 rising-word signal by disclosures naming models released in
           2025-26? Estimated pooled, within 2026 only, and with period-specific discloser terms.
  Study 2  Prevalence: the excess share of held-out PubMed abstracts containing at least one rare marker word, relative
           to the 2021-22 trend (Kobak et al. 2025), by year, for the frozen lexicon and the refreshed lexicon.
Writes extensions_results.json and ../paper/tables/tab_generation.tex."""
import gzip, json, math, re, warnings
import numpy as np, pandas as pd

import e10_revision as E
import e11_impact as I
from textmetrics import TOKEN

warnings.filterwarnings("ignore")

OLDER = re.compile(r"gpt[\s\-]?(?:3\.5|4(?!\.[15])(?:\.0|o|-?turbo|o-mini| mini)?)\b|chat ?gpt[\s\-]?(?:3\.5|4o?|4\.0)\b|"
                   r"claude[\s\-]?(?:instant|[23](?:\.5)?\b)|gemini[\s\-]?(?:pro|1(?:\.\d)?\b|advanced)|\bbard\b", re.I)
NEWER = re.compile(r"gpt[\s\-]?(?:4\.[15]|5(?:\.\d)?)|chat ?gpt[\s\-]?5|\bo[134](?:-mini)?\b|claude[\s\-]?(?:3\.7|4)|"
                   r"(?:sonnet|opus|haiku)[\s\-]?4|gemini[\s\-]?(?:2|3)|deepseek|qwen|kimi|doubao", re.I)


def generation(sentence):
    o, n = bool(OLDER.search(sentence)), bool(NEWER.search(sentence))
    return "older" if o and not n else "newer" if n and not o else "mixed" if o and n else "unspecified"


def disclosure_sentences():
    P = pd.read_csv("revision_pairs.csv", index_col=0)
    P = P[~P.fp & ~P.dup_ctrl & ~P.ctrl_discloses]
    D = json.load(open("disclosers.json"))["papers"]
    s = {int(k): D[int(i)]["disclosure"] for k, i in P["paper_index"].items()}
    for suf, off in (("", 10000), ("2026", 20000)):
        for k, p in enumerate(json.load(open(f"replication{suf}_units.json"))):
            s[off + k] = p["disc"]["disclosure"]
    return s


def study1(df):
    sents = disclosure_sentences()
    df["gen"] = df["pair"].map(lambda k: generation(sents.get(int(k), "")))
    counts = df.drop_duplicates("pair").groupby(["gen", "cohort"]).size().unstack(fill_value=0).to_dict()
    out = {"counts_by_cohort": counts, "n_pairs": df.drop_duplicates("pair")["gen"].value_counts().to_dict()}
    for scope, sub in (("all", df), ("2026_only", df[df.cohort == "2026"])):
        out[scope] = {}
        for g in ("older", "newer", "unspecified"):
            s = sub[sub.gen == g]
            out[scope][g] = {y: (E.fit(s, y + "_z")["d_w3"] | {"n_pairs": int(s.pair.nunique())}) if s.pair.nunique() >= 15 else None
                             for y in ("mwi", "rising", "mtld")}
        s = sub[sub.gen.isin(["older", "newer"])]
        out[scope]["newer_minus_older"] = {y: E.fit_mod(s, (s.gen == "newer").astype(int), y + "_z") for y in ("mwi", "rising", "mtld")}
    # generation contrast net of calendar time: period-specific wave and discloser terms
    s = df[df.gen.isin(["older", "newer"])].copy()
    newer = (s.gen == "newer").astype(float)
    w1, w3 = (s.wave == "W1").astype(float), (s.wave == "W3").astype(float)
    x = s.assign(W1=w1, W3=w3, log_na=np.log(s.n_authors), log_nt=np.log(s.n_tokens), g3=s.d_w3 * newer, g1=s.d_w1 * newer,
                 gw1=w1 * newer, gw3=w3 * newer)
    extra = []
    for pnm in sorted(s["period"].unique())[1:]:
        q = (s["period"] == pnm).astype(float)
        for nm, v in ((f"p{len(extra)}a", w1 * q), (f"p{len(extra)}b", w3 * q), (f"p{len(extra)}c", s.d_w3 * q),
                      (f"p{len(extra)}d", s.d_w1 * q)):
            x[nm] = v; extra.append(nm)
    out["newer_minus_older_period_adjusted"] = {}
    for y in ("mwi", "rising", "mtld"):
        xx = x.dropna(subset=[y + "_z"])
        b, V = E.fe_fit(xx, y + "_z", ["g3", "d_w3", "d_w1", "W1", "W3", "log_na", "structured", "log_nt", "g1", "gw1", "gw3"] + extra)
        se = math.sqrt(V[0, 0])
        out["newer_minus_older_period_adjusted"][y] = {"b": float(b[0]), "lo": float(b[0] - 1.96 * se),
                                                      "hi": float(b[0] + 1.96 * se), "p": float(2 * E.norm.sf(abs(b[0] / se)))}
    return out


def study2():
    """Excess share of held-out abstracts containing >=1 rare marker (2022 document frequency < 1%), vs. the linear
    2021-22 extrapolation of that share. Frozen lexicon: held-out (odd-PMID) halves of 2019-2026. Refreshed lexicon:
    the same held-out halves; it was derived on the even-PMID half of 2026."""
    L = json.load(open("../marker_list/marker_list_v1.json"))
    R = json.load(open("../marker_list/lexicon_2026.json"))
    docs = {}  # year -> list of token sets (odd PMIDs)
    with gzip.open("../marker_list/corpus.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if int(r["pmid"]) % 2 == 1:
                docs.setdefault(r["year"], []).append(set(TOKEN.findall(r["abstract"].lower())))
    with gzip.open("../marker_list/population_2026.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if int(r["pmid"]) % 2 == 1:
                docs.setdefault(2026, []).append(set(TOKEN.findall(r["abstract"].lower())))
    df22 = {}
    for s in docs[2022]:
        for w in s:
            df22[w] = df22.get(w, 0) + 1
    n22 = len(docs[2022])
    rare = lambda ws: sorted(w for w in ws if df22.get(w, 0) / n22 < 0.01)
    # size-matched refreshed list: the rare rising words with the largest 2025->2026 gain on the DERIVATION halves
    # (even PMIDs), as many as in the rare frozen list; evaluated, like the frozen list, on the held-out halves
    ev = {2025: {}, 2026: {}}; nev = {2025: 0, 2026: 0}
    for path, yr in (("../marker_list/corpus.jsonl.gz", 2025), ("../marker_list/population_2026.jsonl.gz", 2026)):
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if int(r["pmid"]) % 2 == 0 and r.get("year", 2026) == yr:
                    nev[yr] += 1
                    for w in set(TOKEN.findall(r["abstract"].lower())):
                        ev[yr][w] = ev[yr].get(w, 0) + 1
    gain = lambda w: ev[2026].get(w, 0) / nev[2026] - ev[2025].get(w, 0) / nev[2025]
    fr = rare(L["words"])
    matched = sorted(sorted(rare(R["rising_2026"]), key=gain, reverse=True)[:len(fr)])
    sets = {"frozen": fr, "refreshed": rare(R["rising_2026"]), "refreshed_matched": matched}
    out = {"n_words": {k: len(v) for k, v in sets.items()}, "n_docs": {y: len(v) for y, v in sorted(docs.items())}}
    for name, ws in sets.items():
        W = set(ws)
        share = {y: float(np.mean([bool(s & W) for s in v])) for y, v in sorted(docs.items())}
        slope = max(share[2022] - share[2021], 0)
        rows = {}
        for y in (2024, 2025, 2026):
            exp = share[2022] + slope * (y - 2022)
            n = len(docs[y])
            se = math.sqrt(share[y] * (1 - share[y]) / n)
            rows[y] = {"observed": share[y], "expected": exp, "excess": share[y] - exp,
                       "lo": share[y] - exp - 1.96 * se, "hi": share[y] - exp + 1.96 * se}
        out[name] = {"share_by_year": share, "excess": rows}
    out["matched_examples"] = sorted(matched, key=gain, reverse=True)[:12]
    return out


def sampled_days(path="../marker_list/population_2026.jsonl.gz"):
    """Day index (0-9) of each record of the 2026 benchmark sample. m5 writes the ten sampled Entrez dates in
    chronological order, and PMIDs are assigned in order of entry, so a new day starts where the running median
    PMID jumps."""
    p = [int(json.loads(line)["pmid"]) for line in gzip.open(path, "rt")]
    w, starts = 250, [0]
    med = [np.median(p[i:i + w]) for i in range(0, len(p), w)]
    for k in range(1, len(med)):
        if abs(med[k] - med[k - 1]) > 20000:
            b = k * w  # refine the boundary to the record where PMIDs move to the next day's range
            mid = (med[k] + med[k - 1]) / 2
            starts.append(next(i for i in range(b - w, b + w) if all(x > mid for x in p[i:i + 50])))
    assert len(starts) == 10, starts
    return np.searchsorted(starts, np.arange(len(p)), side="right") - 1


def study2_forward():
    """Forward test of the refreshed prevalence series. The study2 refreshed list is selected for rising from 2025
    to 2026, so its 2025->2026 rise holds by construction. Here the list is derived only from the first five sampled
    days of 2026 (13 Jan - 11 May; m6 rule: ratio >= 1.5, gap >= 0.0005, >= 30 documents vs 2025, no v1 candidate,
    no AI-topic word, no index paper; even PMIDs) and evaluated on the held-out odd PMIDs of the last five days
    (29 May - 21 Sep), which played no part in selecting it."""
    import sys
    sys.path.insert(0, "../marker_list")
    from m6_lexicon2026 import AI_TOPIC, index_pmids
    from m2_excess import MIN_RATIO, MIN_GAP, MIN_DOCS
    day = sampled_days()
    skip = index_pmids()
    v1 = json.load(open("../marker_list/marker_list_v1.json"))
    v1all = set(v1["words"]) | set(v1["excluded_content_words"])
    docs = {}  # (period, half) -> list of token sets
    with gzip.open("../marker_list/corpus.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            docs.setdefault((r["year"], int(r["pmid"]) % 2), []).append(set(TOKEN.findall(r["abstract"].lower())))
    with gzip.open("../marker_list/population_2026.jsonl.gz", "rt") as f:
        for k, line in enumerate(f):
            r = json.loads(line)
            if r["pmid"] in skip:
                continue
            per = "2026a" if day[k] < 5 else "2026b"
            docs.setdefault((per, int(r["pmid"]) % 2), []).append(set(TOKEN.findall(r["abstract"].lower())))
    count = lambda ds: {w: c for w, c in __import__("collections").Counter(w for s in ds for w in s).items()}
    c25, c26 = count(docs[(2025, 0)]), count(docs[("2026a", 0)])
    n25, n26 = len(docs[(2025, 0)]), len(docs[("2026a", 0)])
    rising = [w for w, c in c26.items() if c >= MIN_DOCS and w.isalpha() and w not in AI_TOPIC and w not in v1all
              and c / n26 - c25.get(w, 0) / n25 >= MIN_GAP and (w not in c25 or (c / n26) / (c25[w] / n25) >= MIN_RATIO)]
    odd = {y: docs[(y, 1)] for y in (2021, 2022, 2024, 2025, "2026a", "2026b")}
    df22 = count(odd[2022]); n22 = len(odd[2022])
    rare = lambda ws: sorted(w for w in ws if df22.get(w, 0) / n22 < 0.01)
    fr = rare(v1["words"])
    gain = lambda w: c26.get(w, 0) / n26 - c25.get(w, 0) / n25
    fwd = sorted(sorted(rare(rising), key=gain, reverse=True)[:len(fr)])
    out = {"n_rising": len(rising), "n_words": len(fr), "n_docs": {str(y): len(v) for y, v in odd.items()},
           "windows": {"2026a": "13 Jan - 11 May 2026", "2026b": "29 May - 21 Sep 2026"},
           "examples": sorted(fwd, key=gain, reverse=True)[:12]}
    for name, ws in (("frozen", fr), ("forward", fwd)):
        W = set(ws)
        share = {y: float(np.mean([bool(s & W) for s in v])) for y, v in odd.items()}
        slope = max(share[2022] - share[2021], 0)
        rows = {}
        for y, t in ((2024, 2024), (2025, 2025), ("2026a", 2026.2), ("2026b", 2026.6)):
            exp = share[2022] + slope * (t - 2022)
            se = math.sqrt(share[y] * (1 - share[y]) / len(odd[y]))
            rows[str(y)] = {"observed": share[y], "expected": exp, "excess": share[y] - exp,
                            "lo": share[y] - exp - 1.96 * se, "hi": share[y] - exp + 1.96 * se}
        d = rows["2026b"]["excess"] - rows["2026a"]["excess"]
        se = math.sqrt(sum(share[y] * (1 - share[y]) / len(odd[y]) for y in ("2026a", "2026b")))
        out[name] = {"excess": rows, "late_minus_early": {"b": d, "lo": d - 1.96 * se, "hi": d + 1.96 * se},
                     "selection_gain": share["2026a"] - share[2025], "continuation": share["2026b"] - share["2026a"]}
    return out


def study2_momentum():
    """Pre-LLM benchmark for the forward test: how much do words selected for rising over one interval keep rising
    for reasons unrelated to LLMs (topic and format trends)? Words are selected on the derivation halves for their
    rise from year a to year b (same rule as the forward list, same number of rare forms) and evaluated on the
    held-out halves in year b and the following year c. Rare = below 1% document frequency in year a."""
    import sys
    from collections import Counter
    sys.path.insert(0, "../marker_list")
    from m6_lexicon2026 import AI_TOPIC
    from m2_excess import MIN_RATIO, MIN_GAP, MIN_DOCS
    v1 = json.load(open("../marker_list/marker_list_v1.json"))
    v1all = set(v1["words"]) | set(v1["excluded_content_words"])
    docs = {}
    with gzip.open("../marker_list/corpus.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["year"] <= 2022:
                docs.setdefault((r["year"], int(r["pmid"]) % 2), []).append(set(TOKEN.findall(r["abstract"].lower())))
    cnt = {k: Counter(w for s in v for w in s) for k, v in docs.items()}
    share = lambda W, y: float(np.mean([bool(s & W) for s in docs[(y, 1)]]))
    out = {}
    for a, b, c in ((2019, 2020, 2021), (2020, 2021, 2022)):
        ca, cb, na, nb = cnt[(a, 0)], cnt[(b, 0)], len(docs[(a, 0)]), len(docs[(b, 0)])
        rising = [w for w, k in cb.items() if k >= MIN_DOCS and w.isalpha() and w not in AI_TOPIC and w not in v1all
                  and k / nb - ca.get(w, 0) / na >= MIN_GAP and (w not in ca or (k / nb) / (ca[w] / na) >= MIN_RATIO)]
        ra, nra = cnt[(a, 1)], len(docs[(a, 1)])
        rare = [w for w in rising if ra.get(w, 0) / nra < 0.01]
        gain = lambda w: cb.get(w, 0) / nb - ca.get(w, 0) / na
        W = set(sorted(rare, key=gain, reverse=True)[:N_MATCH])
        s = {y: share(W, y) for y in (a, b, c)}
        se = lambda y1, y2: math.sqrt(sum(s[y] * (1 - s[y]) / len(docs[(y, 1)]) for y in (y1, y2)))
        out[f"{a}-{b}"] = {"n_rising": len(rising), "n_words": len(W), "share": {str(y): v for y, v in s.items()},
                           "selection_gain": s[b] - s[a], "continuation": s[c] - s[b],
                           "continuation_ci": [s[c] - s[b] - 1.96 * se(b, c), s[c] - s[b] + 1.96 * se(b, c)],
                           "examples": sorted(W, key=gain, reverse=True)[:8]}
    return out


N_MATCH = 159  # rare forms in the frozen lexicon (study2 n_words["frozen"]); set from study2 in main()


def table(res):
    S = res["study1"]
    fmt = lambda e: "--" if e is None else (f"{e['b']:.2f} [{e['lo']:.2f}, {e['hi']:.2f}]").replace("-", "\\ensuremath{-}")
    L = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         "Model named in the disclosure & Pairs & Frozen 2024--25 lexicon & Words rising 2025--26 \\\\", "\\midrule",
         "\\multicolumn{4}{@{}l}{\\emph{All periods}} \\\\"]
    lab = {"older": "Released 2024 or earlier", "newer": "Released 2025--26", "unspecified": "No version named"}
    for scope, head in (("all", None), ("2026_only", "\\emph{2026 index papers only}")):
        if head:
            L.append("\\addlinespace\\multicolumn{4}{@{}l}{" + head + "} \\\\")
        for g in ("older", "newer", "unspecified"):
            e = S[scope][g]
            n = e["mwi"]["n_pairs"] if e["mwi"] else "--"
            L.append(f"{lab[g]} & {n} & {fmt(e['mwi'])} & {fmt(e['rising'])} \\\\")
        d = S[scope]["newer_minus_older"]
        L.append(f"Difference, newer $-$ older & & {fmt(d['mwi'])} & {fmt(d['rising'])} \\\\")
    d = S["newer_minus_older_period_adjusted"]
    L.append("\\addlinespace Difference, net of period & & " + f"{fmt(d['mwi'])} & {fmt(d['rising'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open("../paper/tables/tab_generation.tex", "w").write("\n".join(L))


def main():
    df = I.load()
    L26 = json.load(open("../marker_list/lexicon_2026.json"))
    df["rising"] = E.lex_rate(df, L26["rising_2026"])
    df["rising_z"] = I.zscore_on_main(df, "rising")
    global N_MATCH
    res = {"study1": study1(df), "study2": study2(), "study2_forward": study2_forward()}
    N_MATCH = res["study2"]["n_words"]["frozen"]
    res["study2_momentum"] = study2_momentum()
    json.dump(res, open("extensions_results.json", "w"), indent=1, default=float)
    table(res)
    s1 = res["study1"]
    print("pairs by generation:", s1["n_pairs"]); print("by cohort:", s1["counts_by_cohort"])
    for scope in ("all", "2026_only"):
        for g in ("older", "newer", "unspecified"):
            e = s1[scope][g]
            print(scope, g, {y: (round(v["b"], 2), round(v["lo"], 2), round(v["hi"], 2), v["n_pairs"]) if v else None for y, v in e.items()})
        print(scope, "newer-older", {y: (round(v["b"], 2), round(v["p"], 3)) for y, v in s1[scope]["newer_minus_older"].items()})
    print("period-adjusted", {y: (round(v["b"], 2), round(v["lo"], 2), round(v["hi"], 2), round(v["p"], 3)) for y, v in s1["newer_minus_older_period_adjusted"].items()})
    s2 = res["study2"]
    print("study2 words", s2["n_words"])
    for k in ("frozen", "refreshed", "refreshed_matched"):
        print(k, {y: (round(v["observed"], 3), round(v["excess"], 3)) for y, v in s2[k]["excess"].items()})
    print("matched examples", s2["matched_examples"])
    fw = res["study2_forward"]
    print("forward", fw["n_rising"], fw["n_words"], fw["n_docs"], fw["examples"])
    for k in ("frozen", "forward"):
        print(k, {y: round(v["excess"], 3) for y, v in fw[k]["excess"].items()}, fw[k]["late_minus_early"],
              "gain", round(fw[k]["selection_gain"], 3), "continuation", round(fw[k]["continuation"], 3))
    for k, v in res["study2_momentum"].items():
        print("pre-LLM momentum", k, v["n_rising"], v["n_words"], {y: round(x, 3) for y, x in v["share"].items()},
              "gain", round(v["selection_gain"], 3), "continuation", round(v["continuation"], 3), v["examples"])


if __name__ == "__main__":
    main()
