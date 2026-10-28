"""E11 (exploratory, round-2 revision): analyses that quantify how fast the marker lexicon lost validity and what that
means for detectors. Uses the main and both replication samples together (outputs of e10_revision.py).
  A  decay curve: discloser excess by period of the index paper, for the frozen lexicon, the 2025-26 rising-word
     lexicon and the lexicon-free outcomes; a linear time-trend model and a bootstrap CI for its zero-crossing date
  B  temporal transfer: detectors trained on disclosure labels from one period, tested on the other
  C  placebo for the rising-word lexicon: frequency-matched random lexicons, 2026 papers
  D  relative-magnitude breakdown value for parallel trends (Rambachan and Roth 2023)
Writes impact_results.json and ../paper/tables/tab_transfer.tex, tab_summary.tex."""
import datetime as dt, gzip, json, math, os, re, warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import e10_revision as E
from e3_controls import parse_date
from textmetrics import TOKEN

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2032)
NBOOT = int(os.environ.get("NBOOT", 1000))
NPLACEBO = int(os.environ.get("NPLACEBO", 200))
OUTC = [("mwi", "Version-1 lexicon"), ("rising", "2025–26 rising words"), ("mtld", "Lexical diversity"),
        ("flesch", "Reading ease"), ("drift", "Function-word drift")]
PERIODS = [("2024", (2024, 1), (2024, 12)), ("2025 H1", (2025, 1), (2025, 6)), ("2025 H2", (2025, 7), (2025, 12)),
           ("2026 Q1", (2026, 1), (2026, 3)), ("2026 Q2", (2026, 4), (2026, 6)), ("2026 Q3–Q4", (2026, 7), (2026, 12))]


def load():
    """All analysed texts of the three samples, with text, sample, phrasing and index date of the pair."""
    U = {u["doi"]: u for u in json.load(open("discloser_units.json")) if u["eligible"]}
    CU = [c for c in json.load(open("control_units.json")) if not c.get("unmatched")]
    pairs = [(U[c["matched_to"]], c) for c in CU if c["matched_to"] in U]
    info = {k: (parse_date(d["pub_date"]), d["query"]) for k, (d, c) in enumerate(pairs)}
    for suf, off in (("", 10000), ("2026", 20000)):
        for k, p in enumerate(json.load(open(f"replication{suf}_units.json"))):
            info[off + k] = (parse_date(p["disc"]["pub_date"]), p["disc"]["query"])
    parts = []
    for mf, tf, sample in (("revision_metrics.csv", "dataset.csv", "main"),
                           ("revision_replication_metrics.csv", "replication_dataset.csv", "rep2425"),
                           ("revision_replication2026_metrics.csv", "replication2026_dataset.csv", "rep2026")):
        m = pd.read_csv(mf)
        t = pd.read_csv(tf, usecols=["pair", "group", "work_id", "wave", "text"])
        m = m.merge(t, on=["pair", "group", "work_id", "wave"], how="left", validate="one_to_one")
        m["sample"] = sample
        parts.append(m)
    df = pd.concat(parts, ignore_index=True)
    df["cohort"] = df["cohort"].astype(str)  # the 2026 replication file stores the cohort as a number
    assert df["text"].notna().all()
    df["date"] = df["pair"].map(lambda k: info[k][0])
    df["query"] = df["pair"].map(lambda k: info[k][1])
    df["t_years"] = df["date"].map(lambda d: (d - dt.date(2024, 1, 1)).days / 365.25)
    clamp = lambda d: min(max((d.year, d.month), PERIODS[0][1]), PERIODS[-1][2])  # e.g. one 2027 issue date -> last period
    df["period"] = df["date"].map(lambda d: next(n for n, a, b in PERIODS if a <= clamp(d) <= b))
    df["early"] = (df["cohort"] == "2024-25").astype(int)
    return df


def zscore_on_main(df, col):
    ref = df.loc[(df["sample"] == "main") & (df.group == 0) & (df.wave == "W2"), col]
    return (df[col] - ref.mean()) / ref.std()


# ---------------------------------------------------------------- A  decay curve
def decay(df):
    out = {"periods": {}}
    for name, *_ in PERIODS:
        s = df[df["period"] == name]
        out["periods"][name] = {"n_pairs": int(s.pair.nunique()),
                                **{y: E.fit(s, y + "_z")["d_w3"] for y, _ in OUTC}}
    # linear trend in the discloser excess: d_w3 x years since Jan 2024 (pair-level moderator)
    out["trend"] = {}
    for y, _ in OUTC:
        mod = df["t_years"]
        out["trend"][y] = {"slope_per_year": E.fit_mod(df, mod, y + "_z")}
    # excess at Jan 2024 and zero crossing for the version-1 lexicon, with a pair bootstrap
    def coefs(d):
        w1, w3 = (d.wave == "W1").astype(float), (d.wave == "W3").astype(float)
        t = d["t_years"]
        x = d.assign(d_w3_t=d.d_w3 * t, d_w1_t=d.d_w1 * t, w1_t=w1 * t, w3_t=w3 * t, W1=w1, W3=w3,
                     log_na=np.log(d.n_authors), log_nt=np.log(d.n_tokens))
        b, V = E.fe_fit(x, "mwi_z", ["d_w3", "d_w3_t"] + ["d_w1", "W1", "W3", "log_na", "structured", "log_nt",
                                                              "d_w1_t", "w1_t", "w3_t"])
        return b[0], b[1], V
    a, s, V = coefs(df)
    pairs = df["pair"].unique()
    g = {k: v for k, v in df.groupby("pair")}
    cross = []
    for _ in range(NBOOT):
        pick = rng.choice(pairs, len(pairs))
        d = pd.concat([g[k].assign(pair=f"{k}_{i}", author=g[k]["author"] + f"_{i}") for i, k in enumerate(pick)],
                      ignore_index=True)
        aa, ss, _ = coefs(d)
        cross.append(-aa / ss if ss < 0 else np.inf)
    cross = np.array(cross)
    fin = cross[np.isfinite(cross)]
    to_date = lambda y: (dt.date(2024, 1, 1) + dt.timedelta(days=float(y) * 365.25)).isoformat()
    out["mwi_line"] = {"excess_jan2024": float(a), "slope_per_year": float(s), "se_slope": float(math.sqrt(V[1, 1])),
                       "zero_crossing": to_date(-a / s) if s < 0 else None,
                       "zero_crossing_ci": [to_date(np.percentile(fin, 2.5)), to_date(np.percentile(fin, 97.5))],
                       "share_boot_declining": float(np.mean(np.isfinite(cross))), "n_boot": NBOOT,
                       "half_life_years": float(-0.5 * a / s) if s < 0 else None}
    # robustness of the trend to search phrasing (phrasing-specific wave and discloser effects)
    q = pd.get_dummies(df["query"]).astype(float)
    w1, w3 = (df.wave == "W1").astype(float), (df.wave == "W3").astype(float)
    x = df.assign(d_w3_t=df.d_w3 * df.t_years, d_w1_t=df.d_w1 * df.t_years, w1_t=w1 * df.t_years,
                  w3_t=w3 * df.t_years, W1=w1, W3=w3, log_na=np.log(df.n_authors), log_nt=np.log(df.n_tokens))
    extra = []
    for c in q.columns[1:]:
        for nm, v in ((f"q_{c}_w1", w1 * q[c]), (f"q_{c}_w3", w3 * q[c]), (f"q_{c}_d3", df.d_w3 * q[c]),
                      (f"q_{c}_d1", df.d_w1 * q[c])):
            x[nm] = v; extra.append(nm)
    out["trend_phrasing_adjusted"] = {}
    for y, _ in OUTC:
        b, V = E.fe_fit(x.dropna(subset=[y + "_z"]), y + "_z", ["d_w3_t", "d_w3", "d_w1", "W1", "W3", "log_na", "structured", "log_nt",
                                      "d_w1_t", "w1_t", "w3_t"] + extra)
        se = math.sqrt(V[0, 0])
        out["trend_phrasing_adjusted"][y] = {"b": float(b[0]), "lo": float(b[0] - 1.96 * se), "hi": float(b[0] + 1.96 * se),
                                             "p": float(2 * norm.sf(abs(b[0] / se)))}
    return out


# ---------------------------------------------------------------- B  temporal transfer of detectors
def transfer(df):
    pre = df[df.wave != "W3"]
    w3 = df[df.wave == "W3"].groupby(["pair", "group"]).first().reset_index()
    w3 = w3[w3["pair"].isin(pre["pair"].unique())]
    vec = TfidfVectorizer(token_pattern=TOKEN.pattern, lowercase=True, min_df=5, sublinear_tf=True)
    vec.fit(pd.concat([w3["text"], pre["text"]]))
    Xw3 = vec.transform(w3["text"])
    # author baseline: mean tf-idf vector of the author's pre-LLM abstracts
    P = vec.transform(pre["text"])
    keys = list(zip(pre["pair"], pre["group"]))
    idx = {}
    for i, k in enumerate(keys):
        idx.setdefault(k, []).append(i)
    base = np.vstack([np.asarray(P[idx[(p, g)]].mean(0)).ravel() for p, g in zip(w3["pair"], w3["group"])])
    Xdiff = Xw3.toarray() - base
    st = df[df.wave != "W3"].groupby(["pair", "group"])[["mwi", "sent_len", "sent_sd", "mtld", "flesch"]].agg(["mean", "std"])
    Xsty = np.column_stack([(w3[o].to_numpy() - st[(o, "mean")].reindex(list(zip(w3.pair, w3.group))).to_numpy()) /
                            st[(o, "std")].reindex(list(zip(w3.pair, w3.group))).replace(0, 1).fillna(1).to_numpy()
                            for o in ("mwi", "sent_len", "sent_sd", "mtld", "flesch")])
    Xsty = (Xsty - Xsty.mean(0)) / Xsty.std(0)
    y = w3["group"].to_numpy()
    pr = w3["pair"].to_numpy()
    early = (w3["cohort"] == "2024-25").to_numpy()
    feats = {"text": Xw3, "text_minus_own_baseline": Xdiff, "five_style_features_baseline": Xsty}

    def auc_ci(yt, sc, prt):
        up = np.unique(prt)
        rows = {k: np.where(prt == k)[0] for k in up}
        bs = []
        for _ in range(NBOOT):
            ii = np.concatenate([rows[k] for k in rng.choice(up, len(up))])
            bs.append(roc_auc_score(yt[ii], sc[ii]))
        return {"auc": float(roc_auc_score(yt, sc)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                "n_pairs": int(len(up))}

    out = {}
    for fname, X in feats.items():
        out[fname] = {}
        for tr_name, tr in (("2024-25", early), ("2026", ~early)):
            for te_name, te in (("2024-25", early), ("2026", ~early)):
                if tr_name == te_name:  # within period: cross-validation keeping pairs together
                    sc = np.zeros(int(tr.sum()))
                    Xt, yt, pt = X[tr], y[tr], pr[tr]
                    for a, b in GroupKFold(5).split(Xt, yt, pt):
                        sc[b] = LogisticRegression(max_iter=2000, C=1.0).fit(Xt[a], yt[a]).predict_proba(Xt[b])[:, 1]
                    out[fname][f"{tr_name}->{te_name}"] = auc_ci(yt, sc, pt)
                else:
                    m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr])
                    out[fname][f"{tr_name}->{te_name}"] = auc_ci(y[te], m.predict_proba(X[te])[:, 1], pr[te])
    return out


# ---------------------------------------------------------------- C  placebo for the rising-word lexicon
def rising_placebo(df, rising):
    late = df[df["cohort"] == "2026"].reset_index(drop=True)
    cand = set(pd.read_csv("../marker_list/candidates.csv")["word"]) | set(rising)
    d25, n25 = {}, 0
    with gzip.open("../marker_list/corpus.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["year"] == 2025 and int(r["pmid"]) % 2 == 0:
                n25 += 1
                for t in set(re.findall(TOKEN, r["abstract"].lower())):
                    d25[t] = d25.get(t, 0) + 1
    cv = CountVectorizer(token_pattern=TOKEN.pattern, lowercase=True)
    X = cv.fit_transform(late["text"]); voc = cv.vocabulary_
    ntok = np.asarray(X.sum(1)).ravel()
    rate = lambda ws: 1000 * np.asarray(X[:, [voc[w] for w in ws if w in voc]].sum(1)).ravel() / ntok
    lg = lambda w: np.log10(d25.get(w, 0.5) / n25)
    words = [w for w in rising if w in voc]
    bins = np.quantile([lg(w) for w in words], np.linspace(0, 1, 11))
    bin_of = lambda w: int(np.clip(np.searchsorted(bins, lg(w)) - 1, 0, 9))
    pool = sorted(w for w in voc if w not in cand and w.isalpha() and d25.get(w, 0) >= 5)
    by_bin = {b: [w for w in pool if bin_of(w) == b] for b in range(10)}
    need = pd.Series([bin_of(w) for w in words]).value_counts().to_dict()
    f = E.Fast(late)
    ref = ((late.group == 0) & (late.wave == "W2")).to_numpy()
    z = lambda r: (r - r[ref].mean()) / r[ref].std()
    obs = f.beta(z(rate(words)), late.d_w1, late.d_w3)[1]
    draws = []
    for _ in range(NPLACEBO):
        ws = [w for b, k in need.items() for w in rng.choice(by_bin[b], size=min(k, len(by_bin[b])), replace=False)]
        draws.append(f.beta(z(rate(ws)), late.d_w1, late.d_w3)[1])
    draws = np.array(draws)
    return {"obs_beta3_2026": float(obs), "placebo_mean": float(draws.mean()), "placebo_p95": float(np.percentile(draws, 95)),
            "share_ge_obs": float(np.mean(draws >= obs)), "n": int(len(draws)), "n_pairs": int(late.pair.nunique())}


# ---------------------------------------------------------------- D  breakdown value
def breakdown(df):
    out = {}
    for name, s in (("main 2024-25", df[(df["sample"] == "main") & (df.early == 1)]),
                    ("replication 2024-25", df[df["sample"] == "rep2425"]),
                    ("all 2024-25", df[df.early == 1])):
        e = E.fit(s)
        mx = max(abs(e["d_w1"]["lo"]), abs(e["d_w1"]["hi"]))
        out[name] = {"b3_lo": e["d_w3"]["lo"], "max_abs_b1": mx, "M_breakdown": e["d_w3"]["lo"] / mx}
    return out


def tables(res):
    T = res["transfer"]
    lab = {"text": "Words of the 2024--26 abstract", "text_minus_own_baseline": "Words, minus own 2014--21 profile",
           "five_style_features_baseline": "Five style features, own baseline"}
    f = lambda e: f"{e['auc']:.2f} [{e['lo']:.2f}, {e['hi']:.2f}]"
    L = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{Trained on 2024--25} & \\multicolumn{2}{c}{Trained on 2026} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
         "Features & Tested on 2024--25 & Tested on 2026 & Tested on 2026 & Tested on 2024--25 \\\\", "\\midrule"]
    for k, name in lab.items():
        t = T[k]
        L.append(f"{name} & {f(t['2024-25->2024-25'])} & {f(t['2024-25->2026'])} & {f(t['2026->2026'])} & {f(t['2026->2024-25'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open("../paper/tables/tab_transfer.tex", "w").write("\n".join(L))


def main():
    df = load()
    L26 = json.load(open("../marker_list/lexicon_2026.json"))
    df["rising"] = E.lex_rate(df, L26["rising_2026"])
    for y, _ in OUTC:
        df[y + "_z"] = zscore_on_main(df, y) if y == "rising" else df[y + "_z"]
    res = {"n_pairs": int(df.pair.nunique()), "pairs_by_sample": df.groupby("sample").pair.nunique().to_dict()}
    res["decay"] = decay(df)
    res["transfer"] = transfer(df)
    res["rising_placebo"] = rising_placebo(df, L26["rising_2026"])
    res["breakdown"] = breakdown(df)
    json.dump(res, open("impact_results.json", "w"), indent=1, default=float)
    tables(res)
    print(json.dumps({k: res[k] for k in ("n_pairs", "pairs_by_sample", "breakdown", "rising_placebo")}, indent=1, default=float))
    print(json.dumps(res["decay"]["mwi_line"], indent=1))
    for p, v in res["decay"]["periods"].items():
        print(p, v["n_pairs"], {y: round(v[y]["b"], 2) for y, _ in OUTC})
    print({y: (round(v["slope_per_year"]["b"], 2), round(v["slope_per_year"]["p"], 4)) for y, v in res["decay"]["trend"].items()})
    for k, v in res["transfer"].items():
        print(k, {a: round(b["auc"], 3) for a, b in v.items()})


if __name__ == "__main__":
    main()
