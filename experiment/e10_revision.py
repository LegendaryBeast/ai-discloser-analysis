"""E10 (revision): every estimate reported in the revised manuscript, on the audited analysis sample.
Inputs: dataset.csv, discloser/control units, e8_verify.json, audit_v2.json, replication datasets (if present).
Outputs: revision_results.json, revision_metrics.csv and LaTeX tables in ../paper/tables/."""
import json, math, os, re, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from coding import code
from textmetrics import metrics, TOKEN, FW_INDEX
import e8_verify as ev
from e9_replication import disclosure2, SENT2, AI_ANY, NON_USE

NONUSE_SCOPE = re.compile(r"manuscript|this (?:work|study|article|paper)|writing|preparation", re.I)

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)

# LAPACK's divide-and-conquer SVD (gesdd) occasionally fails to converge on these large dummy-variable designs;
# fall back to the slower but more robust gesvd driver. Estimates are unaffected.
import scipy.linalg
_np_svd = np.linalg.svd


def _robust_svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    try:
        return _np_svd(a, full_matrices=full_matrices, compute_uv=compute_uv, hermitian=hermitian)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv, lapack_driver="gesvd")


np.linalg.svd = _robust_svd
LEX = json.load(open("../marker_list/marker_list_v1.json"))
MARK, CONTENT = LEX["words"], LEX["excluded_content_words"]
TOP5 = json.load(open("../marker_list/validation.json"))["top5"]
STRICT = [w for w in MARK if float(LEX["word_stats"][w]["ratio2024"]) >= 2 and float(LEX["word_stats"][w]["ratio2025"]) >= 2]
# Forms whose everyday or domain senses are common in abstracts (sensitivity only)
POLYSEMOUS = ["hold", "holds", "lies", "marks", "stays", "stands", "serves", "serving", "presents", "pose", "posed",
              "poses", "posing", "offer", "offers", "offering", "linked", "raising", "promise", "creation", "foundation"]
ENGLISH = {"US", "GB", "CA", "AU", "NZ", "IE"}
OUT = ["mwi", "sent_len", "sent_sd", "mtld", "flesch", "drift"]
LABEL = {"mwi": "Marker-word index", "sent_len": "Sentence length", "sent_sd": "Sentence-length SD",
         "mtld": "Lexical diversity (MTLD)", "flesch": "Flesch reading ease", "drift": "Function-word self-drift"}
BASE = "C(wave) + np.log(n_authors) + structured + np.log(n_tokens)"
TAB = "../paper/tables"
os.makedirs(TAB, exist_ok=True)


# ---------------------------------------------------------------- sample
def build(df):
    """Per-text metrics, complete pairs only (E5 rules)."""
    df = df[df["text"].apply(lambda t: len(TOKEN.findall(t.lower())) >= 80)].reset_index(drop=True)
    M = df["text"].apply(metrics)
    for k in ("n_tokens", "mwi", "sent_len", "sent_sd", "mtld", "flesch"):
        df[k] = [m[k] for m in M]
    df["fw"] = [m["fw"] for m in M]
    ok = df.groupby(["pair", "group"])["wave"].nunique().unstack().eq(3).all(axis=1)
    return df[df["pair"].isin(ok[ok].index)].reset_index(drop=True)


def add_drift(df):
    """Cosine distance of a text's function-word profile from the centroid of the author's other W1 texts."""
    F = np.vstack(df["fw"].to_numpy())
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    drift = np.full(len(df), np.nan)
    for a, idx in df.groupby("author").groups.items():
        idx = np.asarray(list(idx))
        w1 = idx[(df.loc[idx, "wave"] == "W1").to_numpy()]
        for i in idx:
            base = [j for j in w1 if j != i]
            if base:
                cen = F[base].mean(0)
                n = np.linalg.norm(cen)
                if n > 0 and np.linalg.norm(F[i]) > 0:
                    drift[i] = 1 - F[i] @ cen / n
    df["drift"] = drift
    return df


def lex_rate(df, words):
    ws = set(words)
    return df["text"].apply(lambda t: 1000 * sum(x in ws for x in TOKEN.findall(t.lower())) /
                            max(1, len(TOKEN.findall(t.lower()))))


def prep(df, ref=None):
    df = df.copy()
    df["d_w1"] = ((df["group"] == 1) & (df["wave"] == "W1")).astype(int)
    df["d_w3"] = ((df["group"] == 1) & (df["wave"] == "W3")).astype(int)
    df["w"] = 1 / df.groupby(["author", "wave"])["wave"].transform("size")
    ref = ref if ref is not None else {y: (df.loc[(df.group == 0) & (df.wave == "W2"), y].mean(),
                                           df.loc[(df.group == 0) & (df.wave == "W2"), y].std()) for y in OUT}
    for y in OUT:
        df[y + "_z"] = (df[y] - ref[y][0]) / ref[y][1]
    return df, ref


def wfit(model, **kw):
    """Fit with the default pseudo-inverse; fall back to QR if LAPACK's SVD fails to converge (full-rank designs)."""
    try:
        return model.fit(**kw)
    except np.linalg.LinAlgError:
        return model.fit(method="qr", **kw)


# ---------------------------------------------------------------- estimation
def fe_fit(data, y, xcols, weights=True, cluster="author"):
    """WLS with author fixed effects by weighted within-author demeaning (Frisch-Waugh-Lovell). Cluster-robust
    covariance with the same small-sample correction as statsmodels (G/(G-1) * (N-1)/(N-K), K incl. the fixed effects)
    and normal-based inference, so results equal the dummy-variable fit."""
    A = pd.factorize(data["author"])[0]
    w = data["w"].to_numpy(float) if weights else np.ones(len(data))
    X = np.column_stack([data[c].to_numpy(float) for c in xcols])
    Y = data[y].to_numpy(float)
    ws = np.bincount(A, weights=w)
    dm = lambda M: M - (np.vstack([np.bincount(A, weights=w * M[:, j]) for j in range(M.shape[1])]).T / ws[:, None])[A]
    Xd, Yd = dm(X), dm(Y[:, None]).ravel()
    sw = np.sqrt(w)
    Xs, Ys = Xd * sw[:, None], Yd * sw
    bread = np.linalg.inv(Xs.T @ Xs)
    b = bread @ Xs.T @ Ys
    e = Ys - Xs @ b
    G = pd.factorize(data[cluster])[0]
    S = np.zeros((G.max() + 1, X.shape[1]))
    np.add.at(S, G, Xs * e[:, None])
    n, k, g = len(Y), X.shape[1] + A.max() + 1, G.max() + 1
    V = bread @ (S.T @ S) @ bread * (g / (g - 1)) * ((n - 1) / (n - k))
    return b, V


FAST_X = ["d_w1", "d_w3", "W1", "W3", "log_na", "structured", "log_nt"]


def fit(data, y="mwi_z", extra="", weights=True, fe="C(author)", cluster="author", terms=("d_w1", "d_w3")):
    data = data.dropna(subset=[y])
    if fe == "C(author)" and extra in ("", " + C(wave):english_author") and terms == ("d_w1", "d_w3"):
        d = data.assign(W1=(data["wave"] == "W1").astype(float), W3=(data["wave"] == "W3").astype(float),
                        log_na=np.log(data["n_authors"]), log_nt=np.log(data["n_tokens"]))
        xc = list(FAST_X)
        if extra:
            d = d.assign(W1e=d["W1"] * d["english_author"], W3e=d["W3"] * d["english_author"])
            xc += ["W1e", "W3e"]
        b, V = fe_fit(d, y, xc, weights, cluster)
        out = {}
        for i, k in enumerate(("d_w1", "d_w3")):
            se = math.sqrt(V[i, i])
            out[k] = {"b": float(b[i]), "se": se, "lo": float(b[i] - 1.959964 * se), "hi": float(b[i] + 1.959964 * se),
                      "p": float(2 * norm.sf(abs(b[i] / se)))}
        out["n_pairs"], out["n_texts"] = int(data["pair"].nunique()), int(len(data))
        tb = b[0] + b[1]
        tse = math.sqrt(V[0, 0] + V[1, 1] + 2 * V[0, 1])
        out["trend_adj"] = {"b": float(tb), "lo": float(tb - 1.96 * tse), "hi": float(tb + 1.96 * tse),
                            "p": float(2 * norm.sf(abs(tb / tse)))}
        return out
    return fit_sm(data, y, extra, weights, fe, cluster, terms)


def fit_sm(data, y="mwi_z", extra="", weights=True, fe="C(author)", cluster="author", terms=("d_w1", "d_w3")):
    data = data.dropna(subset=[y])
    f = f"{y} ~ d_w1 + d_w3 + {BASE} + {fe}{extra}"
    mod = smf.wls(f, data=data, weights=data["w"]) if weights else smf.ols(f, data=data)
    r = wfit(mod, cov_type="cluster", cov_kwds={"groups": pd.factorize(data[cluster])[0]})
    ci = r.conf_int()
    out = {k: {"b": float(r.params[k]), "se": float(r.bse[k]), "lo": float(ci.loc[k, 0]), "hi": float(ci.loc[k, 1]),
               "p": float(r.pvalues[k])} for k in terms}
    out["n_pairs"] = int(data["pair"].nunique())
    out["n_texts"] = int(len(data))
    if "d_w1" in terms and "d_w3" in terms:
        c = r.cov_params()
        b = r.params["d_w3"] + r.params["d_w1"]
        se = math.sqrt(c.loc["d_w3", "d_w3"] + c.loc["d_w1", "d_w1"] + 2 * c.loc["d_w3", "d_w1"])
        out["trend_adj"] = {"b": float(b), "lo": float(b - 1.96 * se), "hi": float(b + 1.96 * se),
                            "p": float(2 * norm.sf(abs(b / se)))}
    return out


def fit_mod(data, mod, y="mwi_z"):
    """Interaction of the discloser effects with a pair-level binary moderator."""
    w1, w3 = (data["wave"] == "W1").astype(int), (data["wave"] == "W3").astype(int)
    d = data.assign(d_w1_M=data["d_w1"] * mod, d_w3_M=data["d_w3"] * mod, w1_M=w1 * mod, w3_M=w3 * mod,
                    W1=w1.astype(float), W3=w3.astype(float), log_na=np.log(data["n_authors"]),
                    log_nt=np.log(data["n_tokens"])).dropna(subset=[y])
    b, V = fe_fit(d, y, ["d_w3_M"] + FAST_X + ["d_w1_M", "w1_M", "w3_M"])
    se = math.sqrt(V[0, 0])
    return {"b": float(b[0]), "se": se, "lo": float(b[0] - 1.959964 * se), "hi": float(b[0] + 1.959964 * se),
            "p": float(2 * norm.sf(abs(b[0] / se)))}


def tost(est, margin=0.10):
    """Two one-sided tests: is |beta| < margin? Returns the larger one-sided p and the 90% CI."""
    b, se = est["b"], est["se"]
    p = max(norm.sf((b + margin) / se), norm.cdf((b - margin) / se))
    return {"margin": margin, "p": float(p), "ci90": [float(b - 1.645 * se), float(b + 1.645 * se)]}


def holm(ps):
    order = np.argsort(ps)
    adj, m, run = np.empty(len(ps)), len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


class Fast:
    """Weighted within-author estimator of (beta1, beta3) for permutation and placebo loops."""

    def __init__(self, df):
        self.A = pd.factorize(df["author"])[0]
        self.W = df["w"].to_numpy()
        wave = pd.get_dummies(df["wave"])[["W1", "W3"]].to_numpy(float)
        cov = np.column_stack([wave, np.log(df["n_authors"]), df["structured"], np.log(df["n_tokens"])])
        self.C = self.dm(cov)
        self.sw = np.sqrt(self.W)

    def dm(self, M):
        M = np.atleast_2d(np.asarray(M, float).T).T
        s = np.zeros((self.A.max() + 1, M.shape[1]))
        np.add.at(s, self.A, M * self.W[:, None])
        return M - (s / np.bincount(self.A, weights=self.W)[:, None])[self.A]

    def beta(self, y, d1, d3):
        Z = np.column_stack([self.dm(np.column_stack([d1, d3])), self.C]) * self.sw[:, None]
        return np.linalg.lstsq(Z, self.dm(y).ravel() * self.sw, rcond=None)[0][:2]


def randomisation(df, n=int(os.environ.get("NPERM", 5000))):
    f = Fast(df)
    y = df["mwi_z"].to_numpy()
    obs = f.beta(y, df["d_w1"], df["d_w3"])
    pid, grp = df["pair"].to_numpy(), df["group"].to_numpy()
    up = np.unique(pid)
    w1, w3 = (df["wave"] == "W1").to_numpy(), (df["wave"] == "W3").to_numpy()
    pos = np.searchsorted(up, pid)
    perm = []
    for _ in range(n):
        flip = rng.integers(0, 2, len(up))[pos]
        g = np.where(flip == 1, 1 - grp, grp)
        perm.append(f.beta(y, (g == 1) & w1, (g == 1) & w3))
    perm = np.array(perm)
    return {"b3": float(obs[1]), "p3": float(np.mean(np.abs(perm[:, 1]) >= abs(obs[1]))),
            "b1": float(obs[0]), "p1": float(np.mean(np.abs(perm[:, 0]) >= abs(obs[0]))), "n_perm": n}


def author_shifts(df, col="mwi"):
    aw = df.groupby(["pair", "group", "wave"])[col].mean().unstack()
    aw["d32"], aw["d21"] = aw["W3"] - aw["W2"], aw["W2"] - aw["W1"]
    return aw.reset_index()


def pi_hat(p):
    null = p["d21"].mean()
    return (p.loc[p.group == 0, "d32"].mean() - null) / (p.loc[p.group == 1, "d32"].mean() - null)


def pi_boot(df, n=int(os.environ.get("NBOOT", 2000))):
    p = author_shifts(df)
    pairs = p["pair"].unique()
    w = p.pivot_table(index="pair", columns="group", values=["d32", "d21"]).loc[pairs]
    c32, d32 = w[("d32", 0)].to_numpy(), w[("d32", 1)].to_numpy()
    d21 = (w[("d21", 0)] + w[("d21", 1)]).to_numpy() / 2
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(pairs), len(pairs))
        null = d21[i].mean()
        bs.append((c32[i].mean() - null) / (d32[i].mean() - null))
    return {"pi": float(pi_hat(p)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "d32_disc": float(p.loc[p.group == 1, "d32"].mean()), "d32_ctrl": float(p.loc[p.group == 0, "d32"].mean()),
            "d21": float(p["d21"].mean()), "n_pairs": int(len(pairs))}


def detection(df):
    outs = ["mwi", "sent_len", "sent_sd", "mtld", "flesch"]
    pre = df[df.wave != "W3"].groupby(["pair", "group"])[outs].agg(["mean", "std"])
    w2 = df[df.wave == "W2"].groupby(["pair", "group"])[outs].mean()
    t3 = df[df.wave == "W3"].groupby(["pair", "group"])[outs].mean()
    yl = t3.index.get_level_values("group").to_numpy()
    pr = t3.index.get_level_values("pair").to_numpy()
    sd = lambda o: pre[(o, "std")].reindex(t3.index).replace(0, np.nan).fillna(1).to_numpy()
    raw = t3["mwi"].to_numpy()
    diff = raw - w2["mwi"].reindex(t3.index).to_numpy()
    zs = (raw - pre[("mwi", "mean")].reindex(t3.index).to_numpy()) / sd("mwi")
    Xr = t3[outs].to_numpy()
    Xn = np.column_stack([(t3[o].to_numpy() - pre[(o, "mean")].reindex(t3.index).to_numpy()) / sd(o) for o in outs])
    st = lambda M: (M - M.mean(0)) / M.std(0)
    cv = GroupKFold(5)
    p_r = cross_val_predict(LogisticRegression(max_iter=1000), st(Xr), yl, cv=cv, groups=pr, method="predict_proba")[:, 1]
    p_n = cross_val_predict(LogisticRegression(max_iter=1000), st(Xn), yl, cv=cv, groups=pr, method="predict_proba")[:, 1]
    up = np.unique(pr)
    rows = {k: np.where(pr == k)[0] for k in up}

    def auc(score):
        bs = []
        for _ in range(int(os.environ.get("NBOOT", 2000))):
            idx = np.concatenate([rows[k] for k in rng.choice(up, len(up))])
            bs.append(roc_auc_score(yl[idx], score[idx]))
        return {"auc": float(roc_auc_score(yl, score)), "lo": float(np.percentile(bs, 2.5)),
                "hi": float(np.percentile(bs, 97.5))}
    return {"mwi_raw": auc(raw), "mwi_minus_w2": auc(diff), "mwi_z_own": auc(zs), "five_raw_cv": auc(p_r),
            "five_baseline_cv": auc(p_n), "n_pairs": int(len(up))}


# ---------------------------------------------------------------- main
def main():
    res = {}
    DJ = json.load(open("disclosers.json"))
    D = DJ["papers"]
    pidx = {p["doi"]: i for i, p in enumerate(D)}
    U = {u["doi"]: u for u in json.load(open("discloser_units.json")) if u["eligible"]}
    CU = [c for c in json.load(open("control_units.json")) if not c.get("unmatched")]
    pairs = [(U[c["matched_to"]], c) for c in CU if c["matched_to"] in U]
    V = json.load(open(os.environ.get("E8", "e8_verify.json")))
    AUD = json.load(open("audit_v2.json"))
    df = build(pd.read_csv("dataset.csv"))
    df = add_drift(df)
    fin = sorted(df["pair"].unique())

    # pair-level information and exclusion flags
    seen, P = set(), {}
    for k, (d, c) in enumerate(pairs):
        cd = code(d["disclosure"])
        cv = dict(V["controls"].get(str(k), {}))
        if k in fin and cv.get("fulltext"):
            # re-read the cached control full text with the audited instrument (E9) and flag declarations of non-use
            text, _ = ev.fulltext(cv["pmcid"])
            sents = SENT2.split(text)
            e8s = cv.get("disclosure")
            cv["disclosure"] = disclosure2(text) or (e8s if e8s and not NON_USE.search(e8s) else None)
            cv["nonuse"] = any(AI_ANY.search(s) and NON_USE.search(s) and NONUSE_SCOPE.search(s) for s in sents)
        i = pidx[d["doi"]]
        dup = c["anchor"]["id"] in seen
        if k in fin:
            seen.add(c["anchor"]["id"])
        P[k] = {"paper_index": i, "query": d["query"], "year": min(int(str(d["year"])), 2026),
                "frame": "pilot" if i < 160 else "ncbi", "fp": str(i) in AUD["excluded"],
                "ambiguous": str(i) in AUD["ambiguous"], "dup_ctrl": dup,
                "ctrl_screened": bool(cv.get("fulltext")), "ctrl_discloses": bool(cv.get("disclosure")),
                "ctrl_any_ai": bool(cv.get("ai_mentions")), "ctrl_nonuse": bool(cv.get("nonuse")),
                "ctrl_source": "epmc" if c.get("pmcid") else "openalex_fallback",
                "disc_type": V["discloser_article_type"].get(str(k)), "ctrl_type": cv.get("article_type"),
                "english": (d["anchor"]["countries"] or ["NA"])[0] in ENGLISH,
                "chatgpt": cd["chatgpt"], "non_llm_only": cd["non_llm_only"], "intensity": cd["intensity"],
                "drafting": cd["intensity"] == 3, "pass2": c.get("pass", 1) == 2,
                "first_share": float(np.mean([w["position"] == "first" for w in d["history"]])) if d["history"] else 0.}
    P = pd.DataFrame.from_dict(P, orient="index").loc[fin]
    P["cohort"] = np.where(P["year"] <= 2025, "2024-25", "2026")
    flow = {"complete_pairs": len(P), "fp_disclosure": int(P.fp.sum()), "dup_control_author": int((P.dup_ctrl & ~P.fp).sum()),
            "control_discloses": int((P.ctrl_discloses & ~P.fp & ~P.dup_ctrl).sum())}
    keep = P.index[~P.fp & ~P.dup_ctrl & ~P.ctrl_discloses]
    flow["analysis_pairs"] = len(keep)
    flow["controls_screened"] = int(P.loc[keep, "ctrl_screened"].sum())
    flow["controls_openalex_fallback"] = int((P.loc[keep, "ctrl_source"] == "openalex_fallback").sum())
    flow["controls_nonuse_statement"] = int(P.loc[keep, "ctrl_nonuse"].sum())
    flow["by_cohort"] = P.loc[keep, "cohort"].value_counts().to_dict()
    flow["cohort_by_query"] = pd.crosstab(P.loc[keep, "query"], P.loc[keep, "cohort"]).to_dict()
    flow["cohort_by_frame"] = pd.crosstab(P.loc[keep, "frame"], P.loc[keep, "cohort"]).to_dict()
    flow["disc_type"] = P.loc[keep, "disc_type"].fillna("unknown").value_counts().to_dict()
    flow["ctrl_type"] = P.loc[keep, "ctrl_type"].fillna("unknown").value_counts().to_dict()
    res["flow"] = flow
    units_all = json.load(open("discloser_units.json"))
    res["collection_flow"] = {"pool": DJ["pool_size"], "screened": DJ["screened"], "detected": len(D),
                              "unique_anchor_authors": len(units_all), "eligible": sum(u["eligible"] for u in units_all),
                              "matched": len(CU)}
    P.to_csv("revision_pairs.csv")

    full = df.copy()
    df, ref = prep(df[df["pair"].isin(keep)].reset_index(drop=True))
    res["ref"] = {y: list(map(float, v)) for y, v in ref.items()}
    res["n_texts"] = int(len(df))
    res["texts_by_wave"] = df["wave"].value_counts().to_dict()
    res["median_texts_author_wave"] = df.groupby(["author", "wave"]).size().groupby("wave").median().to_dict()
    df["cohort"] = df["pair"].map(P["cohort"])
    df["english_author"] = df["country"].isin(ENGLISH).astype(int)
    df["pairwave"] = df["pair"].astype(str) + "_" + df["wave"]
    coh = (df["cohort"] == "2024-25").astype(int)
    sub = {"pooled": df, "2024-25": df[coh == 1], "2026": df[coh == 0]}

    # balance (standardised mean differences)
    bal = []
    for k in keep:
        d, c = pairs[k]
        for g, u in ((1, d), (0, c)):
            h = u["history"]
            bal.append({"pair": k, "group": g, "team": u["n_authors"],
                        "english": (u["anchor"]["countries"] or ["NA"])[0] in ENGLISH,
                        "corresponding": u["anchor"]["rule"] == "corresponding",
                        "w1": sum(w["year"] < 2017 for w in h), "w2": sum(w["year"] >= 2019 for w in h),
                        "first_share": float(np.mean([w["position"] == "first" for w in h]))})
    B = pd.DataFrame(bal)
    mw = df.groupby(["pair", "group", "wave"])["mwi"].mean().unstack().reset_index()
    B = B.merge(mw[["pair", "group", "W1", "W2"]], on=["pair", "group"])
    res["balance"] = {}
    for v in ("team", "english", "corresponding", "w1", "w2", "first_share", "W1", "W2"):
        a, b = B.loc[B.group == 1, v].astype(float), B.loc[B.group == 0, v].astype(float)
        res["balance"][v] = {"disc_mean": float(a.mean()), "ctrl_mean": float(b.mean()), "disc_median": float(a.median()),
                             "ctrl_median": float(b.median()),
                             "smd": float((a.mean() - b.mean()) / math.sqrt((a.var() + b.var()) / 2))}

    # descriptive author-level means by group, wave and cohort
    aw = df.groupby(["cohort", "pair", "group", "wave"])["mwi"].mean().reset_index()
    res["mwi_means"] = {}
    for (cg, g, w), s in pd.concat([aw.assign(cohort="pooled"), aw]).groupby(["cohort", "group", "wave"])["mwi"]:
        res["mwi_means"][f"{cg}|{g}|{w}"] = {"mean": float(s.mean()), "se": float(s.std() / math.sqrt(len(s))), "n": int(len(s))}
    w3 = df[df.wave == "W3"]
    res["share_any_marker_w3"] = {str(g): float((s["mwi"] > 0).mean()) for g, s in w3.groupby("group")}

    # main event study, all outcomes, pooled and by cohort
    res["main"] = {}
    for name, s in sub.items():
        res["main"][name] = {y: fit(s, y + "_z") for y in OUT}
        res["main"][name]["mwi_pair_cluster"] = fit(s, "mwi_z", cluster="pair")
        res["main"][name]["mwi_raw_units"] = fit(s, "mwi")
    res["holm_secondary_pooled"] = dict(zip(OUT[1:], holm([res["main"]["pooled"][y]["d_w3"]["p"] for y in OUT[1:]]).tolist()))
    res["holm_secondary_2024_25"] = dict(zip(OUT[1:], holm([res["main"]["2024-25"][y]["d_w3"]["p"] for y in OUT[1:]]).tolist()))
    res["tost_beta1"] = {name: tost(res["main"][name]["mwi"]["d_w1"]) for name in sub}
    res["mde80"] = {name: 2.80 * res["main"][name]["mwi"]["d_w3"]["se"] for name in sub}
    res["cohort_interaction"] = {y: fit_mod(df, coh, y + "_z") for y in OUT}

    # cohort contrast within query and within sampling frame; joint model with query and tool
    qc = (df["pair"].map(P["query"]) == "chatgpt_language").astype(int)
    res["by_query_cohort"] = {}
    for qn, qm in (("chatgpt_phrasing", qc == 1), ("other_phrasings", qc == 0)):
        for cn, cm in (("2024-25", coh == 1), ("2026", coh == 0)):
            res["by_query_cohort"][f"{qn}|{cn}"] = fit(df[qm & cm])["d_w3"] | {"n_pairs": int(df[qm & cm].pair.nunique())}
    fr = (df["pair"].map(P["frame"]) == "pilot").astype(int)
    res["by_frame_cohort"] = {}
    for fn, fm in (("pilot", fr == 1), ("ncbi", fr == 0)):
        for cn, cm in (("2024-25", coh == 1), ("2026", coh == 0)):
            res["by_frame_cohort"][f"{fn}|{cn}"] = fit(df[fm & cm])["d_w3"] | {"n_pairs": int(df[fm & cm].pair.nunique())}
    chat = df["pair"].map(P["chatgpt"]).astype(int)
    W1, W3 = (df["wave"] == "W1").astype(int), (df["wave"] == "W3").astype(int)
    j = df.assign(a=df.d_w3 * coh, b=df.d_w3 * qc, t=df.d_w3 * chat, a1=df.d_w1 * coh, b1=df.d_w1 * qc, t1=df.d_w1 * chat,
                  c1=W1 * coh, c3=W3 * coh, q1=W1 * qc, q3=W3 * qc, h1=W1 * chat, h3=W3 * chat)
    r = smf.wls(f"mwi_z ~ d_w1 + d_w3 + a + b + t + a1 + b1 + t1 + c1 + c3 + q1 + q3 + h1 + h3 + {BASE} + C(author)",
                data=j, weights=j["w"])
    r = wfit(r, cov_type="cluster", cov_kwds={"groups": pd.factorize(j["author"])[0]})
    res["joint_moderators"] = {k: {"b": float(r.params[v]), "p": float(r.pvalues[v]), "lo": float(r.conf_int().loc[v, 0]),
                                   "hi": float(r.conf_int().loc[v, 1])}
                               for k, v in (("cohort_2024_25", "a"), ("chatgpt_phrasing_query", "b"), ("chatgpt_named", "t"))}
    res["by_year"] = {str(y): fit(df[df["pair"].map(P["year"]) == y])["d_w3"] | {"n_pairs": int((P.loc[keep, "year"] == y).sum())}
                      for y in (2024, 2025, 2026)}

    # robustness table, pooled and by cohort
    df["mwi_top5_z"] = (lambda s: (s - s[(df.group == 0) & (df.wave == "W2")].mean()) / s[(df.group == 0) & (df.wave == "W2")].std())(lex_rate(df, [w for w in MARK if w not in TOP5]))
    zr = lambda s: (s - s[(df.group == 0) & (df.wave == "W2")].mean()) / s[(df.group == 0) & (df.wave == "W2")].std()
    df["mwi_strict_z"] = zr(lex_rate(df, STRICT))
    df["mwi_all347_z"] = zr(lex_rate(df, MARK + CONTENT))
    df["mwi_nopoly_z"] = zr(lex_rate(df, [w for w in MARK if w not in POLYSEMOUS]))
    df["content_z"] = zr(lex_rate(df, CONTENT))
    lo_, hi_ = df["mwi"].quantile([0.01, 0.99])
    df["mwi_wins_z"] = zr(df["mwi"].clip(lo_, hi_))
    sub = {"pooled": df, "2024-25": df[coh == 1], "2026": df[coh == 0]}
    first_only = df.groupby("author").apply(lambda s: ((s.wave == "W1") & (s.position == "first")).any()
                                            and ((s.wave == "W2") & (s.position == "first")).any())
    strict_auth = set(first_only[first_only].index)
    sdf = df[df["author"].isin(strict_auth)]
    sp = sdf.groupby("pair")["group"].nunique()
    sdf = sdf[sdf["pair"].isin(sp[sp == 2].index)]
    sdf = sdf[sdf["wave"].eq("W3") | (sdf["position"] == "first")].copy()
    sdf["w"] = 1 / sdf.groupby(["author", "wave"])["wave"].transform("size")
    ra = lambda s: s["pair"].map(P["disc_type"]).eq("research-article") & s["pair"].map(P["ctrl_type"]).eq("research-article")
    specs = [("Main specification", lambda s: fit(s)),
             ("Pair-clustered SEs", lambda s: fit(s, cluster="pair")),
             ("+ Pair $\\times$ wave fixed effects", lambda s: fit(s, extra=" + C(pairwave)")),
             ("+ English-country $\\times$ wave", lambda s: fit(s, extra=" + C(wave):english_author")),
             ("Unweighted", lambda s: fit(s, weights=False)),
             ("Excl.\\ second-pass matches", lambda s: fit(s[~s["pair"].map(P["pass2"])])),
             ("Excl.\\ non-LLM tools only", lambda s: fit(s[~s["pair"].map(P["non_llm_only"])])),
             ("Excl.\\ ambiguous disclosures", lambda s: fit(s[~s["pair"].map(P["ambiguous"])])),
             ("Controls screened in full text", lambda s: fit(s[s["pair"].map(P["ctrl_screened"])])),
             ("Both index papers research articles", lambda s: fit(s[ra(s)])),
             ("Top-5 markers dropped", lambda s: fit(s, "mwi_top5_z")),
             (f"Strict lexicon ({len(STRICT)} words)", lambda s: fit(s, "mwi_strict_z")),
             (f"Polysemous forms dropped ({len(MARK) - len([w for w in MARK if w not in POLYSEMOUS])} words)", lambda s: fit(s, "mwi_nopoly_z")),
             ("All 347 candidates (no coder)", lambda s: fit(s, "mwi_all347_z")),
             ("Winsorised MWI (1st/99th pct.)", lambda s: fit(s, "mwi_wins_z")),
             ("First-authored pre-period texts", None)]
    rob = {}
    for lab, fn in specs:
        rob[lab] = {}
        for name, s in (("pooled", df), ("2024-25", df[coh == 1]), ("2026", df[coh == 0])):
            try:
                if fn is None:
                    ss = sdf if name == "pooled" else sdf[(sdf["cohort"] == "2024-25") == (name == "2024-25")]
                    e = fit(ss)
                else:
                    e = fn(s)
                rob[lab][name] = {"b1": e["d_w1"]["b"], **e["d_w3"], "n_pairs": e["n_pairs"]}
            except Exception as ex:  # empty subset
                print("robustness skipped:", lab, name, ex)
                rob[lab][name] = None
    res["robustness"] = rob

    # secondary (lexicon-free) outcomes under the checks most likely to matter for them
    sec = {}
    for lab, sel, kw in (("all", lambda s: s, {}), ("research_articles", lambda s: s[ra(s)], {}),
                         ("country_x_wave", lambda s: s, dict(extra=" + C(wave):english_author")),
                         ("pair_cluster", lambda s: s, dict(cluster="pair"))):
        sec[lab] = {}
        for name, s in (("pooled", df), ("2024-25", df[coh == 1]), ("2026", df[coh == 0])):
            ss = sel(s)
            sec[lab][name] = {y: fit(ss, y + "_z", **kw)["d_w3"] | {"n_pairs": int(ss.pair.nunique())} for y in OUT}
        if lab == "research_articles":
            r_ = df[ra(df)]
            sec[lab]["interaction"] = {y: fit_mod(r_, coh[ra(df)], y + "_z") for y in OUT}
    res["secondary_robustness"] = sec

    # randomisation inference
    res["randomisation"] = {name: randomisation(s.reset_index(drop=True)) for name, s in sub.items()}

    # placebo lexicons (frequency matched) and content words
    cand = set(pd.read_csv("../marker_list/candidates.csv")["word"])
    import gzip
    df22, n22 = {}, 0
    with gzip.open("../marker_list/corpus.jsonl.gz", "rt") as f:
        for line in f:
            rr = json.loads(line)
            if rr["year"] == 2022 and int(rr["pmid"]) % 2 == 0:
                n22 += 1
                for t in set(re.findall(TOKEN, rr["abstract"].lower())):
                    df22[t] = df22.get(t, 0) + 1
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(token_pattern=TOKEN.pattern, lowercase=True)
    X = vec.fit_transform(df["text"])
    voc = vec.vocabulary_
    ntok = np.asarray(X.sum(1)).ravel()
    rate = lambda words: 1000 * np.asarray(X[:, [voc[w] for w in words if w in voc]].sum(1)).ravel() / ntok
    p22 = {w: c / n22 for w, c in df22.items() if c >= 5}
    pool = sorted(w for w in p22 if w not in cand and w in voc and w.isalpha())  # sorted: set order varies by run
    lg = lambda w: np.log10(p22.get(w, 5 / n22))
    bins = np.quantile([lg(w) for w in MARK], np.linspace(0, 1, 11))
    bin_of = lambda w: int(np.clip(np.searchsorted(bins, lg(w)) - 1, 0, 9))
    by_bin = {b: [w for w in pool if bin_of(w) == b] for b in range(10)}
    need = pd.Series([bin_of(w) for w in MARK]).value_counts().to_dict()
    fast = {n: Fast(s.reset_index(drop=True)) for n, s in sub.items()}
    masks = {"pooled": np.ones(len(df), bool), "2024-25": (coh == 1).to_numpy(), "2026": (coh == 0).to_numpy()}

    def change(rt, m):
        t = pd.DataFrame({"a": df["author"][m], "w": df["wave"][m], "r": rt[m]}).groupby(["a", "w"])["r"].mean().unstack()
        return float((t["W3"] - t["W2"]).mean())

    def zrate(rt):
        ref_ = rt[((df.group == 0) & (df.wave == "W2")).to_numpy()]
        return (rt - ref_.mean()) / ref_.std()
    draws = []
    for _ in range(int(os.environ.get("NPLACEBO", 1000))):
        words = [w for b, k in need.items() for w in rng.choice(by_bin[b], size=k, replace=False)]
        rt = rate(words)
        z = zrate(rt)
        row = [change(rt, masks["pooled"])]
        for n, s in sub.items():
            m = masks[n]
            row.append(fast[n].beta(z[m], df["d_w1"][m], df["d_w3"][m])[1])
        draws.append(row)
    draws = np.array(draws)
    mr = rate(MARK)
    res["placebo_lexicons"] = {"n": len(draws), "marker_change32": change(mr, masks["pooled"]),
                               "marker_change21": float(author_shifts(df)["d21"].mean()),
                               "placebo_change32_mean": float(draws[:, 0].mean()),
                               "placebo_change32_p95": float(np.percentile(draws[:, 0], 95)),
                               "placebo_change32_max": float(draws[:, 0].max()),
                               "content_change32": change(rate(CONTENT), masks["pooled"])}
    for i, n in enumerate(sub, start=1):
        obs = res["main"][n]["mwi"]["d_w3"]["b"]
        res["placebo_lexicons"][f"beta3_{n}"] = {"obs": obs, "mean": float(draws[:, i].mean()),
                                                 "p95": float(np.percentile(draws[:, i], 95)),
                                                 "share_ge_obs": float(np.mean(draws[:, i] >= obs))}
    res["content_lexicon"] = {n: fit(s, "content_z")["d_w3"] for n, s in sub.items()}
    np.save("revision_placebo_draws.npy", draws)

    # lexicon refresh test (exploratory): lexicons derived from 2026 population excess words with the M2 rule
    L26 = json.load(open("../marker_list/lexicon_2026.json")) if os.path.exists("../marker_list/lexicon_2026.json") else None
    refresh_ref = {}

    def refresh(dfx):
        out = {}
        for v in ("rising_2026", "kept", "dropped"):
            r_ = lex_rate(dfx, L26[v])
            if v not in refresh_ref:  # standardise on the main-sample W2 controls
                m0 = (df.group == 0) & (df.wave == "W2")
                rr = lex_rate(df, L26[v])
                refresh_ref[v] = (rr[m0].mean(), rr[m0].std())
            col = f"lex26_{v}_z"
            dfx = dfx.assign(**{col: (r_ - refresh_ref[v][0]) / refresh_ref[v][1]})
            out[v] = {}
            for c in ("pooled", "2024-25", "2026"):
                if c != "pooled" and not (dfx["cohort"] == c).any():
                    continue
                s_ = dfx[dfx["cohort"] == c] if c != "pooled" else dfx
                e_ = fit(s_, col)
                out[v][c] = e_["d_w3"] | {"n_pairs": int(s_.pair.nunique()), "b1": e_["d_w1"]["b"],
                                          "b1_lo": e_["d_w1"]["lo"], "b1_hi": e_["d_w1"]["hi"], "b1_p": e_["d_w1"]["p"]}
        return out
    if L26:
        res["lexicon_refresh"] = {"n_words": {v: len(L26[v]) for v in ("rising_2026", "kept", "dropped")},
                                  "main": refresh(df)}

    # which words carry the 2024-25 excess: per-word DiD in author means, and leave-one-word-out beta3
    m25 = masks["2024-25"]
    words = []
    for w in MARK:
        if w not in voc:
            continue
        r_ = 1000 * X[:, voc[w]].toarray().ravel() / ntok
        t = pd.DataFrame({"g": df["group"], "a": df["author"], "w": df["wave"], "r": r_, "c": df["cohort"]})
        t = t.groupby(["c", "g", "a", "w"])["r"].mean().unstack()
        ch = (t["W3"] - t["W2"]).groupby(level=["c", "g"]).mean()
        allch = (t["W3"] - t["W2"]).groupby(level="g").mean()
        words.append({"word": w, "pooled_disc": float(allch.get(1, 0)), "pooled_ctrl": float(allch.get(0, 0)),
                      "c2425_did": float(ch.get(("2024-25", 1), 0) - ch.get(("2024-25", 0), 0)),
                      "c2026_did": float(ch.get(("2026", 1), 0) - ch.get(("2026", 0), 0))})
    WS = pd.DataFrame(words)
    WS["pooled_mean"] = (WS["pooled_disc"] + WS["pooled_ctrl"]) / 2
    WS.sort_values("pooled_mean", ascending=False).to_csv("revision_word_shifts.csv", index=False)
    loo = []
    f25 = fast["2024-25"]
    d1, d3 = df["d_w1"][m25], df["d_w3"][m25]
    for w in MARK[:int(os.environ.get("NLOO", len(MARK)))]:
        rt = rate([x for x in MARK if x != w])
        loo.append((w, float(f25.beta(zrate(rt)[m25], d1, d3)[1])))
    loo = sorted(loo, key=lambda x: x[1])
    res["loo_2024_25"] = {"min": loo[0], "max": loo[-1], "full": res["main"]["2024-25"]["mwi"]["d_w3"]["b"]}
    res["top_words_did_2024_25"] = WS.sort_values("c2425_did", ascending=False).head(12)[["word", "c2425_did", "c2026_did"]].round(3).values.tolist()
    res["top_words_pooled"] = WS.sort_values("pooled_mean", ascending=False).head(15)[["word", "pooled_disc", "pooled_ctrl"]].round(3).values.tolist()
    res["share_words_did_positive_2024_25"] = float((WS["c2425_did"] > 0).mean())
    res["share_words_did_positive_2026"] = float((WS["c2026_did"] > 0).mean())

    # heterogeneity by disclosure content, within cohort
    res["heterogeneity"] = {}
    for name, col in (("english_country", "english"), ("chatgpt_named", "chatgpt"), ("drafting_disclosed", "drafting"),
                      ("non_llm_only", "non_llm_only")):
        res["heterogeneity"][name] = {}
        for cn, s in (("pooled", df), ("2024-25", df[coh == 1]), ("2026", df[coh == 0])):
            mod = s["pair"].map(P[col]).astype(int)
            e = {}
            for v in (1, 0):
                ss = s[mod == v]
                e[str(v)] = (fit(ss)["d_w3"] | {"n_pairs": int(ss.pair.nunique())}) if ss.pair.nunique() >= 10 else None
            try:
                e["interaction"] = fit_mod(s, mod)
            except Exception:
                e["interaction"] = None
            res["heterogeneity"][name][cn] = e

    # control-to-discloser shift ratio and detection, pooled and by cohort
    res["pi"] = {n: pi_boot(s) for n, s in sub.items()}
    cs = author_shifts(df[df.group == 0])
    res["ctrl_shift_distribution"] = {q: float(cs["d32"].quantile(q)) for q in (0.25, 0.5, 0.75, 0.95)} | {
        "share_nonpositive": float((cs["d32"] <= 0).mean())}
    res["detection"] = {n: detection(s) for n, s in sub.items()}

    # silent vs. declared-non-use controls (controls only; exploratory)
    if P.loc[keep, "ctrl_nonuse"].sum() >= 15:
        c = df[df.group == 0].copy()
        c["nonuse"] = c["pair"].map(P["ctrl_nonuse"]).astype(int)
        c["n_w3"] = ((c["wave"] == "W3") & (c["nonuse"] == 1)).astype(int)
        c["n_w1"] = ((c["wave"] == "W1") & (c["nonuse"] == 1)).astype(int)
        r = wfit(smf.wls(f"mwi_z ~ n_w1 + n_w3 + {BASE} + C(author)", data=c, weights=c["w"]),
                 cov_type="cluster", cov_kwds={"groups": pd.factorize(c["author"])[0]})
        res["nonuse_vs_silent_controls"] = {"b3": float(r.params["n_w3"]), "lo": float(r.conf_int().loc["n_w3", 0]),
                                            "hi": float(r.conf_int().loc["n_w3", 1]), "p": float(r.pvalues["n_w3"]),
                                            "n_nonuse": int(P.loc[keep, "ctrl_nonuse"].sum())}

    # replication samples (independent pairs; see replication_plan.md and its Amendment 1)
    rp_aud = json.load(open("replication_audit.json"))["excluded"] if os.path.exists("replication_audit.json") else {}
    used_authors = set(df["author"])

    def load_rep(suffix, offset, cohort):
        units = json.load(open(f"replication{suffix}_units.json"))
        drop = {offset + k for k, p in enumerate(units) if p["disc"]["pmcid"] in rp_aud}
        raw = pd.read_csv(f"replication{suffix}_dataset.csv")
        raw = raw[~raw["pair"].isin(drop)]
        # safety: drop any pair whose author already appears in an earlier sample
        dup = set(raw.loc[raw["author"].isin(used_authors), "pair"])
        raw = raw[~raw["pair"].isin(dup)].reset_index(drop=True)
        rep = add_drift(build(raw))
        rep, _ = prep(rep, ref)
        rep["cohort"] = cohort
        used_authors.update(rep["author"])
        flow = {"collected_pairs": len(units), "fp_disclosure": len(drop), "dup_author": len(dup),
                "analysis_pairs": int(rep["pair"].nunique())}
        return rep, flow

    def summarise(rep, margin):
        e = fit(rep)
        return {"n_pairs": e["n_pairs"], "n_texts": e["n_texts"], "beta1": e["d_w1"], "beta3": e["d_w3"],
                "p3_one_sided": float(norm.sf(e["d_w3"]["b"] / e["d_w3"]["se"])),
                "tost_beta1": tost(e["d_w1"]), "tost_beta3": tost(e["d_w3"], margin), "trend_adj": e["trend_adj"],
                "all_outcomes": {y: fit(rep, y + "_z")["d_w3"] for y in OUT},
                "pi": pi_boot(rep), "detection": detection(rep), "randomisation": randomisation(rep.reset_index(drop=True)),
                "mwi_means": {f"{g}|{w}": float(v) for (g, w), v in
                              rep.groupby(["pair", "group", "wave"])["mwi"].mean().groupby(["group", "wave"]).mean().items()},
                "pair_cluster": fit(rep, cluster="pair")["d_w3"],
                "disc_query": {}}

    def one_sided(h):
        h["p_one_sided"] = float(h["p"] / 2 if h["b"] > 0 else 1 - h["p"] / 2)
        return h

    if os.path.exists("replication_dataset.csv"):
        rep, res["replication_flow"] = load_rep("", 10000, "2024-25")
        res["replication"] = summarise(rep, 0.25)
        if L26:
            res["lexicon_refresh"]["replication"] = refresh(rep)
        both = pd.concat([rep, df[coh == 0]], ignore_index=True)
        res["replication"]["h2_vs_main_2026"] = one_sided(fit_mod(both, (both["pair"] >= 10000).astype(int)))
        comb = pd.concat([rep, df[coh == 1]], ignore_index=True)
        res["replication"]["combined_2024_25"] = fit(comb)["d_w3"] | {"n_pairs": int(comb.pair.nunique())}
        rep.drop(columns=["text", "fw"]).to_csv("revision_replication_metrics.csv", index=False)
        if os.path.exists("replication2026_dataset.csv"):
            rep26, res["replication2026_flow"] = load_rep("2026", 20000, "2026")
            res["replication2026"] = summarise(rep26, 0.25)
            if L26:
                res["lexicon_refresh"]["replication2026"] = refresh(rep26)
            b2 = pd.concat([rep, rep26], ignore_index=True)
            res["replication2026"]["h3_rep2425_vs_rep2026"] = one_sided(fit_mod(b2, (b2["pair"] < 20000).astype(int)))
            comb26 = pd.concat([rep26, df[coh == 0]], ignore_index=True)
            res["replication2026"]["combined_2026"] = fit(comb26)["d_w3"] | {"n_pairs": int(comb26.pair.nunique())}
            allp = pd.concat([df, rep, rep26], ignore_index=True)
            allp["early"] = (allp["cohort"] == "2024-25").astype(int)
            res["all_samples"] = {"2024-25": fit(allp[allp.early == 1])["d_w3"] | {"n_pairs": int(allp[allp.early == 1].pair.nunique())},
                                  "2026": fit(allp[allp.early == 0])["d_w3"] | {"n_pairs": int(allp[allp.early == 0].pair.nunique())},
                                  "interaction": fit_mod(allp, allp["early"])}
            rep26.drop(columns=["text", "fw"]).to_csv("revision_replication2026_metrics.csv", index=False)

    df.drop(columns=["text", "fw"]).to_csv("revision_metrics.csv", index=False)
    json.dump(res, open("revision_results.json", "w"), indent=1, default=float)
    write_tables(res)
    print(json.dumps({k: res[k] for k in ("flow",)}, indent=1, default=str))
    for n in sub:
        m = res["main"][n]["mwi"]
        print(n, "b1", round(m["d_w1"]["b"], 3), "b3", round(m["d_w3"]["b"], 3), [round(m["d_w3"]["lo"], 3), round(m["d_w3"]["hi"], 3)], "p", m["d_w3"]["p"])
    if "replication" in res:
        r = res["replication"]
        print("replication", r["n_pairs"], r["beta3"], "H2", r["h2_vs_main_2026"])


# ---------------------------------------------------------------- LaTeX tables
def f2(x, d=2):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


def ci(e, d=2):
    return f"{f2(e['b'], d)} [{f2(e['lo'], d)}, {f2(e['hi'], d)}]"


def stars(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write_tables(res):
    # Table: event-study estimates, pooled and by cohort
    L = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         " & Pooled & 2024--25 cohort & 2026 cohort \\\\", "\\midrule"]
    n = {k: res["main"][k]["mwi"]["n_pairs"] for k in res["main"]}
    L.append(f"Pairs & {n['pooled']} & {n['2024-25']} & {n['2026']} \\\\ \\addlinespace")
    L.append("\\multicolumn{4}{@{}l}{\\emph{Placebo, $\\hat\\beta_1$ (2014--16 vs.\\ 2019--21)}} \\\\")
    L.append(f"Marker-word index & {ci(res['main']['pooled']['mwi']['d_w1'])} & {ci(res['main']['2024-25']['mwi']['d_w1'])} & {ci(res['main']['2026']['mwi']['d_w1'])} \\\\ \\addlinespace")
    L.append("\\multicolumn{4}{@{}l}{\\emph{Discloser excess, $\\hat\\beta_3$ (2024--26 vs.\\ 2019--21)}} \\\\")
    for y in OUT:
        cells = []
        for k in ("pooled", "2024-25", "2026"):
            e = res["main"][k][y]["d_w3"]
            cells.append(ci(e) + stars(e["p"]))
        L.append(f"{LABEL[y]} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/tab_es.tex", "w").write("\n".join(L))

    # Table: all outcomes by cohort, all pairs vs. pairs of research articles, with cohort-interaction p-values
    S = res["secondary_robustness"]
    L = ["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{All pairs} & \\multicolumn{3}{c}{Both index papers research articles} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
         "Outcome & 2024--25 & 2026 & Int.\\ $p$ & 2024--25 & 2026 & Int.\\ $p$ \\\\", "\\midrule"]
    n_all = (S["all"]["2024-25"]["mwi"]["n_pairs"], S["all"]["2026"]["mwi"]["n_pairs"])
    n_ra = (S["research_articles"]["2024-25"]["mwi"]["n_pairs"], S["research_articles"]["2026"]["mwi"]["n_pairs"])
    L.append(f"Pairs & {n_all[0]} & {n_all[1]} & & {n_ra[0]} & {n_ra[1]} & \\\\ \\addlinespace")
    for y in OUT:
        a, b = S["all"]["2024-25"][y], S["all"]["2026"][y]
        c, d = S["research_articles"]["2024-25"][y], S["research_articles"]["2026"][y]
        L.append(f"{LABEL[y]} & {ci(a)} & {ci(b)} & {pfmt(res['cohort_interaction'][y]['p'])} & {ci(c)} & {ci(d)} & "
                 f"{pfmt(S['research_articles']['interaction'][y]['p'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/tab_secondary.tex", "w").write("\n".join(L))

    # Table: robustness of beta3 on the MWI
    L = ["\\begin{tabularx}{\\textwidth}{@{}L r c r c r c r@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{Pooled} & \\multicolumn{2}{c}{2024--25 cohort} & \\multicolumn{2}{c}{2026 cohort} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}",
         "Specification & Pairs & $\\hat\\beta_3$ [95\\% CI] & $p$ & $\\hat\\beta_3$ [95\\% CI] & $p$ & $\\hat\\beta_3$ [95\\% CI] & $p$ \\\\",
         "\\midrule"]
    for lab, v in res["robustness"].items():
        a, b, c = v["pooled"], v["2024-25"], v["2026"]
        if not (a and b and c):
            continue
        L.append(f"{lab} & {a['n_pairs']} & {ci(a)} & {pfmt(a['p'])} & {ci(b)} & {pfmt(b['p'])} & {ci(c)} & {pfmt(c['p'])} \\\\")
    L += ["\\bottomrule", "\\end{tabularx}"]
    open(f"{TAB}/tab_robust.tex", "w").write("\n".join(L))

    # Table: detection
    lab = [("mwi_raw", "MWI of the 2024--26 abstract"), ("mwi_minus_w2", "MWI minus own 2019--21 mean"),
           ("mwi_z_own", "MWI as $z$-score of own pre-period"), ("five_raw_cv", "Five features, raw (CV)"),
           ("five_baseline_cv", "Five features, baseline-normalised (CV)")]
    cols = ["pooled", "2024-25", "2026"] + [k for k in ("replication", "replication2026") if k in res]
    names = {"pooled": "Pooled", "2024-25": "2024--25", "2026": "2026", "replication": "Repl.\\ 2024--25",
             "replication2026": "Repl.\\ 2026"}
    head = "".join(f" & {names[k]}" for k in cols) + " \\\\"
    L = ["\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{Main sample}" + (f" & \\multicolumn{{{len(cols) - 3}}}{{c}}{{Replication}}" if len(cols) > 3 else "") + " \\\\",
         "Score" + head, "\\midrule"]
    det = {k: (res["detection"][k] if k in res["detection"] else res[k]["detection"]) for k in cols}
    L.append("Pairs & " + " & ".join(str(det[k]["n_pairs"]) for k in cols) + " \\\\ \\addlinespace")
    for key, name in lab:
        L.append(name + " & " + " & ".join(f"{det[k][key]['auc']:.2f} [{det[k][key]['lo']:.2f}, {det[k][key]['hi']:.2f}]"
                                            for k in cols) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    open(f"{TAB}/tab_detect.tex", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
