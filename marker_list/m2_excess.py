"""M2: excess-vocabulary analysis (after Kobak et al.) on the derivation half of the corpus.
Expected 2024/2025 frequency = linear extrapolation from 2021-2022; a word is a candidate if it is in excess
in BOTH 2024 and 2025. Writes candidates.csv for style/content classification."""
import csv, gzip, json, os, re, sys
from collections import Counter, defaultdict

TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*")
YEARS = [2019, 2020, 2021, 2022, 2024, 2025]
# Candidate thresholds (preregistered with the list)
MIN_RATIO = 1.5      # observed / expected document frequency, in both 2024 and 2025
MIN_GAP = 0.0005     # observed - expected, in both years (0.05 percentage points of abstracts)
MIN_DOCS = 30        # documents containing the word in each target year


def norm(s):
    return " ".join(re.sub(r"[^a-z ]", " ", (s or "").lower()).split())


EXCL_FILE = "pilot_author_pmids.txt"  # PMIDs matched by the name/ORCID rule below; used when present (public release)


def load_exclusions():
    """Pilot study authors (ORCIDs and full names) whose abstracts are removed from the derivation corpus.
    In the public release, author names are not distributed; the matched PMIDs are read from EXCL_FILE instead."""
    if os.path.exists(EXCL_FILE):
        return set(open(EXCL_FILE).read().split()), None
    orcids, names = set(), set()
    for r in json.load(open("../pilot/author_histories.json")):
        for role in ("first", "corresponding", "last"):
            a = r.get(role)
            if a:
                if a.get("orcid"):
                    orcids.add(a["orcid"].rsplit("/", 1)[-1])
                names.add(norm(a["name"]))
    return orcids, names


def is_excluded(rec, orcids, names):
    if names is None:  # PMID list from EXCL_FILE
        return rec["pmid"] in orcids
    return any((a["orcid"] in orcids) or norm(f"{a['fore']} {a['last']}") in names for a in rec["authors"])


def main():
    orcids, names = load_exclusions()
    df = {y: Counter() for y in YEARS}
    n = Counter()
    excluded = 0
    with gzip.open("corpus.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["pmid"]) % 2:          # odd PMIDs = validation half, untouched here
                continue
            if is_excluded(rec, orcids, names):
                excluded += 1
                continue
            n[rec["year"]] += 1
            df[rec["year"]].update(set(TOKEN.findall(rec["abstract"].lower())))
    p = {y: {w: c / n[y] for w, c in df[y].items()} for y in YEARS}
    rows = []
    for w in set(df[2024]) | set(df[2025]):
        p21, p22 = p[2021].get(w, 0), p[2022].get(w, 0)
        stats = {}
        for t in (2024, 2025):
            q = p22 + max(p22 - p21, 0) * (t - 2022)
            obs = p[t].get(w, 0)
            stats[t] = (obs, q, obs / q if q else float("inf"), obs - q, df[t][w])
        if all(s[2] >= MIN_RATIO and s[3] >= MIN_GAP and s[4] >= MIN_DOCS for s in stats.values()):
            rows.append([w] + [f"{p[y].get(w, 0):.6f}" for y in YEARS] +
                        [f"{stats[2024][2]:.2f}", f"{stats[2025][2]:.2f}", f"{stats[2025][3]:.5f}"])
    rows.sort(key=lambda r: -float(r[-1]))
    with open("candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word"] + [f"p{y}" for y in YEARS] + ["ratio2024", "ratio2025", "gap2025"])
        w.writerows(rows)
    json.dump({"n_by_year": n, "excluded_pilot_author_abstracts": excluded,
               "thresholds": {"min_ratio": MIN_RATIO, "min_gap": MIN_GAP, "min_docs": MIN_DOCS},
               "n_candidates": len(rows)}, open("derivation_stats.json", "w"), indent=1)
    print(f"derivation n={dict(n)}, excluded={excluded}, candidates={len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
