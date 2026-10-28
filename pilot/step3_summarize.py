"""Step 3: retention rates of disclosing authors across the 2014-16 and 2019-21 windows."""
import csv, json
from collections import Counter

R = json.load(open("author_histories.json"))
N = len(R)
SUSPECT = 60  # >60 articles in a 3-year window: likely a merged/over-clustered OpenAlex author profile


def has(a, w, first=False):
    return bool(a) and a[w + ("_first" if first else "")] > 0


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


lines = [f"Papers: {N}", f"Corresponding author flagged in OpenAlex: {pct(sum(r['corresponding_flagged'] for r in R), N)}", ""]
for role in ("first", "corresponding", "last"):
    A = [r[role] for r in R if r.get(role)]
    n = len(A)
    lines += [f"== {role} author (n={n}) ==",
              f"  ORCID linked:                {pct(sum(bool(a['orcid']) for a in A), n)}",
              f"  any paper 2014-16:           {pct(sum(has(a, 'w2015') for a in A), n)}",
              f"  any paper 2019-21:           {pct(sum(has(a, 'w2020') for a in A), n)}",
              f"  BOTH windows (any position): {pct(sum(has(a, 'w2015') and has(a, 'w2020') for a in A), n)}",
              f"  BOTH windows as 1st author:  {pct(sum(has(a, 'w2015', 1) and has(a, 'w2020', 1) for a in A), n)}",
              f"  suspect merged profile:      {sum(max(a['w2015'], a['w2020']) > SUSPECT for a in A)}", ""]

# Paper-level: does at least one of first / corresponding author have papers in both windows?
either = sum(any(has(r.get(k), "w2015") and has(r.get(k), "w2020") for k in ("first", "corresponding")) for r in R)
either_first = sum(any(has(r.get(k), "w2015", 1) and has(r.get(k), "w2020", 1) for k in ("first", "corresponding")) for r in R)
lines += [f"Paper-level: first OR corresponding author present in both windows: {pct(either, N)}",
          f"Paper-level: first OR corresponding author FIRST-authored in both windows: {pct(either_first, N)}",
          "", "Countries of first authors: " + ", ".join(f"{c}:{n}" for c, n in Counter(
              (r["first"]["countries"] or ["?"])[0] for r in R if r.get("first")).most_common())]
open("summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

with open("pilot_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pmcid", "doi", "journal", "query", "n_authors", "role", "author", "openalex_id", "orcid",
                "papers_2014_16", "first_auth_2014_16", "papers_2019_21", "first_auth_2019_21", "disclosure"])
    for r in R:
        for role in ("first", "corresponding", "last"):
            a = r.get(role)
            if a:
                w.writerow([r["pmcid"], r["doi"], r["journal"], r["query"], r["n_authors"], role, a["name"], a["id"],
                            a["orcid"] or "", a["w2015"], a["w2015_first"], a["w2020"], a["w2020_first"], r["disclosure"]])
