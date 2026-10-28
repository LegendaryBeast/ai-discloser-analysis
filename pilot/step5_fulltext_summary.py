"""Step 5: summarise full-text availability for retained authors (writes fulltext_summary.txt)."""
import json, statistics
from collections import Counter

A = json.load(open("fulltext_status.json"))
TIERS = ["pmc_oa", "preprint", "pmc_other", "other_oa", "abstract_only", "nothing"]
MINEABLE = {"pmc_oa", "preprint"}           # machine-readable / redistributable author text
READABLE = MINEABLE | {"pmc_other", "other_oa"}  # free to read, may need PDF parsing / licence check
WIN = {"2014-16": range(2014, 2017), "2019-21": range(2019, 2022)}
lines = []


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


for label, only_first in (("ALL papers", False), ("FIRST-AUTHORED papers", True)):
    lines.append(f"== Paper-level tiers, {label} ==")
    for wname, yrs in WIN.items():
        ws = [w for a in A.values() for w in a["works"] if w["year"] in yrs and (w["first"] or not only_first)]
        c = Counter(w["tier"] for w in ws)
        lines.append(f"  {wname} (n={len(ws)}): " + ", ".join(f"{t} {pct(c[t], len(ws))}" for t in TIERS))
    lines.append("")


def per_author(only_first, ok):
    cnt = {}
    for aid, a in A.items():
        cnt[aid] = [sum(1 for w in a["works"] if w["year"] in yrs and w["tier"] in ok and (w["first"] or not only_first))
                    for yrs in WIN.values()]
    return cnt


n = len(A)
lines.append(f"== Author-level (retained authors n={n}; first author in both windows: "
             f"{sum(a['first_both'] for a in A.values())}) ==")
for label, only_first in (("any position", False), ("as first author", True)):
    for okname, ok in (("machine-readable full text (PMC OA / preprint)", MINEABLE),
                       ("any free full text", READABLE),
                       ("abstract available", set(TIERS) - {"nothing"})):
        c = per_author(only_first, ok)
        both = [v for v in c.values() if v[0] and v[1]]
        med = statistics.median(min(v) for v in both) if both else 0
        lines.append(f"  [{label}] >=1 paper with {okname} in BOTH windows: {pct(len(both), n)}"
                     f"  (median papers in thinner window: {med})")
    lines.append("")
open("fulltext_summary.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
