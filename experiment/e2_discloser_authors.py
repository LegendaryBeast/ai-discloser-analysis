"""E2: resolve disclosing papers in OpenAlex, pick the anchor author and check eligibility
(>=2 abstracts in each of 2014-16 and 2019-21). Writes discloser_units.json."""
import json, sys
import oa

AUDIT = json.load(open("audit.json")).get("excluded", {})
D = [p for i, p in enumerate(json.load(open("disclosers.json"))["papers"])
     if str(i) not in AUDIT and p.get("doi") not in AUDIT]

try:
    existing = json.load(open("discloser_units.json"))
    done_dois = {u["doi"] for u in existing}
    units = existing
    seen_authors = {u["anchor"]["id"] for u in units}
    print(f"Resuming e2: already have {len(units)} units ({sum(u['eligible'] for u in units)} eligible)", file=sys.stderr)
except Exception:
    units, seen_authors, done_dois = [], set(), set()

# Filter to pending papers
pending = [p for p in D if p["doi"] not in done_dois]
works = oa.resolve([p["doi"] for p in pending])

exhausted = False
for p in pending:
    w = works.get(p["doi"])
    if not w:
        continue
    a = oa.anchor(w)
    if not a or a["id"] in seen_authors:      # one index paper per author
        continue
    try:
        hist = oa.history(a["id"])
    except oa.BudgetExhausted:
        print(f"OpenAlex daily budget exhausted at {p['doi']}; saving progress.", file=sys.stderr)
        exhausted = True
        break
    seen_authors.add(a["id"])
    units.append({**p, "work_id": w["id"].rsplit("/", 1)[-1], "n_authors": len(w["authorships"]),
                  "topic": ((w.get("primary_topic") or {}).get("subfield") or {}).get("display_name"),
                  "index_abstract": oa.abstract(w), "anchor": a, "history": hist, "eligible": oa.eligible(hist)})
    done_dois.add(p["doi"])
    print(p["pmcid"], a["rule"], len(hist), units[-1]["eligible"], "budget", oa.remaining, file=sys.stderr)
    if len(units) % 25 == 0:
        json.dump(units, open("discloser_units.json", "w"))

json.dump(units, open("discloser_units.json", "w"))
print(f"Total units now: {len(units)}, eligible: {sum(u['eligible'] for u in units)}", file=sys.stderr)
if exhausted:
    print("NOTE: Daily rate limit reached. Re-run e2_discloser_authors.py tomorrow to continue.", file=sys.stderr)
