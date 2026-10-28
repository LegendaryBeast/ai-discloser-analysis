"""E3: one matched non-disclosing control per eligible discloser.
Candidates: open-access research articles in the same journal (ISSN), first published within +-120 days (second pass: +-365 days) of the
disclosing paper, whose full text never mentions generative AI / LLM tools. Candidates are tried in seeded random
order; the first whose anchor author is new and eligible (>=2 abstracts in each pre-LLM window) is kept."""
import datetime as dt, json, random, re, sys, time, urllib.parse, urllib.request
import oa

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
AI_TERMS = ('("ChatGPT" OR "generative AI" OR "generative artificial intelligence" OR "large language model" OR '
            '"AI-assisted" OR "Grammarly" OR "DeepL" OR "QuillBot" OR "Paperpal" OR "Claude" OR "Gemini" OR '
            '"DeepSeek" OR "Copilot" OR "GPT-4" OR "LLM")')
MAX_TRIES = 6
SEED = 2028
# Second pass for unmatched disclosers: wider window and more candidates (run with argument "pass2").
PASS2 = len(sys.argv) > 1 and sys.argv[1] == "pass2"
# "retry": pass-1 settings, re-processing unmatched disclosers (used after fixing multi-ISSN journals).
RETRY = PASS2 or (len(sys.argv) > 1 and sys.argv[1] == "retry")
WINDOW_DAYS = 365 if PASS2 else 120
if PASS2:
    MAX_TRIES = 12


def parse_date(val):
    if not val:
        return dt.date(2025, 7, 1)
    s = str(val).strip()
    try:
        return dt.date.fromisoformat(s)
    except Exception:
        pass
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.match(r"^(\d{4})\s+([A-Za-z]{3})", s)
    if m:
        months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                  "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        mon = months.get(m.group(2).lower()[:3], 7)
        return dt.date(int(m.group(1)), mon, 15)
    m = re.match(r"^(\d{4})", s)
    if m:
        return dt.date(int(m.group(1)), 7, 1)
    return dt.date(2025, 7, 1)


def oa_candidates(issns_str, d0, window_days, discloser_dois):
    """Fallback: retrieve candidates from OpenAlex if Europe PMC search is unavailable."""
    for raw_issn in issns_str.split(";"):
        issn = raw_issn.strip()
        if not issn:
            continue
        d_start = (d0 - dt.timedelta(days=window_days)).isoformat()
        d_end = (d0 + dt.timedelta(days=window_days)).isoformat()
        filter_str = f"primary_location.source.issn:{issn},from_publication_date:{d_start},to_publication_date:{d_end},is_oa:true,type:article"
        try:
            d = oa.oa_get("works", {"filter": filter_str, "per_page": 20, "select": oa.WORK_FIELDS})
            res = []
            for w in d.get("results", []):
                doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                if doi and doi not in discloser_dois:
                    res.append({
                        "doi": doi,
                        "pmcid": ((w.get("ids") or {}).get("pmcid") or "").rsplit("/", 1)[-1] or None,
                        "pmid": ((w.get("ids") or {}).get("pmid") or "").rsplit("/", 1)[-1] or None,
                        "title": w.get("title"),
                        "journalTitle": (w.get("primary_location") or {}).get("source", {}).get("display_name"),
                        "firstPublicationDate": w.get("publication_date")
                    })
            if res:
                return res
        except Exception:
            continue
    return []


def epmc(query, n=100):
    p = {"query": query, "format": "json", "pageSize": n, "resultType": "lite"}
    for i in range(4):
        try:
            with urllib.request.urlopen(f"{EPMC}?{urllib.parse.urlencode(p)}", timeout=90) as r:
                return json.loads(r.read())["resultList"]["result"]
        except Exception:
            time.sleep(3 * (i + 1))
    return []


def main():
    units = [u for u in json.load(open("discloser_units.json")) if u["eligible"]]
    used_authors = {u["anchor"]["id"] for u in json.load(open("discloser_units.json"))}
    discloser_dois = {p["doi"] for p in json.load(open("disclosers.json"))["papers"]}
    rng = random.Random(SEED)
    try:
        done = {c["matched_to"]: c for c in json.load(open("control_units.json"))}
    except FileNotFoundError:
        done = {}
    for u in units:
        prev = done.get(u["doi"])
        if (prev and not (RETRY and prev.get("unmatched"))) or not u.get("issn") or not u.get("pub_date"):
            continue
        d0 = parse_date(u["pub_date"])
        issns = " OR ".join(f'ISSN:"{x.strip()}"' for x in u["issn"].split(";") if x.strip())
        q = (f'({issns}) AND FIRST_PDATE:[{d0 - dt.timedelta(days=WINDOW_DAYS)} TO {d0 + dt.timedelta(days=WINDOW_DAYS)}] '
             f'AND OPEN_ACCESS:y AND PUB_TYPE:"research-article" AND HAS_FT:y NOT {AI_TERMS}')
        cands = [c for c in epmc(q) if c.get("doi") and c["doi"].lower() not in discloser_dois]
        if not cands:
            cands = oa_candidates(u["issn"], d0, WINDOW_DAYS, discloser_dois)
        rng.shuffle(cands)
        cands = cands[:MAX_TRIES]
        try:
            works = oa.resolve([c["doi"].lower() for c in cands])
            match = None
            for c in cands:
                w = works.get(c["doi"].lower())
                a = oa.anchor(w) if w else None
                if not a or a["id"] in used_authors:
                    continue
                hist = oa.history(a["id"])
                if oa.eligible(hist):
                    used_authors.add(a["id"])
                    match = {"matched_to": u["doi"], "doi": c["doi"].lower(), "pmcid": c.get("pmcid"),
                             "pmid": c.get("pmid"), "title": c.get("title"), "journal": c.get("journalTitle"),
                             "pub_date": c.get("firstPublicationDate"), "work_id": w["id"].rsplit("/", 1)[-1],
                             "n_authors": len(w["authorships"]),
                             "topic": ((w.get("primary_topic") or {}).get("subfield") or {}).get("display_name"),
                             "index_abstract": oa.abstract(w), "anchor": a, "history": hist, "eligible": True,
                             "candidates_tried": cands.index(c) + 1, "pool_size": len(cands), "pass": 2 if PASS2 else 1}
                    break
        except oa.BudgetExhausted:
            print("OpenAlex budget exhausted; rerun after reset to continue", file=sys.stderr)
            break
        else:
            done[u["doi"]] = match or {"matched_to": u["doi"], "unmatched": True, "pool_size": len(cands)}
            json.dump(list(done.values()), open("control_units.json", "w"))
            print(u["doi"], "matched" if match else "unmatched", "budget", oa.remaining, file=sys.stderr, flush=True)
    m = [c for c in done.values() if not c.get("unmatched")]
    print(f"eligible disclosers {len(units)}, matched {len(m)}", file=sys.stderr)


if __name__ == "__main__":
    main()
