"""E9 (revision): independent replication samples of disclosers and matched controls (see replication_plan.md).
`python3 e9_replication.py` draws the 2024-25 sample; `python3 e9_replication.py 2026` the 2026 sample (Amendment 1).
Resumable: state is kept in replication[2026]_state.json; rerun after the OpenAlex budget resets to add pairs.
Writes replication[2026]_units.json (pairs) and replication[2026]_dataset.csv (same columns as dataset.csv)."""
import csv, datetime as dt, json, os, random, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

import oa
from e1_disclosers import QUERIES, TOOL, USE, WRITING, SCOPE, GRAPHICS_ONLY, is_false_positive
from e3_controls import AI_TERMS, parse_date
from e4_dataset import clean, pubmed_abstracts, MIN_WORDS

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
# Cohort: "2024-25" (default, the original plan) or "2026" (Amendment 1 of replication_plan.md)
COHORT = sys.argv[1] if len(sys.argv) > 1 else "2024-25"
YEARS = ("2024", "2025") if COHORT == "2024-25" else ("2026",)
SEED = 2029 if COHORT == "2024-25" else 2031
SUFFIX = "" if COHORT == "2024-25" else "2026"
STATE = f"replication{SUFFIX}_state.json"
OTHER_STATES = [f for f in ("replication_state.json", "replication2026_state.json") if f != STATE]
PAIR_OFFSET = 10000 if COHORT == "2024-25" else 20000
HEAD = {"User-Agent": "ai-writing-study/1.0 (mailto:researcher@institution.edu)"}
SENT2 = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])")
HEADING = re.compile(r"^.*?(?:declaration|statement|use)[^.]{0,80}?(?:in the (?:writing|manuscript preparation) process|"
                     r"in scientific writing)\s*:?\s*", re.I)
AI_ANY = re.compile(r"chat ?gpt|\bgpt-?[345]|generative (?:ai|artificial)|large language model|\bllms?\b|ai[- ]assisted|"
                    r"\bai tools?\b|grammarly|deepl\b|quillbot|paperpal|writefull|\bclaude\b|\bgemini\b|copilot|deepseek", re.I)
NON_USE = re.compile(r"\b(no|not|neither|nor|never|without)\b[^.]{0,60}\b(used|employed|utili[sz]ed|use|assist)", re.I)
CTRL_EXTRA = ' OR "AI tools" OR "AI tool" OR "language model" OR "Writefull" OR "Perplexity"'


from e8_verify import get  # cached GET (cache_epmc/), shared with the prefetcher


def fulltext(pmcid):
    raw = get(f"{EPMC}/{pmcid}/fullTextXML")
    try:
        return " ".join(" ".join(ET.fromstring(raw).itertext()).split()) if raw else None
    except ET.ParseError:
        return None


def disclosure2(text):
    """E1 instrument with the audit fixes: better sentence split; purpose terms required outside the section heading."""
    for s in SENT2.split(text):
        if not (re.search(TOOL, s, re.I) and re.search(USE, s, re.I) and re.search(SCOPE, s, re.I)):
            continue
        body = HEADING.sub("", s)
        terms = set(m.lower() for m in re.findall(WRITING, body, re.I)) - {"edit", "writing"}
        if not re.search(WRITING, body, re.I):
            continue
        if re.search(GRAPHICS_ONLY + r"|\bcode\b|coding|programming|bibliograph|literature search", body, re.I) and not terms:
            continue
        if is_false_positive(s) or NON_USE.search(body):
            continue
        return s.strip()[:700]
    return None


def frame():
    """Every OA PMC record of the cohort matching any disclosure phrasing, as {pmcid: query}. 2026 is harvested month
    by month so that no query exceeds the 10,000-record E-utilities limit."""
    out = {}
    periods = [str(y) for y in YEARS] if COHORT == "2024-25" else [f"2026/{m:02d}" for m in range(1, 11)]
    for label, q in QUERIES.items():
        for y in periods:
            term = f"{q} AND {y}[pdat] AND open access[filter]"
            raw = get(f"{ESEARCH}?{urllib.parse.urlencode({'db': 'pmc', 'term': term, 'retmode': 'json', 'retmax': 10000})}")
            for uid in json.loads(raw)["esearchresult"]["idlist"]:
                out.setdefault(f"PMC{uid}", label)
            time.sleep(0.4)
    return out


def epmc_meta(pmcid):
    p = {"query": f"PMCID:{pmcid}", "format": "json", "resultType": "lite"}
    res = json.loads(get(f"{EPMC}/search?{urllib.parse.urlencode(p)}") or b"{}").get("resultList", {}).get("result", [])
    return res[0] if res else None


def control_candidates(issn, d0, days, exclude):
    issns = " OR ".join(f'ISSN:"{x.strip()}"' for x in issn.split(";") if x.strip())
    q = (f'({issns}) AND FIRST_PDATE:[{d0 - dt.timedelta(days=days)} TO {d0 + dt.timedelta(days=days)}] '
         f'AND OPEN_ACCESS:y AND PUB_TYPE:"research-article" AND HAS_FT:y NOT {AI_TERMS[:-1]}{CTRL_EXTRA})')
    p = {"query": q, "format": "json", "pageSize": 100, "resultType": "lite"}
    res = json.loads(get(f"{EPMC}/search?{urllib.parse.urlencode(p)}") or b"{}").get("resultList", {}).get("result", [])
    return [r for r in res if r.get("doi") and r.get("pmcid") and r["doi"].lower() not in exclude]


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    if "frame" not in st:
        st["frame"] = frame()
        st.update(order=None, pos=0, screened=[], pairs=[], log=[])
    main_dois = {p["doi"] for p in json.load(open("disclosers.json"))["papers"]}
    used = {u["anchor"]["id"] for u in json.load(open("discloser_units.json"))}
    used |= {c["anchor"]["id"] for c in json.load(open("control_units.json")) if not c.get("unmatched")}
    used |= {a for p in st["pairs"] for a in (p["disc"]["anchor"]["id"], p["ctrl"]["anchor"]["id"])}
    used |= {s["anchor"] for s in st["screened"] if s.get("anchor")}
    for f in OTHER_STATES:  # never reuse an author from the other replication sample
        if os.path.exists(f):
            o = json.load(open(f))
            used |= {a for p in o["pairs"] for a in (p["disc"]["anchor"]["id"], p["ctrl"]["anchor"]["id"])}
            used |= {x["anchor"] for x in o["screened"] if x.get("anchor")}
    if st["order"] is None:
        order = sorted(st["frame"])
        random.Random(SEED).shuffle(order)
        st["order"] = order
    rng = random.Random(SEED + 1)
    print(f"frame {len(st['frame'])}, resuming at {st['pos']}, pairs so far {len(st['pairs'])}", file=sys.stderr)
    try:
        while st["pos"] < len(st["order"]):
            pmcid = st["order"][st["pos"]]
            st["pos"] += 1
            rec = {"pmcid": pmcid, "query": st["frame"][pmcid]}
            st["screened"].append(rec)
            m = epmc_meta(pmcid)
            if not m or not m.get("doi") or m["doi"].lower() in main_dois or not m.get("journalIssn"):
                rec["status"] = "no_meta_or_in_main"; continue
            if (m.get("pubYear") or "") not in YEARS:
                rec["status"] = "year"; continue
            text = fulltext(pmcid)
            s = disclosure2(text) if text else None
            if not s:
                rec["status"] = "no_disclosure"; continue
            doi = m["doi"].lower()
            rec.update(doi=doi, disclosure=s, year=m["pubYear"], issn=m["journalIssn"],
                       pub_date=m.get("firstPublicationDate"), journal=m.get("journalTitle"), pmid=m.get("pmid"))
            w = oa.resolve([doi]).get(doi)
            a = oa.anchor(w) if w else None
            if not a or a["id"] in used:
                rec["status"] = "no_anchor_or_used"; continue
            rec["anchor"] = a["id"]
            hist = oa.history(a["id"])
            used.add(a["id"])
            if not oa.eligible(hist):
                rec["status"] = "ineligible"; continue
            disc = {**rec, "work_id": w["id"].rsplit("/", 1)[-1], "n_authors": len(w["authorships"]),
                    "index_abstract": oa.abstract(w), "anchor": a, "history": hist}
            d0 = parse_date(rec["pub_date"])
            match = None
            for days, tries in ((120, 6), (365, 12)):
                cands = control_candidates(rec["issn"], d0, days, main_dois | {doi})
                rng.shuffle(cands)
                for c in cands[:tries]:
                    t = fulltext(c["pmcid"])
                    if not t or disclosure2(t) or any(AI_ANY.search(x) and not NON_USE.search(x) for x in SENT2.split(t)
                                                      if re.search(USE, x, re.I)):
                        continue
                    cw = oa.resolve([c["doi"].lower()]).get(c["doi"].lower())
                    ca = oa.anchor(cw) if cw else None
                    if not ca or ca["id"] in used:
                        continue
                    ch = oa.history(ca["id"])
                    used.add(ca["id"])
                    if oa.eligible(ch):
                        match = {"doi": c["doi"].lower(), "pmcid": c["pmcid"], "pmid": c.get("pmid"),
                                 "pub_date": c.get("firstPublicationDate"), "journal": c.get("journalTitle"),
                                 "work_id": cw["id"].rsplit("/", 1)[-1], "n_authors": len(cw["authorships"]),
                                 "index_abstract": oa.abstract(cw), "anchor": ca, "history": ch, "window": days}
                        break
                if match:
                    break
            rec["status"] = "paired" if match else "unmatched"
            if match:
                st["pairs"].append({"disc": disc, "ctrl": match})
            print(f"{pmcid} {rec['status']} pairs={len(st['pairs'])} budget={oa.remaining}", file=sys.stderr, flush=True)
            json.dump(st, open(STATE, "w"))  # after every fully processed record, so a stop loses nothing
    except oa.BudgetExhausted:
        st["pos"] -= 1
        st["screened"].pop()
        print("OpenAlex budget exhausted; rerun after reset to continue", file=sys.stderr)
    json.dump(st, open(STATE, "w"))
    build_dataset(st["pairs"])


def build_dataset(pairs):
    need = [w["pmid"] for p in pairs for u in (p["disc"], p["ctrl"]) for w in u["history"] if not w["abstract"] and w["pmid"]]
    need += [u["pmid"] for p in pairs for u in (p["disc"], p["ctrl"]) if not u.get("index_abstract") and u.get("pmid")]
    pm = pubmed_abstracts(sorted(set(need)))
    rows = []
    for k, p in enumerate(pairs):
        for group, u in ((1, p["disc"]), (0, p["ctrl"])):
            items = [("W1" if w["year"] < 2017 else "W2", w["year"], w["id"], w["n_authors"], w["position"],
                      w["abstract"] or pm.get(w["pmid"] or "")) for w in u["history"]]
            items.append(("W3", int(str(u.get("pub_date") or p["disc"]["pub_date"])[:4]), u["work_id"], u["n_authors"],
                          u["anchor"]["rule"], u.get("index_abstract") or pm.get(u.get("pmid") or "")))
            for wave, year, wid, n_auth, pos, text in items:
                txt, structured = clean(text)
                if len(txt.split()) < MIN_WORDS:
                    continue
                rows.append({"pair": PAIR_OFFSET + k, "group": group, "author": u["anchor"]["id"], "wave": wave, "year": year,
                             "work_id": wid, "n_authors": n_auth, "position": pos, "structured": int(structured),
                             "country": (u["anchor"]["countries"] or ["NA"])[0], "topic": None, "text": txt})
    json.dump(pairs, open(f"replication{SUFFIX}_units.json", "w"))
    if rows:
        with open(f"replication{SUFFIX}_dataset.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"replication pairs {len(pairs)}, rows {len(rows)}", file=sys.stderr)


if __name__ == "__main__":
    main()
