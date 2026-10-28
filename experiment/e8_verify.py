"""E8 (revision): data-integrity checks that need only Europe PMC / NCBI (no OpenAlex budget).
1. Article type, first-publication date and PMCID of every index paper (disclosers and controls), from Europe PMC.
2. Full-text check of every control: does it contain an AI-use disclosure that the E3 query missed?
3. Size of the full sampling frame by publication year for each disclosure query (the E1 pool was capped at 2,500 hits
   per query, newest first), so that the sample can be compared with, and reweighted to, the population.
Writes e8_verify.json. Responses are cached in cache_epmc/."""
import hashlib, json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

from e1_disclosers import QUERIES, TOOL, USE, WRITING, SCOPE, GRAPHICS_ONLY, SENT, is_false_positive

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache_epmc")
os.makedirs(CACHE, exist_ok=True)
HEAD = {"User-Agent": "ai-writing-study/1.0 (mailto:researcher@institution.edu)"}
# Any mention of a generative-AI or AI writing tool, used to flag controls for reading
AI_ANY = re.compile(r"chat ?gpt|\bgpt-?[345]|generative (?:ai|artificial)|large language model|\bllms?\b|"
                    r"ai[- ]assisted|\bai tools?\b|grammarly|deepl\b|quillbot|paperpal|writefull|\bclaude\b|"
                    r"\bgemini\b|copilot|deepseek|language model", re.I)


def get(url, binary=False):
    key = os.path.join(CACHE, hashlib.sha1(url.encode()).hexdigest())
    if os.path.exists(key):
        return open(key, "rb").read()
    for i in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEAD), timeout=90) as r:
                data = r.read()
            open(key, "wb").write(data)
            time.sleep(0.15)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(key, "wb").write(b"")
                return b""
            time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(3 * (i + 1))
    return b""


def meta(dois):
    out = {}
    for i in range(0, len(dois), 40):
        q = " OR ".join(f'DOI:"{d}"' for d in dois[i:i + 40])
        p = {"query": q, "format": "json", "pageSize": 100, "resultType": "core"}
        raw = get(f"{EPMC}/search?{urllib.parse.urlencode(p)}")
        for r in json.loads(raw or b"{}").get("resultList", {}).get("result", []):
            d = (r.get("doi") or "").lower()
            if d in out and out[d].get("pmcid"):
                continue
            out[d] = {"pmcid": r.get("pmcid"), "pmid": r.get("pmid"),
                      "first_pdate": r.get("firstPublicationDate"),
                      "pub_types": (r.get("pubTypeList") or {}).get("pubType", []),
                      "title": r.get("title"), "journal": ((r.get("journalInfo") or {}).get("journal") or {}).get("title")}
    return out


def fulltext(pmcid):
    raw = get(f"{EPMC}/{pmcid}/fullTextXML")
    if not raw:
        return None, None
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None, None
    return " ".join(" ".join(root.itertext()).split()), root.get("article-type")


def disclosure_in(text):
    """Same instrument as E1: the first sentence disclosing AI use for writing the manuscript."""
    for s in SENT.findall(text):
        if (re.search(TOOL, s, re.I) and re.search(USE, s, re.I) and re.search(WRITING, s, re.I)
                and re.search(SCOPE, s, re.I)):
            terms = set(m.lower() for m in re.findall(WRITING, s, re.I)) - {"edit"}
            if re.search(GRAPHICS_ONLY, s, re.I) and not terms:
                continue
            if is_false_positive(s):
                continue
            return s.strip()[:700]
    return None


def frame_counts():
    out = {}
    for label, q in QUERIES.items():
        out[label] = {}
        for y in (2023, 2024, 2025, 2026, 2027):
            term = f"{q} AND {y}[pdat] AND open access[filter]"
            raw = get(f"{ESEARCH}?{urllib.parse.urlencode({'db': 'pmc', 'term': term, 'retmode': 'json', 'retmax': 0})}")
            out[label][y] = int(json.loads(raw)["esearchresult"]["count"])
    # union across queries by year (papers can match several phrasings)
    union = "(" + " OR ".join(f"({q})" for q in QUERIES.values()) + ")"
    out["any"] = {}
    for y in (2023, 2024, 2025, 2026, 2027):
        term = f"{union} AND {y}[pdat] AND open access[filter]"
        raw = get(f"{ESEARCH}?{urllib.parse.urlencode({'db': 'pmc', 'term': term, 'retmode': 'json', 'retmax': 0})}")
        out["any"][y] = int(json.loads(raw)["esearchresult"]["count"])
    return out


def main():
    U = {u["doi"]: u for u in json.load(open("discloser_units.json")) if u["eligible"]}
    C = [c for c in json.load(open("control_units.json")) if not c.get("unmatched")]
    pairs = [(U[c["matched_to"]], c) for c in C if c["matched_to"] in U]
    dois = sorted({u["doi"].lower() for d, c in pairs for u in (d, c)})
    print("metadata for", len(dois), "index papers", file=sys.stderr)
    M = meta(dois)
    # prefetch all full texts in parallel (cached), then process sequentially
    from concurrent.futures import ThreadPoolExecutor
    ids = [x for d, c in pairs for x in (d.get("pmcid"), c.get("pmcid") or (M.get(c["doi"].lower()) or {}).get("pmcid")) if x]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda p: get(f"{EPMC}/{p}/fullTextXML"), ids))
    ctrl = {}
    for k, (d, c) in enumerate(pairs):
        doi = c["doi"].lower()
        pmcid = c.get("pmcid") or (M.get(doi) or {}).get("pmcid")
        rec = {"pair": k, "doi": doi, "pmcid": pmcid, "fulltext": False}
        if pmcid:
            text, atype = fulltext(pmcid)
            if text:
                rec.update(fulltext=True, article_type=atype, n_chars=len(text), disclosure=disclosure_in(text),
                           ai_mentions=[s.strip()[:300] for s in SENT.findall(text) if AI_ANY.search(s)][:6])
        ctrl[k] = rec
        if k % 50 == 0:
            print(k, file=sys.stderr, flush=True)
    disc_type = {}
    for k, (d, c) in enumerate(pairs):
        text, atype = fulltext(d["pmcid"]) if d.get("pmcid") else (None, None)
        disc_type[k] = atype
    out = {"meta": M, "controls": ctrl, "discloser_article_type": disc_type, "frame_counts": frame_counts()}
    json.dump(out, open("e8_verify.json", "w"), indent=1)
    n_ft = sum(r["fulltext"] for r in ctrl.values())
    n_disc = sum(bool(r.get("disclosure")) for r in ctrl.values())
    n_any = sum(bool(r.get("ai_mentions")) for r in ctrl.values())
    print(f"controls {len(ctrl)}, with full text {n_ft}, explicit disclosure {n_disc}, any AI mention {n_any}",
          file=sys.stderr)
    print(json.dumps(out["frame_counts"], indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
