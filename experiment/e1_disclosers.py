"""E1: random sample of verified AI-disclosing research articles, 2024-26, from Europe PMC.
Pool = all open-access research articles with full text matching any of three disclosure phrasings; a seeded random
order is walked and papers are kept if a sentence in the full text discloses AI use for writing the manuscript."""
import json, os, random, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
BASE = ('(PUB_YEAR:2024 OR PUB_YEAR:2025 OR PUB_YEAR:2026) AND OPEN_ACCESS:y AND PUB_TYPE:"research-article" '
        'AND HAS_FT:y')
QUERIES = {
    "elsevier_template": '"During the preparation of this work the author"',
    "chatgpt_language": '("used ChatGPT to improve" OR "used ChatGPT to enhance" OR "ChatGPT was used to improve" '
                        'OR "ChatGPT was used for language" OR "used ChatGPT for language")',
    "genai_generic": '("generative AI" AND ("to improve the readability" OR "language editing" OR "language polishing"))',
}
TARGET = 1350
SEED = 2027

TOOL = r"(ChatGPT|GPT-?4o?|GPT-?3\.5|generative AI|generative artificial intelligence|large language model|LLMs?|" \
       r"Claude|Gemini|Copilot|DeepSeek|Kimi|Grammarly|QuillBot|PaperPal|DeepL|AI tools?|AI-assisted tools?)"
USE = r"\b(used|uses|using|employed|utili[sz]ed|to assist|assisted (?:with|in)|was applied)\b"
WRITING = r"(language|readability|grammar|grammatical|writing|edit|polish|proofread|clarity|translat|wording|English|fluency)"
SCOPE = r"(manuscript|this (?:work|article|paper|study)|the authors?\b|\bwe used|language edit|polish|proofread|grammar|readability)"
GRAPHICS_ONLY = r"(graphic|figure|image|illustration|picture)"
SENT = re.compile(r"[^.]*\.")


def is_false_positive(s):
    """Filter out non-use declarations, journal policies, guidelines, and study descriptions."""
    if not re.search(r'\b(was used|used\b[^\.]*\b(?:to improve|for language|to edit|to polish|for readability))\b', s, re.I):
        if re.search(r'\b(?:no|neither|none|declare[s]? that no)\s+(?:generative AI|AI tools?|artificial intelligence|LLM)[^\.]*\b(?:were|was|are|have been)\s+(?:used|employed|applied)\b', s, re.I):
            return True
        if re.search(r'\b(?:did not use|have not used|has not used)\s+(?:generative AI|AI tools?|artificial intelligence|ChatGPT|LLM)', s, re.I):
            return True
    if re.search(r'\b(should only be used|authors? (?:must|should|are required to)|guidelines? for|policy:)\b', s, re.I):
        return True
    if re.search(r'(□|\bchecklist\b|\bquestionnaire\b|\bitem reported\b|\byes\s*□\s*no\b|\bdid you use\b)', s, re.I):
        return True
    if re.search(r'\b(Conceptualization|Data curation|Formal analysis|Writing [–-] original draft)\b', s):
        return True
    if re.search(r'\b(this paper introduces|this study evaluates|we evaluate the performance of (?:LLMs|large language models))\b', s, re.I):
        return True
    return False


def get(url):
    headers = {"User-Agent": "ai-writing-study/1.0 (mailto:researcher@institution.edu)"}
    for i in range(100):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 503:
                print(f"Europe PMC search service temporarily unavailable (503); waiting 20s (attempt {i+1}/100)...",
                      file=sys.stderr, flush=True)
                time.sleep(20)
            else:
                print(f"retry HTTP {e.code}: {e}", file=sys.stderr, flush=True)
                time.sleep(min(30, 3 * (i + 1)))
        except Exception as e:
            print("retry", e, file=sys.stderr, flush=True)
            time.sleep(min(30, 3 * (i + 1)))
    raise RuntimeError(url)


NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


def pool_ncbi():
    """Fetch candidate pool from NCBI PMC (fast and reliable)."""
    print("Harvesting candidate pool via NCBI PMC...", file=sys.stderr, flush=True)
    cands, hits = {}, {}
    headers = {"User-Agent": "ai-writing-study/1.0 (mailto:researcher@institution.edu)"}
    for label, q in QUERIES.items():
        term = f'{q} AND 2024:2026[pdat] AND open access[filter]'
        params = {"db": "pmc", "term": term, "retmode": "json", "retmax": 2500}
        url = f"{NCBI_ESEARCH}?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.loads(r.read())
            hits[label] = int(d["esearchresult"]["count"])
            id_list = d["esearchresult"]["idlist"]
        print(f"NCBI {label}: {hits[label]} hits ({len(id_list)} downloaded)", file=sys.stderr, flush=True)

        for i in range(0, len(id_list), 200):
            chunk = id_list[i:i + 200]
            sparams = {"db": "pmc", "id": ",".join(chunk), "retmode": "json"}
            surl = f"{NCBI_ESUMMARY}?" + urllib.parse.urlencode(sparams)
            sreq = urllib.request.Request(surl, headers=headers)
            with urllib.request.urlopen(sreq, timeout=45) as r:
                res = json.loads(r.read()).get("result", {})
                for uid in chunk:
                    doc = res.get(uid, {})
                    doi = next((x["value"] for x in doc.get("articleids", []) if x["idtype"] == "doi"), None)
                    pmid = next((x["value"] for x in doc.get("articleids", []) if x["idtype"] == "pmid"), None)
                    if doi:
                        cands.setdefault(doi.lower(), {
                            "doi": doi.lower(),
                            "pmcid": f"PMC{uid}",
                            "pmid": pmid,
                            "title": doc.get("title"),
                            "journalTitle": doc.get("source"),
                            "pubYear": doc.get("pubdate", "")[:4],
                            "firstPublicationDate": doc.get("pubdate"),
                            "query": label
                        })
            time.sleep(0.35)
    return cands, hits


def pool(query):
    out, cursor = [], "*"
    while True:
        p = {"query": f"{query} AND {BASE}", "format": "json", "pageSize": 1000, "resultType": "lite", "cursorMark": cursor}
        d = json.loads(get(f"{EPMC}/search?{urllib.parse.urlencode(p)}"))
        out += d["resultList"]["result"]
        nxt = d.get("nextCursorMark")
        if not d["resultList"]["result"] or nxt in (None, cursor):
            return d["hitCount"], out
        cursor = nxt


def disclosure(pmcid):
    """Return the first sentence that discloses AI use for writing this manuscript, else None."""
    text = " ".join(" ".join(ET.fromstring(get(f"{EPMC}/{pmcid}/fullTextXML")).itertext()).split())
    for s in SENT.findall(text):
        if (re.search(TOOL, s, re.I) and re.search(USE, s, re.I) and re.search(WRITING, s, re.I)
                and re.search(SCOPE, s, re.I)):
            writing_terms = set(m.lower() for m in re.findall(WRITING, s, re.I)) - {"edit"}
            if re.search(GRAPHICS_ONLY, s, re.I) and not writing_terms:
                continue
            if is_false_positive(s):
                continue
            return s.strip()[:700]
    return None


def main():
    rng = random.Random(SEED)
    cands, hits = {}, {}
    if os.path.exists("candidates_pool.json"):
        d_pool = json.load(open("candidates_pool.json"))
        cands, hits = d_pool["cands"], d_pool["hits"]
        print(f"Loaded existing candidates pool: {len(cands)} papers", file=sys.stderr)
    else:
        try:
            cands, hits = pool_ncbi()
        except Exception as e:
            print(f"NCBI pool failed ({e}), falling back to Europe PMC...", file=sys.stderr)
            for label, q in QUERIES.items():
                hits[label], res = pool(q)
                for r in res:
                    if r.get("doi") and r.get("pmcid"):
                        cands.setdefault(r["doi"].lower(), {**r, "query": label})
                print(f"{label}: {hits[label]} hits", file=sys.stderr)
        json.dump({"cands": cands, "hits": hits}, open("candidates_pool.json", "w"))

    order = sorted(cands)
    rng.shuffle(order)
    kept, rejected = [], 0
    if os.path.exists("disclosers.json"):
        try:
            prev = json.load(open("disclosers.json"))
            kept = prev.get("papers", [])
            rejected = prev.get("rejected", 0)
            print(f"Resuming e1: already kept {len(kept)}/{TARGET}, rejected {rejected}", file=sys.stderr)
        except Exception:
            pass
    done_dois = {p["doi"] for p in kept}
    for doi in order:
        if len(kept) >= TARGET:
            break
        if doi in done_dois:
            continue
        r = cands[doi]
        try:
            s = disclosure(r["pmcid"])
        except Exception as e:
            print("fulltext fail", r["pmcid"], e, file=sys.stderr)
            continue
        if not s:
            rejected += 1
            continue
        kept.append({"doi": doi, "pmcid": r["pmcid"], "pmid": r.get("pmid"), "title": r.get("title"),
                     "journal": r.get("journalTitle"), "issn": r.get("journalIssn"), "year": r.get("pubYear"),
                     "pub_date": r.get("firstPublicationDate"), "query": r["query"], "disclosure": s})
        if len(kept) % 50 == 0:
            print(f"Progress: kept {len(kept)}/{TARGET}, rejected {rejected}", file=sys.stderr, flush=True)
            json.dump({"hit_counts": hits, "pool_size": len(cands), "seed": SEED, "screened": len(kept) + rejected,
                       "rejected": rejected, "papers": kept}, open("disclosers.json", "w"), indent=1)
        time.sleep(0.12)
    json.dump({"hit_counts": hits, "pool_size": len(cands), "seed": SEED, "screened": len(kept) + rejected,
               "rejected": rejected, "papers": kept}, open("disclosers.json", "w"), indent=1)
    print(f"Final: pool {len(cands)}, kept {len(kept)}, rejected {rejected}", file=sys.stderr)


if __name__ == "__main__":
    main()
