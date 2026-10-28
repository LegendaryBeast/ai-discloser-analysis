"""Step 1: collect candidate 2025-26 open-access research articles with AI-use disclosure statements
from Europe PMC, draw a seeded random sample, and confirm each disclosure in the full text."""
import json, random, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
BASE = '(PUB_YEAR:2025 OR PUB_YEAR:2026) AND OPEN_ACCESS:y AND PUB_TYPE:"research-article" AND HAS_FT:y'

# Disclosure phrasings, labelled by the template they come from.
QUERIES = {
    "elsevier_template": '"During the preparation of this work the author"',
    "chatgpt_language": '("used ChatGPT to improve" OR "used ChatGPT to enhance" OR "ChatGPT was used to improve" '
                        'OR "ChatGPT was used for language" OR "used ChatGPT for language")',
    "genai_generic": '("generative AI" AND ("to improve the readability" OR "language editing" OR "language polishing"))',
}
PER_QUERY_POOL = 300   # candidates pulled per phrasing before sampling
SAMPLE_PER_QUERY = {"elsevier_template": 25, "chatgpt_language": 15, "genai_generic": 15}  # oversample; some fail checks
SEED = 2026

# Sentence-level pattern used to confirm that the paper *discloses* AI use (not just studies it).
DISCLOSURE_RE = re.compile(
    r"[^.]*\b(ChatGPT|GPT-?4o?|GPT-?3\.5|generative AI|large language model|LLM|Claude|Gemini|Copilot|DeepSeek|"
    r"Grammarly|QuillBot|AI-assisted|artificial intelligence)\b[^.]*\b(used|employed|utili[sz]ed|assist)[^.]*"
    r"\b(language|readability|grammar|writing|edit|polish|proofread|clarity|translat|wording|English)[^.]*\.",
    re.I)


def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def search(query, n):
    out, cursor = [], "*"
    while len(out) < n:
        params = {"query": f"{query} AND {BASE}", "format": "json", "pageSize": 100,
                  "resultType": "lite", "cursorMark": cursor}
        d = json.loads(get(f"{EPMC}/search?{urllib.parse.urlencode(params)}"))
        res = d["resultList"]["result"]
        if not res:
            break
        out += res
        if d.get("nextCursorMark") in (None, cursor):
            break
        cursor = d["nextCursorMark"]
    return d["hitCount"], out[:n]


def disclosure_sentence(pmcid):
    xml = get(f"{EPMC}/{pmcid}/fullTextXML")
    text = " ".join(ET.fromstring(xml).itertext())
    text = re.sub(r"\s+", " ", text)
    m = DISCLOSURE_RE.search(text)
    return m.group(0).strip()[:600] if m else None


def main():
    rng = random.Random(SEED)
    hits, sample, seen = {}, [], set()
    for label, q in QUERIES.items():
        total, pool = search(q, PER_QUERY_POOL)
        hits[label] = total
        pool = [p for p in pool if p.get("doi") and p.get("pmcid") and p["doi"] not in seen]
        rng.shuffle(pool)
        kept = 0
        for p in pool:
            if kept >= SAMPLE_PER_QUERY[label]:
                break
            try:
                sent = disclosure_sentence(p["pmcid"])
            except Exception as e:
                print("fulltext fail", p["pmcid"], e, file=sys.stderr)
                continue
            if not sent:
                continue
            seen.add(p["doi"])
            sample.append({"query": label, "pmcid": p["pmcid"], "doi": p["doi"].lower(), "title": p["title"],
                           "journal": p.get("journalTitle"), "year": p.get("pubYear"),
                           "authors": p.get("authorString"), "disclosure": sent})
            kept += 1
            time.sleep(0.2)
        print(f"{label}: {total} hits, kept {kept}", file=sys.stderr)
    json.dump({"hit_counts": hits, "seed": SEED, "sample": sample}, open("sample_papers.json", "w"), indent=1)
    print(f"total sampled: {len(sample)}", file=sys.stderr)


if __name__ == "__main__":
    main()
