"""E4: assemble the paper-level dataset (W1, W2 history abstracts + the W3 index abstract) for matched pairs.
Abstracts missing in OpenAlex are fetched from PubMed. Writes dataset.csv."""
import csv, json, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
LABELS = (r"background|objectives?|aims?|purpose|introduction|methods?|materials and methods|design|setting|"
          r"participants|patients|interventions?|measurements?|main outcome measures?|results?|findings|"
          r"conclusions?|interpretation|significance|context|importance|limitations|clinical relevance")
LABEL_RE = re.compile(rf"(?:^|(?<=[.\s]))({LABELS})\s*[:.]\s*", re.I)
COPYRIGHT_RE = re.compile(r"(©|\(c\)\s*\d{4}|copyright\b).*$", re.I)
MIN_WORDS = 100


def pubmed_abstracts(pmids):
    cache_file = "pubmed_cache.json"
    cache = {}
    try:
        cache = json.load(open(cache_file))
    except Exception:
        pass
    missing = [p for p in pmids if p not in cache]
    if missing:
        for i in range(0, len(missing), 200):
            q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(missing[i:i + 200]), "retmode": "xml",
                                        "tool": "ai_style_study"})
            for t in range(4):
                try:
                    time.sleep(0.4)
                    xml = urllib.request.urlopen(f"{EU}?{q}", timeout=120).read()
                    break
                except Exception:
                    time.sleep(4 * (t + 1))
            for art in ET.fromstring(xml).iter("PubmedArticle"):
                parts = art.findall("MedlineCitation/Article/Abstract/AbstractText")
                if parts:
                    cache[art.findtext("MedlineCitation/PMID")] = " ".join(" ".join(p.itertext()) for p in parts)
        try:
            json.dump(cache, open(cache_file, "w"))
        except Exception:
            pass
    return {p: cache[p] for p in pmids if p in cache}


def clean(text):
    text = " ".join((text or "").split())
    structured = len(LABEL_RE.findall(text)) >= 2
    text = COPYRIGHT_RE.sub("", LABEL_RE.sub(" ", text))
    text = re.sub(r"^\s*abstract\s*[:.]?\s*", "", text, flags=re.I)
    return " ".join(text.split()), structured


def main():
    disc = {u["doi"]: u for u in json.load(open("discloser_units.json")) if u["eligible"]}
    ctrl = [c for c in json.load(open("control_units.json")) if not c.get("unmatched")]
    pairs = [(disc[c["matched_to"]], c) for c in ctrl if c["matched_to"] in disc]
    need = [w["pmid"] for d, c in pairs for u in (d, c) for w in u["history"] if not w["abstract"] and w["pmid"]]
    need += [u["pmid"] for d, c in pairs for u in (d, c) if not u.get("index_abstract") and u.get("pmid")]
    pm = pubmed_abstracts(sorted(set(need)))
    rows = []
    for k, (d, c) in enumerate(pairs):
        for group, u in ((1, d), (0, c)):
            items = [("W1" if w["year"] < 2017 else "W2", w["year"], w["id"], w["n_authors"], w["position"],
                      w["abstract"] or pm.get(w["pmid"] or "")) for w in u["history"]]
            w3_date = u.get("pub_date") or d.get("pub_date") or "2025"
            w3_year = int(str(w3_date)[:4])
            items.append(("W3", w3_year, u["work_id"], u["n_authors"], u["anchor"]["rule"],
                          u.get("index_abstract") or pm.get(u.get("pmid") or "")))
            for wave, year, wid, n_auth, pos, text in items:
                txt, structured = clean(text)
                if len(txt.split()) < MIN_WORDS:
                    continue
                rows.append({"pair": k, "group": group, "author": u["anchor"]["id"], "wave": wave, "year": year,
                             "work_id": wid, "n_authors": n_auth, "position": pos, "structured": int(structured),
                             "country": (u["anchor"]["countries"] or ["NA"])[0], "topic": u.get("topic"),
                             "text": txt})
    with open("dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"pairs {len(pairs)}, rows {len(rows)}, PubMed-filled {len(pm)}", file=sys.stderr)


if __name__ == "__main__":
    main()
