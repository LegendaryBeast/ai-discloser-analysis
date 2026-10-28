"""Rebuild the text files that the public release cannot redistribute, from identifiers, and verify every text
against the SHA-256 recorded when the analysis was run.

  experiment/dataset.csv                 main-sample abstracts (W1, W2 and W3 index papers)
  experiment/replication_dataset.csv     2024-25 replication sample
  experiment/replication2026_dataset.csv 2026 replication sample
  marker_list/corpus.jsonl.gz            218,895 PubMed abstracts used to derive and validate the lexicon
  marker_list/population_2026.jsonl.gz   50,274 PubMed abstracts from 2026 (benchmark and refresh test)

Abstracts come from OpenAlex (by work ID) or PubMed (by PMID), exactly as in the original pipeline. Usage:
  python3 rehydrate.py                 # everything (about 1,000 OpenAlex and 1,400 PubMed requests)
  python3 rehydrate.py --only texts    # only the experiment datasets
  python3 rehydrate.py --limit 200     # quick check on the first 200 texts of each file
An OpenAlex API key (free) in experiment/.env as OPENALEX_API_KEY=... raises the daily request limit.
Writes rehydration_report.json with the number of texts that match, differ (changed upstream) or are missing."""
import argparse, csv, gzip, hashlib, json, os, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(ROOT, "experiment"), os.path.join(ROOT, "marker_list")]
os.chdir(os.path.join(ROOT, "experiment"))
import oa                                   # noqa: E402  (cached OpenAlex client of the pipeline)
from e4_dataset import clean                # noqa: E402  (same text cleaning as the pipeline)
from m1_download import parse              # noqa: E402  (same PubMed parsing as the lexicon corpus)

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
sha = lambda s: hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def efetch(pmids):
    """PubMed XML for a list of PMIDs, 200 per request."""
    for i in range(0, len(pmids), 200):
        q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids[i:i + 200]), "retmode": "xml",
                                    "tool": "ai_style_study_rehydrate"})
        for t in range(6):
            try:
                time.sleep(0.4)
                yield urllib.request.urlopen(f"{EFETCH}?{q}", timeout=180).read()
                break
            except Exception:
                time.sleep(5 * (t + 1))


def pubmed_texts(pmids):
    """PMID -> abstract text joined as in e4_dataset.pubmed_abstracts."""
    out = {}
    for xml in efetch(sorted(set(pmids))):
        for art in ET.fromstring(xml).iter("PubmedArticle"):
            parts = art.findall("MedlineCitation/Article/Abstract/AbstractText")
            if parts:
                out[art.findtext("MedlineCitation/PMID")] = " ".join(" ".join(p.itertext()) for p in parts)
    return out


def openalex_texts(ids):
    """OpenAlex work ID -> abstract reconstructed from the inverted index (oa.abstract)."""
    out, ids = {}, sorted(set(ids))
    for i in range(0, len(ids), 50):
        d = oa.oa_get("works", {"filter": "openalex_id:" + "|".join(ids[i:i + 50]), "per_page": 50,
                                "select": "id,abstract_inverted_index"})
        for w in d["results"]:
            out[w["id"].rsplit("/", 1)[-1]] = oa.abstract(w)
    return out


def texts(limit):
    man = list(csv.DictReader(gzip.open(os.path.join(ROOT, "experiment", "texts_manifest.csv.gz"), "rt")))
    report = {}
    for name in ("dataset", "replication_dataset", "replication2026_dataset"):
        rows = [r for r in man if r["file"] == name][:limit or None]
        if not rows:
            continue
        oa_t = openalex_texts([r["work_id"] for r in rows if r["source"] == "openalex"])
        pm_t = pubmed_texts([r["pmid"] for r in rows if r["source"] == "pubmed"])
        cols = [c for c in man[0] if c not in ("file", "source", "pmid", "sha256")] + ["text"]
        stat = {"match": 0, "changed": 0, "missing": 0}
        with open(f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                raw = oa_t.get(r["work_id"]) if r["source"] == "openalex" else pm_t.get(r["pmid"])
                if not raw:
                    stat["missing"] += 1
                    continue
                txt, _ = clean(raw)
                stat["match" if sha(txt) == r["sha256"] else "changed"] += 1
                w.writerow({**{c: r[c] for c in cols if c != "text"}, "text": txt})
        report[name] = stat
        print(name, stat, file=sys.stderr, flush=True)
    return report


def corpus(manifest, out_path, fields, limit):
    man = list(csv.DictReader(gzip.open(manifest, "rt")))[:limit or None]
    want = {r["pmid"]: r for r in man}
    got = {}
    for xml in efetch(list(want)):
        for rec in parse(xml):
            if rec["pmid"] in want:
                got[rec["pmid"]] = rec
    stat = {"match": 0, "changed": 0, "missing": 0}
    with gzip.open(out_path, "wt") as f:
        for p, r in want.items():
            rec = got.get(p)
            if not rec:
                stat["missing"] += 1
                continue
            stat["match" if sha(rec["abstract"]) == r["sha256"] else "changed"] += 1
            extra = {k: (int(r[k]) if k == "year" else r[k]) for k in fields}
            f.write(json.dumps({**({k: rec[k] for k in ("pmid", "journal", "authors", "abstract")}
                                   if "year" in fields else {"pmid": rec["pmid"], "abstract": rec["abstract"]}),
                                **extra}) + "\n")
    print(os.path.basename(out_path), stat, file=sys.stderr, flush=True)
    return stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("texts", "corpora"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    rep = {}
    if a.only in (None, "texts"):
        rep.update(texts(a.limit))
    if a.only in (None, "corpora"):
        ml = os.path.join(ROOT, "marker_list")
        rep["corpus"] = corpus(os.path.join(ml, "corpus_manifest.csv.gz"), os.path.join(ml, "corpus.jsonl.gz"),
                               ("year", "edat"), a.limit)
        rep["population_2026"] = corpus(os.path.join(ml, "population_2026_manifest.csv.gz"),
                                        os.path.join(ml, "population_2026.jsonl.gz"), (), a.limit)
    json.dump(rep, open(os.path.join(ROOT, "rehydration_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
