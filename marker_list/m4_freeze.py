"""M4: freeze the marker list. Reads candidates.csv and classification.csv (word, decision, reason; decision =
style | content), writes marker_list_v1.json with the style words, derivation parameters and a SHA-256 hash,
plus the derivation-corpus PMIDs so the main study can exclude them."""
import csv, datetime, gzip, hashlib, json

cand = {r["word"]: r for r in csv.DictReader(open("candidates.csv"))}
cls = {r["word"]: r for r in csv.DictReader(open("classification.csv"))}
missing = set(cand) - set(cls)
assert not missing, f"unclassified candidates: {sorted(missing)}"
words = sorted(w for w, r in cls.items() if r["decision"] == "style" and w in cand)
digest = hashlib.sha256("\n".join(words).encode()).hexdigest()
stats = json.load(open("derivation_stats.json"))
json.dump({
    "version": "v1",
    "frozen_on": datetime.date.today().isoformat(),
    "sha256_of_sorted_words": digest,
    "n_words": len(words),
    "words": words,
    "word_stats": {w: {k: cand[w][k] for k in ("p2021", "p2022", "p2024", "p2025", "ratio2024", "ratio2025")}
                   for w in words},
    "excluded_content_words": sorted(w for w, r in cls.items() if r["decision"] == "content"),
    "derivation": {"corpus": "Random PubMed sample: 10 weekdays per year (Entrez date), 2019-22 and 2024-25; "
                             "English journal articles with abstracts; even PMIDs only",
                   "expected_frequency": "p2022 + max(p2022 - p2021, 0) * (year - 2022)",
                   "n_by_year": stats["n_by_year"],
                   "thresholds": stats["thresholds"],
                   "excluded_pilot_author_abstracts": stats["excluded_pilot_author_abstracts"],
                   "tokenizer": "lowercase, regex [a-z]+(?:-[a-z]+)*, document frequency"},
    "use_rule": "Count tokens matching any listed word form, per 1,000 tokens (MWI). Study texts whose PMID is in "
                "derivation_pmids.txt.gz are excluded from the outcome analysis.",
}, open("marker_list_v1.json", "w"), indent=1)
with gzip.open("corpus.jsonl.gz", "rt") as f, gzip.open("derivation_pmids.txt.gz", "wt") as out:
    for line in f:
        pmid = json.loads(line)["pmid"]
        if int(pmid) % 2 == 0:
            out.write(pmid + "\n")
print(f"{len(words)} style words frozen, sha256 {digest}")
