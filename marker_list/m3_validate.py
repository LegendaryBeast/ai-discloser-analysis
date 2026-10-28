"""M3: validate the frozen marker list on the held-out half (odd PMIDs): marker-word index by year,
share of abstracts with >=1 marker, and the same without the 5 most frequent markers."""
import gzip, json, re
from collections import Counter

TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*")
L = json.load(open("marker_list_v1.json"))
WORDS = set(L["words"])


def run(half):
    tok, hits, docs, any_hit, word_docs = Counter(), Counter(), Counter(), Counter(), Counter()
    with gzip.open("corpus.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["pmid"]) % 2 != half:
                continue
            y, toks = rec["year"], TOKEN.findall(rec["abstract"].lower())
            m = [t for t in toks if t in WORDS]
            tok[y] += len(toks); hits[y] += len(m); docs[y] += 1; any_hit[y] += bool(m)
            word_docs.update((y, w) for w in set(m))
    return tok, hits, docs, any_hit, word_docs


def main():
    tok, hits, docs, any_hit, word_docs = run(1)
    top5 = [w for w, _ in Counter({w: word_docs[(2025, w)] for w in WORDS}).most_common(5)]
    out, lines = {"top5": top5, "years": {}}, ["year | n abstracts | MWI per 1,000 tokens | abstracts with >=1 marker | MWI without top-5"]
    with gzip.open("corpus.jsonl.gz", "rt") as f:
        tok5, hit5 = Counter(), Counter()
        rest = WORDS - set(top5)
        for line in f:
            rec = json.loads(line)
            if int(rec["pmid"]) % 2 != 1:
                continue
            toks = TOKEN.findall(rec["abstract"].lower())
            tok5[rec["year"]] += len(toks); hit5[rec["year"]] += sum(t in rest for t in toks)
    for y in sorted(docs):
        row = {"n": docs[y], "mwi": 1000 * hits[y] / tok[y], "share_any": any_hit[y] / docs[y],
               "mwi_no_top5": 1000 * hit5[y] / tok5[y]}
        out["years"][y] = row
        lines.append(f"{y} | {row['n']} | {row['mwi']:.3f} | {100 * row['share_any']:.1f}% | {row['mwi_no_top5']:.3f}")
    json.dump(out, open("validation.json", "w"), indent=1)
    open("validation.txt", "w").write("\n".join(lines) + f"\n\ntop-5 markers in 2025: {', '.join(top5)}\n")
    print("\n".join(lines)); print("top5:", top5)


if __name__ == "__main__":
    main()
