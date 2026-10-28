"""M5 (revision): population benchmark for 2026. Same sampling rule as M1 (random weekdays, one per tenth of the
window, all English journal articles with an abstract and that Entrez date), over 1 Jan to 30 Sep 2026 so that
indexing is complete. The 2026 abstracts were not used to derive the lexicon v1. Writes population_2026.json and the
abstracts themselves (population_2026.jsonl.gz), from which m6 derives a 2026 excess-word lexicon for a refresh test."""
import datetime as dt, gzip, json, random, re, sys, urllib.parse

from m1_download import EU, get, parse, search

SEED, DAYS = 2030, 10
TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*")
WORDS = set(json.load(open("marker_list_v1.json"))["words"])
TOP5 = set(json.load(open("validation.json"))["top5"])


def days(rng):
    start, n = dt.date(2026, 1, 1), (dt.date(2026, 9, 30) - dt.date(2026, 1, 1)).days + 1
    out = []
    for k in range(DAYS):
        lo, hi = k * n // DAYS, (k + 1) * n // DAYS - 1
        while True:
            d = start + dt.timedelta(days=rng.randint(lo, hi))
            if d.weekday() < 5:
                out.append(d)
                break
    return out


def main():
    rng = random.Random(SEED)
    tok = hit = hit5 = docs = anyh = 0
    corpus = gzip.open("population_2026.jsonl.gz", "wt")  # kept for deriving a 2026 lexicon (m6)
    for day in days(rng):
        r = search(2026, day)
        count = min(int(r["count"]), 9999)
        for start in range(0, count, 1000):
            p = urllib.parse.urlencode({"db": "pubmed", "query_key": r["querykey"], "WebEnv": r["webenv"],
                                        "retstart": start, "retmax": 1000, "retmode": "xml", "tool": "ai_style_study"})
            for rec in parse(get(f"{EU}/efetch.fcgi?{p}")):
                corpus.write(json.dumps({"pmid": rec["pmid"], "abstract": rec["abstract"]}) + "\n")
                t = TOKEN.findall(rec["abstract"].lower())
                m = [x for x in t if x in WORDS]
                tok += len(t); hit += len(m); hit5 += sum(x not in TOP5 for x in m); docs += 1; anyh += bool(m)
        print(day, count, docs, file=sys.stderr, flush=True)
    corpus.close()
    out = {"year": 2026, "n": docs, "mwi": 1000 * hit / tok, "share_any": anyh / docs, "mwi_no_top5": 1000 * hit5 / tok,
           "window": "2026-01-01 to 2026-09-30", "seed": SEED}
    json.dump(out, open("population_2026.json", "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
