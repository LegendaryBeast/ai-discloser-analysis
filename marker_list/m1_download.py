"""M1: download a random PubMed abstract corpus for deriving the marker-word list.
Sampling: for each year, 10 random weekdays (one per tenth of the year, seed 2026); all English journal-article
records with an abstract whose Entrez date is that day and whose publication year is that year."""
import datetime as dt, gzip, json, os, random, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
YEARS = [2019, 2020, 2021, 2022, 2024, 2025]
DAYS_PER_YEAR = 10
SEED = 2026
OUT = "corpus.jsonl.gz"
LOG = "download.log"


def get(url, data=None):
    for i in range(6):
        try:
            time.sleep(0.4)  # stay under NCBI's 3 requests/second without an API key
            with urllib.request.urlopen(url, data=data, timeout=180) as r:
                return r.read()
        except Exception as e:
            print("retry", i, e, file=sys.stderr)
            time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def sample_days(year, rng):
    start, n = dt.date(year, 1, 1), (dt.date(year, 12, 31) - dt.date(year, 1, 1)).days + 1
    days = []
    for k in range(DAYS_PER_YEAR):
        lo, hi = k * n // DAYS_PER_YEAR, (k + 1) * n // DAYS_PER_YEAR - 1
        while True:
            d = start + dt.timedelta(days=rng.randint(lo, hi))
            if d.weekday() < 5:
                days.append(d)
                break
    return days


def parse(xml):
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        mc = art.find("MedlineCitation")
        abst = mc.find("Article/Abstract")
        if abst is None:
            continue
        text = " ".join(" ".join(a.itertext()).strip() for a in abst.findall("AbstractText"))
        authors = []
        for au in mc.findall("Article/AuthorList/Author"):
            orcid = next((i.text for i in au.findall("Identifier") if i.get("Source") == "ORCID"), None)
            authors.append({"last": au.findtext("LastName"), "fore": au.findtext("ForeName"),
                            "orcid": (orcid or "").strip().rsplit("/", 1)[-1] or None})
        yield {"pmid": mc.findtext("PMID"), "journal": mc.findtext("Article/Journal/ISOAbbreviation"),
               "authors": authors, "abstract": " ".join(text.split())}


def search(year, day):
    term = f"{year}[dp] AND {day:%Y/%m/%d}[edat] AND hasabstract AND english[la] AND journal article[pt]"
    q = urllib.parse.urlencode({"db": "pubmed", "term": term, "usehistory": "y", "retmax": 0,
                                "retmode": "json", "tool": "ai_style_study"})
    return json.loads(get(f"{EU}/esearch.fcgi?{q}"))["esearchresult"]


def main():
    rng, rng_swap = random.Random(SEED), random.Random(SEED + 1)
    plan = [(y, d) for y in YEARS for d in sample_days(y, rng)]
    # Resume: keep only days that finished (logged as "written"); a partially written day is dropped.
    done = set(re.findall(r"^\d{4} (\S+): \d+ hits, \d+ written", open(LOG).read(), re.M)) if os.path.exists(LOG) else set()
    keep = []
    if os.path.exists(OUT):
        with gzip.open(OUT, "rt") as f:
            keep = [l for l in f if json.loads(l)["edat"] in done]
    n_total = len(keep)
    with gzip.open(OUT, "wt") as f:
        f.writelines(keep)
        for year, day in plan:
            r = search(year, day)
            # E-utilities cannot page past 9,999 records; such (rare) days are swapped for another weekday
            # within +-10 days, drawn from a separate seeded generator so the rest of the plan is unchanged.
            while int(r["count"]) > 9999:
                print(f"{year} {day}: {r['count']} hits > 9999, swapping day", file=sys.stderr, flush=True)
                day = max(dt.date(year, 1, 1), min(dt.date(year, 12, 31), day + dt.timedelta(days=rng_swap.randint(-10, 10))))
                if day.weekday() >= 5:
                    continue
                r = search(year, day)
            if f"{day}" in done:
                continue
            count, n_day = int(r["count"]), 0
            for start in range(0, count, 1000):
                p = urllib.parse.urlencode({"db": "pubmed", "query_key": r["querykey"], "WebEnv": r["webenv"],
                                            "retstart": start, "retmax": 1000, "retmode": "xml",
                                            "tool": "ai_style_study"})
                for rec in parse(get(f"{EU}/efetch.fcgi?{p}")):
                    rec["year"], rec["edat"] = year, f"{day}"
                    f.write(json.dumps(rec) + "\n")
                    n_day += 1
            n_total += n_day
            print(f"{year} {day}: {count} hits, {n_day} written (total {n_total})", file=sys.stderr, flush=True)
            with open(LOG, "a") as lg:
                lg.write(f"{year} {day}: {count} hits, {n_day} written (total {n_total})\n")


if __name__ == "__main__":
    main()
