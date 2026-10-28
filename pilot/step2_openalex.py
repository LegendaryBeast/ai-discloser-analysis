"""Step 2: resolve the sampled disclosure papers in OpenAlex, pick first / corresponding / last authors,
and count each author's articles in the 2014-16 and 2019-21 windows (any position, and as first author)."""
import json, sys, time, urllib.parse, urllib.request

OA = "https://api.openalex.org"
# Not AI-use-for-writing disclosures (graphics only, empty template heading, false matches); see README.
EXCLUDE_IDX = {0, 19, 24, 43, 51}
WINDOWS = {"w2015": (2014, 2016), "w2020": (2019, 2021)}
YEARS = [y for lo, hi in WINDOWS.values() for y in range(lo, hi + 1)]


def get(path, params):
    url = f"{OA}/{path}?{urllib.parse.urlencode(params)}"
    for i in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.loads(r.read())
        except Exception as e:
            if i == 3:
                raise
            time.sleep(3 * (i + 1))


def resolve_papers(dois):
    found = {}
    for i in range(0, len(dois), 50):
        d = get("works", {"filter": "doi:" + "|".join(dois[i:i + 50]), "per_page": 50,
                          "select": "id,doi,publication_year,authorships"})
        for w in d["results"]:
            found[w["doi"].replace("https://doi.org/", "").lower()] = w
    return found


def author_history(aid):
    """Articles/reviews by this author in the two windows, by year, any position vs first position."""
    rows, page = [], 1
    while True:
        d = get("works", {"filter": f"author.id:{aid},publication_year:{'|'.join(map(str, YEARS))},type:article|review",
                          "per_page": 200, "page": page, "select": "id,publication_year,authorships"})
        rows += d["results"]
        if len(rows) >= d["meta"]["count"] or page >= 3:
            break
        page += 1
    out = {}
    for name, (lo, hi) in WINDOWS.items():
        ws = [w for w in rows if lo <= w["publication_year"] <= hi]
        first = [w for w in ws if any(a["author_position"] == "first" and (a["author"].get("id") or "").endswith(aid)
                                      for a in w["authorships"])]
        out[name] = len(ws)
        out[name + "_first"] = len(first)
    return out


def main():
    sample = json.load(open("sample_papers.json"))["sample"]
    papers = [p for i, p in enumerate(sample) if i not in EXCLUDE_IDX]
    oa = resolve_papers([p["doi"] for p in papers])
    cache, results = {}, []
    for p in papers:
        w = oa.get(p["doi"])
        rec = {k: p[k] for k in ("query", "pmcid", "doi", "title", "journal", "year", "disclosure")}
        rec["in_openalex"] = bool(w)
        if w:
            auths = w["authorships"]
            rec["n_authors"] = len(auths)
            roles = {"first": next((a for a in auths if a["author_position"] == "first"), None),
                     "last": next((a for a in auths if a["author_position"] == "last"), None),
                     "corresponding": next((a for a in auths if a.get("is_corresponding")), None)}
            rec["corresponding_flagged"] = roles["corresponding"] is not None
            for role, a in roles.items():
                if not a or not a["author"].get("id"):
                    rec[role] = None
                    continue
                aid = a["author"]["id"].rsplit("/", 1)[-1]
                if aid not in cache:
                    cache[aid] = author_history(aid)
                    time.sleep(0.1)
                rec[role] = {"id": aid, "name": a["author"]["display_name"], "orcid": a["author"].get("orcid"),
                             "countries": a.get("countries"), **cache[aid]}
        results.append(rec)
        print(p["pmcid"], rec.get("n_authors"), file=sys.stderr)
    json.dump(results, open("author_histories.json", "w"), indent=1)
    print(f"papers: {len(papers)}, resolved: {sum(r['in_openalex'] for r in results)}, authors queried: {len(cache)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
