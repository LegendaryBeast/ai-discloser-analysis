"""Step 4: for authors retained in both windows, classify each 2014-16 / 2019-21 paper by how its text can be obtained:
PMC open-access subset (machine-readable XML), in PMC but not OA subset, preprint server, other OA, or abstract only."""
import json, sys, time, urllib.parse, urllib.request

OA = "https://api.openalex.org"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
YEARS = "2014|2015|2016|2019|2020|2021"
PREPRINT_SERVERS = ("arxiv", "biorxiv", "medrxiv", "ssrn", "research square", "preprints.org", "psyarxiv", "osf")


def get(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.loads(r.read())
        except Exception:
            if i == 3:
                raise
            time.sleep(3 * (i + 1))


def oa(params):
    return get(f"{OA}/works?{urllib.parse.urlencode(params)}")


def author_works(aid):
    base = f"author.id:{aid},publication_year:{YEARS},type:article|review"
    rows, page = [], 1
    while True:
        d = oa({"filter": base, "per_page": 200, "page": page,
                "select": "id,doi,ids,publication_year,open_access,locations,authorships"})
        rows += d["results"]
        if len(rows) >= d["meta"]["count"] or page >= 3:
            break
        page += 1
    no_abs = {w["id"] for w in oa({"filter": base + ",has_abstract:false", "per_page": 200, "select": "id"})["results"]}
    out = []
    for w in rows:
        hosts = " ".join(((l.get("source") or {}).get("display_name") or "") + " " + (l.get("landing_page_url") or "")
                         for l in w["locations"]).lower()
        out.append({"id": w["id"].rsplit("/", 1)[-1], "year": w["publication_year"],
                    "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
                    "pmid": (w["ids"].get("pmid") or "").rsplit("/", 1)[-1],
                    "first": any(a["author_position"] == "first" and (a["author"].get("id") or "").endswith(aid)
                                 for a in w["authorships"]),
                    "is_oa": w["open_access"]["is_oa"],
                    "preprint": any(s in hosts for s in PREPRINT_SERVERS),
                    "has_abstract": w["id"] not in no_abs})
    return out


def epmc_status(works):
    """Look works up in Europe PMC by PMID (or DOI): returns key -> (in_pmc, in_oa_subset)."""
    keys = [("EXT_ID:%s AND SRC:MED" % w["pmid"], w["pmid"]) if w["pmid"] else (f'DOI:"{w["doi"]}"', w["doi"])
            for w in works if w["pmid"] or w["doi"]]
    status = {}
    for i in range(0, len(keys), 20):
        chunk = keys[i:i + 20]
        q = " OR ".join(f"({c})" for c, _ in chunk)
        d = get(f"{EPMC}?{urllib.parse.urlencode({'query': q, 'format': 'json', 'pageSize': 100, 'resultType': 'lite'})}")
        for r in d["resultList"]["result"]:
            st = (r.get("inPMC") == "Y" or bool(r.get("pmcid")), r.get("isOpenAccess") == "Y" and bool(r.get("pmcid")))
            for k in (r.get("pmid"), (r.get("doi") or "").lower()):
                if k:
                    status[k] = st
        time.sleep(0.2)
    return status


def tier(w):
    if w["pmc_oa"]:
        return "pmc_oa"
    if w["preprint"]:
        return "preprint"
    if w["in_pmc"]:
        return "pmc_other"
    if w["is_oa"]:
        return "other_oa"
    return "abstract_only" if w["has_abstract"] else "nothing"


def main():
    R = json.load(open("author_histories.json"))
    retained = {}
    for r in R:
        for role in ("first", "corresponding", "last"):
            a = r.get(role)
            if a and a["w2015"] and a["w2020"]:
                retained.setdefault(a["id"], {"name": a["name"], "orcid": a["orcid"], "roles": set(),
                                              "first_both": bool(a["w2015_first"] and a["w2020_first"])})
                retained[a["id"]]["roles"].add(role)
    print(f"retained authors: {len(retained)}", file=sys.stderr)
    for aid, info in retained.items():
        works = author_works(aid)
        st = epmc_status(works)
        for w in works:
            w["in_pmc"], w["pmc_oa"] = st.get(w["pmid"]) or st.get(w["doi"]) or (False, False)
            w["tier"] = tier(w)
        info["works"], info["roles"] = works, sorted(info["roles"])
        print(aid, info["name"], len(works), file=sys.stderr)
    json.dump(retained, open("fulltext_status.json", "w"), indent=1)


if __name__ == "__main__":
    main()
