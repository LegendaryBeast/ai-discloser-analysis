"""Shared helpers: cached OpenAlex calls (the free tier allows ~1,000 calls/day), author histories, abstracts."""
import hashlib, json, os, sys, time, urllib.parse, urllib.request

OA = "https://api.openalex.org"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)
ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(ENV_FILE):
    for line in open(ENV_FILE):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
W1, W2 = range(2014, 2017), range(2019, 2022)
YEARS = "|".join(map(str, list(W1) + list(W2)))
MIN_BUDGET = 40  # stop spending OpenAlex calls below this many remaining
remaining = None
WORK_FIELDS = "id,doi,ids,publication_year,authorships,abstract_inverted_index,language,primary_topic,type"


class BudgetExhausted(Exception):
    pass


def oa_get(path, params):
    global remaining
    api_key = os.environ.get("OPENALEX_API_KEY")
    if api_key and "api_key" not in params:
        params = {**params, "api_key": api_key}
    url = f"{OA}/{path}?{urllib.parse.urlencode(params)}"
    key = os.path.join(CACHE, hashlib.sha1(url.encode()).hexdigest() + ".json")
    if os.path.exists(key):
        return json.load(open(key))
    if remaining is not None and remaining < MIN_BUDGET:
        raise BudgetExhausted
    headers = {"User-Agent": "ai-writing-study/1.0 (mailto:researcher@institution.edu)"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    for i in range(4):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as r:
                remaining = int(r.headers.get("x-ratelimit-remaining", 9999))
                data = json.loads(r.read())
            json.dump(data, open(key, "w"))
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise BudgetExhausted
            time.sleep(3 * (i + 1))
        except Exception:
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


def resolve(dois):
    out = {}
    for i in range(0, len(dois), 50):
        d = oa_get("works", {"filter": "doi:" + "|".join(dois[i:i + 50]), "per_page": 50, "select": WORK_FIELDS})
        for w in d["results"]:
            if w.get("doi"):
                out[w["doi"].replace("https://doi.org/", "").lower()] = w
    return out


def abstract(w):
    inv = w.get("abstract_inverted_index")
    if not inv:
        return None
    pos = {}
    for word, idx in inv.items():
        for i in idx:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def anchor(w):
    """Pragmatic anchor: the corresponding author, else the last author, else the first author."""
    au = [a for a in w["authorships"] if (a.get("author") or {}).get("id")]
    for rule, pick in (("corresponding", [a for a in au if a.get("is_corresponding")]),
                       ("last", [a for a in au if a["author_position"] == "last"]),
                       ("first", [a for a in au if a["author_position"] == "first"])):
        if pick:
            a = pick[0]
            return {"id": a["author"]["id"].rsplit("/", 1)[-1], "name": a["author"]["display_name"],
                    "orcid": a["author"].get("orcid"), "countries": a.get("countries") or [], "rule": rule}
    return None


def history(aid):
    """English articles/reviews (<=20 authors) by the author in W1 and W2, with abstract text when OpenAlex has it."""
    rows, page = [], 1
    while True:
        d = oa_get("works", {"filter": f"author.id:{aid},publication_year:{YEARS},type:article|review",
                             "per_page": 200, "page": page, "select": WORK_FIELDS})
        rows += d["results"]
        if len(rows) >= d["meta"]["count"] or page >= 2:
            break
        page += 1
    out = []
    for w in rows:
        if (w.get("language") not in (None, "en")) or len(w["authorships"]) > 20:
            continue
        out.append({"id": w["id"].rsplit("/", 1)[-1], "year": w["publication_year"],
                    "pmid": ((w.get("ids") or {}).get("pmid") or "").rsplit("/", 1)[-1] or None,
                    "n_authors": len(w["authorships"]),
                    "position": next((a["author_position"] for a in w["authorships"]
                                      if ((a.get("author") or {}).get("id") or "").endswith(aid)), None),
                    "abstract": abstract(w)})
    return out


def eligible(hist, k=2):
    ok = lambda ws: sum(1 for w in ws if w["abstract"] or w["pmid"]) >= k
    return ok([w for w in hist if w["year"] in W1]) and ok([w for w in hist if w["year"] in W2])
