"""M6 (revision, exploratory refresh test): a 2026 excess-word lexicon derived with the M2 rule.
Expected 2026 frequency = linear extrapolation from 2021-22 on the derivation half (even PMIDs) of the M1 corpus;
observed 2026 frequency = even-PMID half of the 2026 benchmark sample (M5), excluding every index paper of the
experiment. Same thresholds as M2 (ratio >= 1.5, gap >= 0.0005, >= 30 documents), applied to 2026 alone. No coder:
all candidates are kept except AI-topic words fixed in advance (AI_TOPIC), because controls were selected for not
mentioning AI. Writes lexicon_2026.json with four word lists used by experiment/e10_revision.py:
  all     every 2026 candidate (after the AI-topic filter)
  new     2026 candidates that were not among the 347 version-1 candidates (words newly in excess in 2026)
  kept    version-1 style words that are still 2026 candidates
  dropped version-1 style words that are no longer 2026 candidates
  rising_2026  non-v1 words whose document frequency rose >= 1.5x from 2025 to 2026 (the list used in the test)"""
import gzip, hashlib, json, re, sys
from collections import Counter

from m2_excess import TOKEN, MIN_RATIO, MIN_GAP, MIN_DOCS, load_exclusions, is_excluded

AI_TOPIC = {"ai", "artificial", "intelligence", "llm", "llms", "chatgpt", "gpt", "generative", "chatbot", "chatbots",
            "agentic", "agent", "agents", "prompt", "prompts", "prompting", "transformer", "transformers", "multimodal",
            "language", "machine", "learning", "deep", "neural", "algorithm", "algorithms", "model", "models"}


def index_pmids():
    out = set()
    for f in ("discloser_units.json", "control_units.json", "replication_units.json", "replication2026_units.json"):
        try:
            data = json.load(open(f"../experiment/{f}"))
        except FileNotFoundError:
            continue
        for u in data:
            for x in ((u["disc"], u["ctrl"]) if "disc" in u else (u,)):
                if x.get("pmid"):
                    out.add(str(x["pmid"]))
    return out


def main():
    orcids, names = load_exclusions()
    df, n = {2021: Counter(), 2022: Counter()}, Counter()
    with gzip.open("corpus.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            if rec["year"] in (2021, 2022) and int(rec["pmid"]) % 2 == 0 and not is_excluded(rec, orcids, names):
                n[rec["year"]] += 1
                df[rec["year"]].update(set(TOKEN.findall(rec["abstract"].lower())))
    skip = index_pmids()
    d26, n26 = Counter(), 0
    with gzip.open("population_2026.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["pmid"]) % 2 or rec["pmid"] in skip:
                continue
            n26 += 1
            d26.update(set(TOKEN.findall(rec["abstract"].lower())))
    p21 = {w: c / n[2021] for w, c in df[2021].items()}
    p22 = {w: c / n[2022] for w, c in df[2022].items()}
    rows = []
    for w, c in d26.items():
        q = p22.get(w, 0) + max(p22.get(w, 0) - p21.get(w, 0), 0) * 4
        obs = c / n26
        if c >= MIN_DOCS and obs - q >= MIN_GAP and (q == 0 or obs / q >= MIN_RATIO) and w not in AI_TOPIC and w.isalpha():
            rows.append((w, obs, q))
    cand26 = sorted(w for w, _, _ in rows)
    # rising_2026 (Amendment 2): one-year rise from 2025 (M1 derivation half) to 2026, same thresholds
    d25, n25 = Counter(), 0
    with gzip.open("corpus.jsonl.gz", "rt") as f:
        for line in f:
            rec = json.loads(line)
            if rec["year"] == 2025 and int(rec["pmid"]) % 2 == 0 and not is_excluded(rec, orcids, names):
                n25 += 1
                d25.update(set(TOKEN.findall(rec["abstract"].lower())))
    v1all = set(json.load(open("marker_list_v1.json"))["words"]) | set(json.load(open("marker_list_v1.json"))["excluded_content_words"])
    rising = sorted(w for w, c in d26.items() if c >= MIN_DOCS and w.isalpha() and w not in AI_TOPIC and w not in v1all
                    and c / n26 - d25.get(w, 0) / n25 >= MIN_GAP and (d25.get(w, 0) == 0 or (c / n26) / (d25[w] / n25) >= MIN_RATIO))
    v1 = json.load(open("marker_list_v1.json"))
    style, content = set(v1["words"]), set(v1["excluded_content_words"])
    out = {"derivation": {"n_2021": n[2021], "n_2022": n[2022], "n_2026": n26, "excluded_index_pmids": len(skip),
                          "rule": "M2 thresholds applied to 2026 vs. 2021-22 extrapolation; AI-topic words removed"},
           "all": cand26, "new": sorted(set(cand26) - style - content), "kept": sorted(set(cand26) & style),
           "dropped": sorted(style - set(cand26)), "rising_2026": rising, "ai_topic_filter": sorted(AI_TOPIC),
           "note": "all/new use the 4-year extrapolation and are dominated by topical drift; the refresh test uses "
                   "rising_2026 (Amendment 2 of experiment/replication_plan.md), kept and dropped."}
    out["derivation"]["n_2025"] = n25
    out["sha256_all"] = hashlib.sha256("\n".join(cand26).encode()).hexdigest()
    json.dump(out, open("lexicon_2026.json", "w"), indent=1)
    print({k: len(v) for k, v in out.items() if isinstance(v, list)}, file=sys.stderr)
    top = sorted(rows, key=lambda r: -(r[1] - r[2]))[:30]
    print("largest 2026 excess:", ", ".join(w for w, _, _ in top), file=sys.stderr)
    print("new (sample):", ", ".join(out["new"][:60]), file=sys.stderr)


if __name__ == "__main__":
    main()
