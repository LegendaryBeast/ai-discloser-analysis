"""Independent human check of the two judgement calls in the study: (1) the disclosure audit and (2) the style/content
classification of the 347 lexicon candidates.

  python3 human_check.py make    # writes blinded coding sheets to human_check/ (run once)
  python3 human_check.py score   # after a person has filled in the sheets: agreement and Cohen's kappa
                                 # -> human_check/human_check_results.json (read by paper/make_numbers.py)

The disclosure sheet holds 150 sentences: every sentence excluded or marked ambiguous in the audits (so that the
check has power on exclusions) plus a seeded random sample of kept sentences, shuffled. The coder sees only the
sentence and the rule, never the original decision, which is stored separately in human_check/_key_*.csv."""
import csv, json, os, random, sys

OUT = "human_check"
RULE = ("Keep a sentence only if the authors state that they used an AI tool on the text of THIS manuscript "
        "(language, grammar, translation, readability, drafting or structuring). Exclude non-disclosures and uses "
        "that do not touch the text (figures, code, literature search only).")


def sentences():
    """Every audited disclosure sentence with the audit decision: main-sample complete pairs and both replications."""
    import pandas as pd
    a = json.load(open("audit_v2.json")); r = json.load(open("replication_audit.json"))
    D = json.load(open("disclosers.json"))["papers"]
    P = pd.read_csv("revision_pairs.csv", index_col=0)
    rows = []
    for k, i in P["paper_index"].items():
        dec = "exclude" if str(i) in a["excluded"] else "ambiguous" if str(i) in a["ambiguous"] else "keep"
        rows.append(("main", f"paper{i}", D[i]["disclosure"], dec))
    for suf in ("", "2026"):
        for p in json.load(open(f"replication{suf}_units.json")):
            dec = "exclude" if p["disc"]["pmcid"] in r["excluded"] else "keep"
            rows.append((f"replication{suf or '2024-25'}", p["disc"]["pmcid"], p["disc"]["disclosure"], dec))
    return rows


def make():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(2033)
    rows = sentences()
    flagged = [r for r in rows if r[3] != "keep"]
    kept = [r for r in rows if r[3] == "keep"]
    sample = flagged + rng.sample(kept, 150 - len(flagged))
    rng.shuffle(sample)
    with open(f"{OUT}/disclosure_sheet.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item", "sentence", "your_decision (keep/exclude)", "note"])
        for n, r in enumerate(sample, 1):
            w.writerow([n, r[2], "", ""])
    with open(f"{OUT}/_key_disclosure.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item", "source", "id", "audit_decision"])
        for n, r in enumerate(sample, 1):
            w.writerow([n, r[0], r[1], r[3]])
    words = list(csv.DictReader(open("../marker_list/classification.csv")))
    rng.shuffle(words)
    with open(f"{OUT}/lexicon_sheet.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["word", "your_label (style/content)"])
        for r in words:
            w.writerow([r["word"], ""])
    open(f"{OUT}/README.md", "w").write(f"""# Human check (needed before submission)

Two sheets, to be filled in by a person who has **not** seen the original decisions. Do not open the `_key_` files.

1. `disclosure_sheet.csv` (150 sentences). Rule: {RULE}
   Write `keep` or `exclude` in the third column.
2. `lexicon_sheet.csv` (347 words). A word is **style** if its meaning is generic and could appear in an abstract
   from any field; it is **content** if it names a disease, organism, molecule, material, technique, measurement,
   statistical tool, place, topic or AI technology. Write `style` or `content`.

Then run `python3 human_check.py score` in `experiment/`, and rebuild the paper (`paper/make_numbers.py`, LaTeX):
the agreement statistics then appear in the manuscript automatically.
""")
    print(f"wrote {OUT}/disclosure_sheet.csv ({len(sample)} items, {len(flagged)} flagged) and lexicon_sheet.csv ({len(words)} words)")


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def score():
    key = {r["item"]: r for r in csv.DictReader(open(f"{OUT}/_key_disclosure.csv"))}
    sheet = [r for r in csv.DictReader(open(f"{OUT}/disclosure_sheet.csv")) if r["your_decision (keep/exclude)"].strip()]
    h = [r["your_decision (keep/exclude)"].strip().lower() for r in sheet]
    ai = ["exclude" if key[r["item"]]["audit_decision"] == "exclude" else "keep" for r in sheet]  # ambiguous counts as keep
    res = {}
    if sheet:
        po, k = kappa(ai, h)
        res["disclosure"] = {"n": len(sheet), "agreement": po, "kappa": k,
                             "disagreements": [{"item": r["item"], "audit": a_, "human": h_}
                                               for r, a_, h_ in zip(sheet, ai, h) if a_ != h_]}
    lex = {r["word"]: r["decision"] for r in csv.DictReader(open("../marker_list/classification.csv"))}
    ls = [r for r in csv.DictReader(open(f"{OUT}/lexicon_sheet.csv")) if r["your_label (style/content)"].strip()]
    if ls:
        lpo, lk = kappa([lex[r["word"]] for r in ls], [r["your_label (style/content)"].strip().lower() for r in ls])
        pairs = [(lex[r["word"]], r["your_label (style/content)"].strip().lower(), r["word"]) for r in ls]
        res["lexicon"] = {"n": len(ls), "agreement": lpo, "kappa": lk,
                          "style_to_content": [w for a, b, w in pairs if a == "style" and b == "content"],
                          "content_to_style": [w for a, b, w in pairs if a == "content" and b == "style"],
                          "human_style_words": sorted(w for a, b, w in pairs if b == "style")}
    json.dump(res, open(f"{OUT}/human_check_results.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"make": make, "score": score}[sys.argv[1]]()
