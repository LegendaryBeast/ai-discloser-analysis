"""Step 6: rule-based coding of the 50 disclosure statements (tool, purpose, intensity). Writes disclosure_codes.csv
and prints the tallies. Intensity: 3 = drafting/structuring, 2 = polishing or translation, 1 = grammar only."""
import csv, json, re
from collections import Counter

TOOLS = [("ChatGPT", r"chat ?gpt|openai|gpt-?4"), ("Claude", r"\bclaude\b"), ("Gemini", r"gemini"),
         ("DeepSeek", r"deepseek"), ("Kimi", r"\bkimi\b"), ("Copilot", r"copilot"), ("Grammarly", r"grammarly"),
         ("PaperPal", r"paperpal"), ("DeepL", r"deepl")]
PURPOSES = [("drafting/structuring", r"\bdrafting\b|structur|summari[sz]|paragraph|case generation|academic phrasing"),
            ("translation", r"translat"),
            ("polishing", r"readab|polish|clarity|fluen|language edit|editing|refine|improv|concise|wording|flow"),
            ("grammar", r"grammar|spelling|proofread|grammatical")]

R = json.load(open("author_histories.json"))
tools, purposes, intensity, rows = Counter(), Counter(), Counter(), []
for r in R:
    d = r["disclosure"].lower()
    t = [n for n, p in TOOLS if re.search(p, d)] or ["unspecified"]
    p = [n for n, pat in PURPOSES if re.search(pat, d)]
    lvl = 3 if "drafting/structuring" in p else 2 if {"polishing", "translation"} & set(p) else 1 if p else 0
    tools.update(t); purposes.update(p); intensity[lvl] += 1
    rows.append([r["pmcid"], ";".join(t), ";".join(p), lvl, r["disclosure"]])
with open("disclosure_codes.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pmcid", "tools", "purposes", "intensity", "disclosure"]); w.writerows(rows)
print("tools:", tools.most_common()); print("purposes:", purposes.most_common()); print("intensity:", sorted(intensity.items()))
