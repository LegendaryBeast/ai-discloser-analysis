# Pilot: can disclosing authors be traced back to 2015 and 2020?

Run on 2026-10-27. Sources: Europe PMC (papers and full text), OpenAlex (author histories, free tier, no API key).

## What was done

1. **Sampling** (`step1_collect.py`). Searched Europe PMC for open-access research articles from 2025–26 with full text, using three disclosure phrasings:

   | Phrasing | Europe PMC hits |
   |---|---|
   | Elsevier template ("During the preparation of this work the author…") | 3,457 |
   | ChatGPT-for-language wording | 1,113 |
   | Generic "generative AI … language editing / readability" | 8,409 |

   I took a seeded random draw (seed 2026) from the 300 most recent hits of each phrasing, then fetched each paper's full text and kept it only if it contained a sentence disclosing AI use for writing.
2. **Exclusions**: 5 of the 55 papers. Two used AI only for graphics or a graphical abstract (#0, #24), one had only the empty template heading (#19), and two matched the search but weren't disclosures (#43, #51). **That left 50 papers, all published in 2026.**
3. **Author lookup** (`step2_openalex.py`). All 50 papers were found in OpenAlex. For the first, corresponding and last author of each, I counted articles and reviews published in **2014–16** and **2019–21**, both in any author position and as first author.
4. **Summary** (`step3_summarize.py`) writes `summary.txt` and `pilot_results.csv` (one row per paper and role, including the disclosure sentence).

## Results

| Role | n | Any paper 2014–16 | Any paper 2019–21 | **Both windows** | Both windows *as first author* |
|---|---|---|---|---|---|
| First author | 47 | 26% | 55% | **23%** | **15%** |
| Corresponding author* | 34 | 50% | 76% | **50%** | 21% |
| Last (senior) author | 49 | 73% | 76% | **71%** | 35% |

\*OpenAlex flags a corresponding author for only 34 of the 50 papers (68%).

At the paper level:

- For 44% of papers, the first or the corresponding author has publications in both windows.
- For 22% of papers, one of those two authors has *first-authored* publications in both windows.

ORCID coverage is 79–84% across the three roles. First authors come from 22 countries, led by China (10) and the US (5).

## What it means for the full study

- **First authors are a poor anchor.** Most first authors were students or early-career researchers who weren't publishing in 2015, so only about 15–23% have papers in both windows.
- **Senior or corresponding authors are a good anchor** for retention (50–71%). The catch is that they often didn't write the earlier papers themselves: only 21–35% were first author on papers in both windows. A senior author's 2015 paper may have been written by a student.
- **Expected yield.** If you anchor on the author who wrote the text (first author in every wave), expect about 15–22% of disclosing papers to qualify. To get 300 usable author trajectories you'd need about **1,500–2,000 disclosing papers**. That's achievable: these three phrasings alone return about 13,000 Europe PMC hits.
- **Two design options:**
  - **(A) Strict:** the same person is first author in all three waves. This gives the cleanest style signal, but about 15–20% yield and a sample biased toward mid-career researchers.
  - **(B) Pragmatic:** follow the corresponding or senior author, with author position as a covariate, plus a robustness check on the strict subset. This gives about 50–70% yield, but the signal is noisier.

## Caveats

- **Name disambiguation.** OpenAlex author profiles can wrongly merge different people, especially for common names (several Chinese names in this sample). Eleven profiles have more than 60 papers in one window; most look like real prolific senior researchers, but I didn't verify them by hand. The full study should require an ORCID and spot-check profiles.
- **Recency bias.** All 50 papers are from 2026, because the draw came from the most recent hits. For the main study, sample across 2024–26.
- **Field.** The sample is almost entirely biomedical, because it comes from Europe PMC. Results may differ for computer science or social sciences.
- **Small sample.** The percentages are rough, about ±12 percentage points at n≈50.
- **Mixed tools.** Some disclosures name tools that aren't LLMs or are only partly LLM-based (Grammarly, PaperPal, DeepL). Code tool type before analysis.
- **Full text.** I didn't check whether the 2014–16 and 2019–21 papers are available in full text. Abstracts are always available; full text depends on open access.

---

# Follow-up: can we get full text for the earlier papers, or only abstracts?

Scripts: `step4_fulltext.py` → `fulltext_status.json`; `step5_fulltext_summary.py` → `fulltext_summary.txt`.

**Who was checked:** the 49 unique authors from the pilot who published in both 2014–16 and 2019–21, in any role. Of these, 25 were first author in both periods. Their 1,963 articles and reviews from those two periods were checked.

**How each paper was classified:**
- `pmc_oa`: in the PMC open-access subset, so the full text is available as XML that can be mined directly. Checked against Europe PMC.
- `preprint`: has a copy on a preprint server (arXiv, bioRxiv, medRxiv and similar).
- `pmc_other`: in PMC but not in the open-access subset.
- `other_oa`: free to read elsewhere, usually as a publisher or repository PDF.
- `abstract_only` / `nothing`: closed, with or without an abstract in OpenAlex.

## Paper level

| Period | n | PMC OA | Preprint | PMC (not OA) | Other OA | Abstract only | Nothing in OpenAlex |
|---|---|---|---|---|---|---|---|
| 2014–16, all | 826 | 19% | 0% | 10% | 27% | 21% | 23% |
| 2019–21, all | 1,137 | 33% | 1% | 9% | 29% | 14% | 14% |
| 2014–16, first-authored | 137 | 18% | 0% | 7% | 33% | 20% | 23% |
| 2019–21, first-authored | 164 | 25% | 0% | 10% | 42% | 11% | 12% |

## Author level (n = 49)

| Usable text in **both** periods | Any author position | As first author |
|---|---|---|
| Machine-readable full text (PMC OA or preprint) | **55%** (median 2 papers in the thinner period) | **16%** (median 1) |
| Any free full text, including PDFs | 90% (median 4) | 37% (median 1.5) |
| Abstract | 96% (median 5) | 47% (median 2) |

## What this means

1. **Abstracts are the only option that works at scale for the three-period design.** Nearly every retained author has abstracts in both periods. Abstracts are also where earlier studies found the strongest LLM signal (Kobak, Liang), so your results will be directly comparable to theirs.
2. **Use full text as a secondary analysis on a subset.** About half of the retained authors have machine-readable full text in both periods, but typically only 2 papers in the thinner period. Restricted to papers the author wrote as first author, that drops to 16%. That's enough for a robustness check, not for the main analysis.
3. **Free PDFs (`other_oa`) add a lot of coverage** (90% of authors). Using them means downloading and parsing PDFs (for example with GROBID) and checking each licence. This is optional extra work.
4. **The 2014–16 period is the bottleneck.** Open access was rarer then, and the PMC OA share roughly doubles between the two periods for all papers. Expect the fewest usable texts in that period.
5. **Abstract coverage here is a lower bound.** OpenAlex has dropped abstracts for some publishers. Of the 345 papers with no abstract in OpenAlex, 182 have a PMID, so their abstracts can be fetched from PubMed.

**Suggested text plan:**
- Main analysis on **abstracts**, taken from PubMed first and OpenAlex as a fallback.
- Robustness check on **full text** from the PMC OA subset and preprints. Analyse the introduction and discussion only; methods sections are formulaic.
- Report results by author position (first-authored papers vs any position).
