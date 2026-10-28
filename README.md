# Replication data and code: AI-use disclosure and within-author writing style, 2014–2026

Data and scripts for a matched three-wave within-author study of authors of biomedical papers that disclose AI
writing assistance, compared with matched non-disclosing authors. The manuscript itself is not part of this package.

## Contents
| Folder | What it holds |
|---|---|
| `pilot/` | Feasibility study: scripts and results, author names removed |
| `marker_list/` | Lexicon derivation and validation: frozen lexicon `marker_list_v1.json` (SHA-256 `84984ef3…8ad2b9e4`), candidates, coding rule, PMID manifests of the 218,895 lexicon abstracts and the 50,274 2026 benchmark abstracts, 2026 refresh lexicon |
| `experiment/` | Data collection (`e1`–`e4`, `e8`, `e9`), audits (`audit_v2.json`, `replication_audit.json`), the replication plan written before collection (`replication_plan.md`), the analysis (`e10_revision.py`), a manifest of all 51,268 analysed texts (`texts_manifest.csv.gz`) and every numerical output (`revision_results.json`) |
| `paper/` | Scripts only: `make_figures.py` draws all figures and `e10_revision.py` writes all tables from the results |
| `rehydrate.py` | Re-fetches the abstracts that cannot be redistributed and verifies each against its SHA-256 |

## Reproduce
1. **Inspect (no software needed).** `experiment/revision_results.json` holds every estimate.
2. **Recompute everything from identifiers** (Python 3.13, about 1–2 hours, internet access):
   ```
   pip install -r requirements.txt
   python3 rehydrate.py               # rebuilds the abstract datasets and both PubMed corpora; writes rehydration_report.json
   bash experiment/run_revision.sh    # refresh lexicon, all estimates and tables (paper/tables/), all figures (paper/figures/)
   ```
   `rehydration_report.json` lists how many texts match the recorded hash. A few may differ if OpenAlex or PubMed
   corrected an abstract since October 2026; the analysis then uses the current text.
3. **Re-collect from scratch** with `experiment/e1_disclosers.py` … `e4_dataset.py`, `e8_verify.py` and
   `e9_replication.py` (see each script's docstring). Results will differ slightly because the databases change.

An OpenAlex API key (free) raises the daily request limit: put `OPENALEX_API_KEY=...` in `experiment/.env`.

## What is not included, and why
- **The manuscript**, which is unpublished.
- **Abstract and full texts**, which are under publisher copyright. Every text is identified (OpenAlex work ID or
  PMID) and hashed, and `rehydrate.py` restores them.
- **Author names and ORCIDs**. OpenAlex author and work IDs are kept because the matching depends on them. The
  pilot-author exclusion in the lexicon derivation uses `marker_list/pilot_author_pmids.txt` instead of names, with
  identical results.
- **API caches, logs and keys.**

## Licences
Code: MIT (`LICENSE`). Data created for this study: CC BY 4.0. Quoted disclosure sentences remain under their articles'
licences (`LICENSE-DATA.md`). Verify file integrity with `shasum -a 256 -c CHECKSUMS.sha256`.
