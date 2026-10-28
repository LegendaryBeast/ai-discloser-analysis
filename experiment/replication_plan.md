# Replication sample: plan fixed before data collection (2026-10-28)

## Why
In the main sample the discloser excess on the marker-word index (MWI) was large for index papers from 2024-25
(beta3 = +1.52 SD, 96 pairs) and absent for 2026 (beta3 = -0.01 SD, 389 pairs). That contrast was found after
looking at the data, and the 2024-25 pairs came almost entirely from one query (ChatGPT-specific phrasing) because the
E1 candidate pool was capped at 2,500 hits per query, newest first. This sample tests the contrast on new pairs drawn
from the complete 2024-25 sampling frame.

## Sampling frame
All open-access PMC records published in 2024 or 2025 matching any of the three E1 disclosure phrasings (no cap), minus
every DOI already in `disclosers.json`. Seeded random order (seed 2029). Disclosure verified in the Europe PMC full text
with the E1 instrument plus two fixes found in the audit: sentences are split only at a full stop followed by a capital
letter (so "GPT-4.1" no longer truncates the sentence), and the purpose terms must appear outside the Elsevier section
heading (so figure-only, code-only and bibliography-only uses are rejected).

## Pairs
Anchor author and eligibility as in E2. Controls as in E3 (same journal, +-120 days, then +-365 days), with three
fixes: no OpenAlex fallback (every control is screened in Europe PMC full text), every author already used as a
discloser or control in the main sample is excluded, and a control whose full text contains an AI-use disclosure is
rejected. Collection stops when the OpenAlex daily budget is spent.

## Hypotheses and analysis (identical model to the main sample, Eq. 1, author-clustered SEs)
- H1: in the replication sample, beta3 on the MWI (z, W2-control scale of the main sample) is > 0. One-sided test,
  alpha = 0.05.
- H2: beta3 (replication, 2024-25) exceeds beta3 (main sample, 2026). Tested as the cohort interaction in a pooled model
  of the replication pairs and the main-sample 2026 pairs. One-sided, alpha = 0.05.
- Placebo beta1 is reported with a TOST equivalence test at +-0.10 SD.
No other outcome is treated as confirmatory.

## Amendment 1 (2026-10-28, written before any 2026 replication data were collected)
H2 as stated reuses the main-sample 2026 pairs, whose frame was capped and newest-first. To make the cohort contrast
independent of the exploratory sample, a second replication sample is drawn from the complete 2026 frame (every
open-access PMC paper from January to October 2026 matching any of the three phrasings, harvested month by month so
that no cap applies), with the same verification, matching and exclusion rules (seed 2031). No author from the main
sample or the 2024-25 replication sample is reused.
- H3: beta3 (2024-25 replication) exceeds beta3 (2026 replication). Tested as the cohort interaction in a pooled model
  of the two replication samples. One-sided, alpha = 0.05.
- The 2026 replication beta3 is reported with its 95% CI and a TOST against +-0.25 SD (no direction is hypothesised).
H1 and H2 are reported as originally specified.

## Stopping rule (recorded 2026-10-28 15:24 local time, before any replication outcome was examined)
Collection is limited by time, not by results. The 2024-25 sample stops at 125 collected pairs. The 2026 sample stops
at 110 collected pairs or at 17:30 local time on 2026-10-28, whichever comes first. No replication estimate is computed
before both samples have stopped.

## Amendment 2 (recorded 2026-10-28 15:32 local time, before any estimate with a 2026 lexicon was computed)
Exploratory lexicon-refresh test. A coder-free lexicon of 2026 excess words relative to the 2021-22 extrapolation
(M2 rule) admitted 1,721 forms dominated by topical drift over the four-year horizon, so it is not used. Instead,
'rising_2026' = word forms whose document frequency rose from 2025 (M1 derivation half) to 2026 (M5, even PMIDs,
index papers excluded) by ratio >= 1.5 and gap >= 0.0005, in >= 30 documents, excluding all 347 version-1 candidates
and AI-topic words. Prediction if the v1 lexicon aged: beta3 on rising_2026 > 0 in the 2026 cohort (main and
replication). This test is exploratory and is reported as such.
