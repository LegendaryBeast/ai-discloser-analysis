#!/bin/bash
# Revision pipeline: from collected data to the compiled manuscript. Data collection (e1-e4, e8, e9, m5) is separate
# because it needs network access and API budget; see ../paper/README.md.
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "$DIR/../marker_list"
echo "[$(date)] M6: 2026 refresh lexicon"
python3 m6_lexicon2026.py

cd "$DIR"
for s in "" 2026; do  # rebuild a replication dataset from its saved state only if it is missing and the state still
    # holds abstracts (in the public release the datasets come from ../rehydrate.py instead)
    if [ ! -f "replication${s}_dataset.csv" ] && [ -f "replication${s}_state.json" ] && grep -q '"abstract"' "replication${s}_state.json"; then
        python3 -c "import json, sys; sys.argv=['e9', '${s:-2024-25}']; import e9_replication as e; e.build_dataset(json.load(open(e.STATE))['pairs'])"
    fi
done
echo "[$(date)] E10: all estimates and tables"
python3 e10_revision.py > e10.log 2>&1
echo "[$(date)] E11: decay curve, temporal transfer, sensitivity"
python3 e11_impact.py > e11.log 2>&1

cd "$DIR/../paper"
echo "[$(date)] Figures (and, where the manuscript is present, numbers.tex and the PDF)"
if [ -f make_numbers.py ] && [ -f main.tex ]; then python3 make_numbers.py; fi   # manuscript only (not released)
python3 make_figures.py
if [ -f main.tex ]; then latexmk -pdf -interaction=nonstopmode main.tex > /dev/null; fi
echo "[$(date)] Done"
