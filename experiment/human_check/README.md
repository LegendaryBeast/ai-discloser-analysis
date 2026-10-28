# Human check (needed before submission)

Two sheets, to be filled in by a person who has **not** seen the original decisions. Do not open the `_key_` files.

1. `disclosure_sheet.csv` (150 sentences). Rule: Keep a sentence only if the authors state that they used an AI tool on the text of THIS manuscript (language, grammar, translation, readability, drafting or structuring). Exclude non-disclosures and uses that do not touch the text (figures, code, literature search only).
   Write `keep` or `exclude` in the third column.
2. `lexicon_sheet.csv` (347 words). A word is **style** if its meaning is generic and could appear in an abstract
   from any field; it is **content** if it names a disease, organism, molecule, material, technique, measurement,
   statistical tool, place, topic or AI technology. Write `style` or `content`.

Then run `python3 human_check.py score` in `experiment/`, and rebuild the paper (`paper/make_numbers.py`, LaTeX):
the agreement statistics then appear in the manuscript automatically.
