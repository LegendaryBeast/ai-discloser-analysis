"""Text metrics shared by the analysis: marker-word index, sentence length and its SD, MTLD, Flesch reading ease and
the function-word profile used for self-drift."""
import json, re
import numpy as np

TOKEN = re.compile(r"[a-z]+(?:-[a-z]+)*")
LEX = json.load(open("../marker_list/marker_list_v1.json"))
MARKERS = set(LEX["words"])
FUNCTION = ("the of and to a in for is on that by this with as are be was at from or an which were it not have has "
            "been these their its than but also both between more such into may can these those however while "
            "although whereas thus therefore our we there other each most only all any no after before during "
            "within without among through under over about".split())
FW_INDEX = {w: i for i, w in enumerate(dict.fromkeys(FUNCTION))}


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if len(s.split()) >= 3]


def syllables(word):
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups) - (1 if word.endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def mtld_pass(tokens, ttr=0.72):
    factors, types, count = 0.0, set(), 0
    for t in tokens:
        types.add(t); count += 1
        if len(types) / count <= ttr:
            factors += 1; types, count = set(), 0
    if count:
        factors += (1 - len(types) / count) / (1 - ttr)
    return len(tokens) / factors if factors else float(len(tokens))


def metrics(text):
    toks = TOKEN.findall(text.lower())
    sl = [len(s.split()) for s in sentences(text)] or [len(text.split())]
    words = re.findall(r"[A-Za-z]+", text)
    fw = np.zeros(len(FW_INDEX))
    for t in toks:
        if t in FW_INDEX:
            fw[FW_INDEX[t]] += 1
    return {"n_tokens": len(toks),
            "mwi": 1000 * sum(t in MARKERS for t in toks) / len(toks),
            "sent_len": float(np.mean(sl)), "sent_sd": float(np.std(sl)),
            "mtld": (mtld_pass(toks) + mtld_pass(toks[::-1])) / 2,
            "flesch": 206.835 - 1.015 * len(words) / len(sl) - 84.6 * sum(map(syllables, words)) / max(1, len(words)),
            "fw": fw / max(1, len(toks))}


