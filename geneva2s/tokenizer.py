"""Character-level SMILES tokenizer.

Reduced 27-char vocab. Stereochemistry ([C@H], [C@@H], [C@@], [C@]) and
cis/trans (/, \\) are erased — the model only learns connectivity + bonds
+ rings. Output SMILES are stereo-unspecified.

Tokenization scheme based on:
  van Deursen R, Ertl P, Tetko IV, Godin G (2020).
  GEN: highly efficient SMILES explorer using autodidactic generative
  examination networks. J Cheminform 12:22.
"""
from __future__ import annotations

import numpy as np

ERTL_REPLACEMENTS = {
    "[nH]": "A",
    "Cl": "L",
    "Br": "R",
    "/": "",
    "\\": "",
    "[C@@H]": "C",
    "[C@H]": "C",
    "[C@@]": "C",
    "[C@]": "C",
}

ERTL_UNREPLACE = [
    ("L", "Cl"),
    ("R", "Br"),
    ("A", "[nH]"),
]

OKCHARS = "CFLRIONSAcons123456789=#()\n"


def replace_multichar(smi: str) -> str:
    """Apply Ertl replacements; return "" if any leftover char is outside OKCHARS."""
    smi = smi.strip().split("\t")[0]
    for k, v in ERTL_REPLACEMENTS.items():
        if k in smi:
            smi = smi.replace(k, v)
    return smi if all(c in OKCHARS for c in smi) else ""


def unreplace_multichar(smi: str) -> str:
    for k, v in ERTL_UNREPLACE:
        smi = smi.replace(k, v)
    return smi


class CharTokenizer:
    def __init__(self, maxlen: int = 42, step: int = 3):
        self.maxlen = maxlen
        self.step = step
        self.chars = sorted(OKCHARS)
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for c, i in self.c2i.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, s: str) -> list[int]:
        return [self.c2i[c] for c in s if c in self.c2i]

    def decode(self, ids) -> str:
        return "".join(self.i2c[int(i)] for i in ids if int(i) in self.i2c)

    def prepare_corpus(self, smiles_list) -> str:
        replaced = [replace_multichar(s) for s in smiles_list]
        kept = [r for r in replaced if 0 < len(r) <= self.maxlen]
        return "\n".join(kept) + "\n" if kept else ""

    def sliding_window(self, text: str):
        ids = self.encode(text)
        x, y = [], []
        for i in range(0, len(ids) - self.maxlen, self.step):
            x.append(ids[i : i + self.maxlen])
            y.append(ids[i + self.maxlen])
        return np.array(x, dtype=np.int64), np.array(y, dtype=np.int64)
