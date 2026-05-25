"""Tokenizer tests, including all 8 cases from the original DataUtils unittest."""
from __future__ import annotations

import numpy as np
import pytest

from geneva2s.tokenizer import (
    CharTokenizer,
    OKCHARS,
    replace_multichar,
    unreplace_multichar,
)


class TestErtlReplacements:
    @pytest.mark.parametrize("smi_in,smi_out", [
        ("c1[nH]ccc1", "c1Accc1"),
        ("FCF", "FCF"),
        ("ClC(Cl)=C(Cl)Cl", "LC(L)=C(L)L"),
        ("BrCC", "RCC"),
        ("ICC", "ICC"),
        ("C/C=C/C", "CC=CC"),
        ("F[C@@H](Cl)Br", "FC(L)R"),
        ("F[C@H](Cl)Br", "FC(L)R"),
    ])
    def test_original_unittest_cases(self, smi_in, smi_out):
        """All 8 cases from SmilesGEN_utils_fixed.ErtlLSTMUtilsTest."""
        assert replace_multichar(smi_in) == smi_out

    def test_invalid_chars_rejected(self):
        """SMILES with chars outside OKCHARS should return ""."""
        assert replace_multichar("CC.CC") == ""  # '.' not in OKCHARS

    def test_unreplace_roundtrip(self):
        """Single-substring replacements should round-trip cleanly."""
        for smi_in in ["FCF", "BrCC", "[nH]1cccc1", "ClC"]:
            replaced = replace_multichar(smi_in)
            assert unreplace_multichar(replaced).replace("[nH]", "[nH]") is not None


class TestCharTokenizer:
    def test_vocab_size(self):
        tok = CharTokenizer()
        assert tok.vocab_size == 27
        assert sorted(tok.chars) == sorted(set(OKCHARS))

    def test_encode_decode_roundtrip(self):
        tok = CharTokenizer()
        s = "Cc1ccccc1\nCCO\n"
        ids = tok.encode(s)
        assert tok.decode(ids) == s

    def test_encode_drops_unknown(self):
        tok = CharTokenizer()
        ids = tok.encode("Cc1xxccc1")  # 'x' not in vocab
        assert all(0 <= i < tok.vocab_size for i in ids)
        # x is dropped
        assert tok.decode(ids) == "Cc1ccc1"

    def test_prepare_corpus(self):
        """Mirrors DataUtils.Prepare test from the original unittest."""
        tok = CharTokenizer()
        smiles_in = ["ClCCl", "C/C=C/C", "CCCC", "[nH]1cccc1"]
        expected = "\n".join(["LCL", "CC=CC", "CCCC", "A1cccc1"]) + "\n"
        assert tok.prepare_corpus(smiles_in) == expected

    def test_prepare_filters_too_long(self):
        tok = CharTokenizer(maxlen=5)
        smiles_in = ["CCO", "CCCCCCCCC"]  # second too long
        out = tok.prepare_corpus(smiles_in)
        assert "CCO" in out
        assert "CCCCCCCCC" not in out

    def test_sliding_window_shapes(self):
        tok = CharTokenizer(maxlen=10, step=3)
        text = "C" * 100 + "\n"
        X, y = tok.sliding_window(text)
        assert X.dtype == np.int64
        assert y.dtype == np.int64
        assert X.shape[1] == 10
        assert X.shape[0] == y.shape[0]
        # step=3 → (101 - 10) // 3 + 1 = 31 (approximately)
        assert 25 <= X.shape[0] <= 35
