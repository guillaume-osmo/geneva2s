"""Tests for geneva2s.adaptive — both InChIKey-prefix and Morgan/Tanimoto explorers."""
from __future__ import annotations

import numpy as np
import pytest

from geneva2s.adaptive import (
    AdaptiveSmilesExplorer,
    adaptive_temperature,
    compute_morgan_batch,
)


# Druglike SMILES — distinct scaffolds → different Morgan clusters
_DRUGLIKE = [
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",        # ibuprofen
    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",      # naproxen
    "CC(=O)Nc1ccc(O)cc1",                # paracetamol
    "CC(=O)Oc1ccccc1C(=O)O",             # aspirin
    "CN1CCCC1c1cccnc1",                  # nicotine (achiral)
    "CC(C)NCC(O)c1ccc(O)c(CO)c1",        # salbutamol
]


# ============================================================================
# adaptive_temperature
# ============================================================================

class TestAdaptiveTemperature:
    def test_5_cycle(self):
        expected = [1.5, 1.2, 1.0, 0.8, 0.6]
        for i in range(5):
            assert adaptive_temperature(i) == expected[i]

    def test_cycles_repeat(self):
        for cyc in range(3):
            for i in range(5):
                assert adaptive_temperature(cyc * 5 + i) == [1.5, 1.2, 1.0, 0.8, 0.6][i]

    def test_ignores_extra_args(self):
        # Should accept the (round, freq_counter, cluster_counter) signature
        # used by the explorers, without using them.
        assert adaptive_temperature(0, {"foo": 1}, {"bar": 2}) == 1.5


# ============================================================================
# AdaptiveSmilesExplorer (InChIKey-prefix, CPU)
# ============================================================================

class TestAdaptiveExplorer:
    def test_accepts_first_occurrence(self):
        def gen(n, t): return ["CCO", "c1ccccc1", "CC(=O)O"]  # all valid, distinct
        e = AdaptiveSmilesExplorer(generator_func=gen, temperature_func=adaptive_temperature)
        e.run_round(n_samples=3, verbose=False)
        assert len(e.get_dataset()) == 3

    def test_rejects_duplicates(self):
        def gen(n, t): return ["CCO"] * 5  # all the same SMILES
        e = AdaptiveSmilesExplorer(generator_func=gen, max_freq=2)
        e.run_round(n_samples=5, verbose=False)
        # First occurrence accepted, rest rejected (dup or freq cap)
        assert len(e.get_dataset()) == 1

    def test_logs_accepted_flag(self):
        def gen(n, t): return ["CCO"] * 5
        e = AdaptiveSmilesExplorer(generator_func=gen, max_freq=2)
        e.run_round(n_samples=5, verbose=False)
        entries = e.generated_data[0]
        assert len(entries) == 5
        # First accepted, others rejected
        assert entries[0]["accepted"] is True
        assert all(entry["accepted"] is False for entry in entries[1:])

    def test_cluster_cap(self):
        # All molecules with the same InChIKey-3-prefix should hit the cap
        # CCO repeated 5 times: max_freq=10 (so freq doesn't reject), cluster cap=2
        def gen(n, t): return ["CCO"] * 5
        e = AdaptiveSmilesExplorer(generator_func=gen, max_freq=10, max_cluster=2)
        e.run_round(n_samples=5, verbose=False)
        # Only one accepted (the dataset check fires after first acceptance)
        assert len(e.get_dataset()) == 1


# ============================================================================
# Morgan fingerprint utilities (mlxmolkit-backed; skipped if not installed)
# ============================================================================

try:
    import mlxmolkit  # noqa: F401
    _HAS_MLXMOLKIT = True
except ImportError:
    _HAS_MLXMOLKIT = False


@pytest.mark.skipif(not _HAS_MLXMOLKIT, reason="mlxmolkit not installed")
class TestMorganBatch:
    def test_invalid_smiles_dropped(self):
        smis = ["CCO", "not_a_smiles_xx", "c1ccccc1"]
        fp_u32, idx_map = compute_morgan_batch(smis, radius=2, nbits=2048)
        assert idx_map == [0, 2]
        # nbits=2048 → 64 uint32 words
        assert fp_u32.shape == (2, 64)

    def test_empty_returns_none(self):
        fp_u32, idx_map = compute_morgan_batch(["not_valid"])
        assert fp_u32 is None
        assert idx_map == []

    def test_druglike_yields_distinct_fps(self):
        fp_u32, idx_map = compute_morgan_batch(_DRUGLIKE, radius=2, nbits=2048)
        assert idx_map == list(range(len(_DRUGLIKE)))
        # Sanity: distinct druglike fps should not be byte-identical
        arr = np.array(fp_u32)
        assert len({tuple(row) for row in arr}) == len(_DRUGLIKE)


# ============================================================================
# MorganAdaptiveExplorer — only run if mlxmolkit available
# ============================================================================

@pytest.mark.skipif(not _HAS_MLXMOLKIT, reason="mlxmolkit not installed")
class TestMorganAdaptiveExplorer:
    def test_accepts_distinct_druglike(self):
        from geneva2s.adaptive import MorganAdaptiveExplorer

        def gen(n, t):
            return _DRUGLIKE[:n] if n <= len(_DRUGLIKE) else _DRUGLIKE

        # Loose cluster threshold so each distinct scaffold opens its own cluster;
        # high novelty threshold so the Tanimoto filter doesn't reject any.
        e = MorganAdaptiveExplorer(
            generator_func=gen,
            cluster_threshold=0.9,
            max_per_cluster=10,
            max_freq=10,
            novelty_threshold=0.99,
        )
        e.run_round(n_samples=len(_DRUGLIKE), verbose=False)
        # Most/all distinct druglike SMILES should be accepted
        assert len(e.get_dataset()) >= len(_DRUGLIKE) - 2

    def test_rejects_repeats(self):
        from geneva2s.adaptive import MorganAdaptiveExplorer

        def gen(n, t): return [_DRUGLIKE[0]] * n

        e = MorganAdaptiveExplorer(generator_func=gen, max_freq=2)
        e.run_round(n_samples=5, verbose=False)
        # Only the first occurrence accepted (subsequent rejected by dup/freq cap)
        assert len(e.get_dataset()) == 1

    def test_morgan_and_erg_are_distinct_classes(self):
        # ErgAdaptiveExplorer is now a real ERG-based class (not an alias).
        from geneva2s.adaptive import ErgAdaptiveExplorer, MorganAdaptiveExplorer
        assert ErgAdaptiveExplorer is not MorganAdaptiveExplorer


# ============================================================================
# ERG fingerprint utilities + ErgAdaptiveExplorer
# ============================================================================

@pytest.mark.skipif(not _HAS_MLXMOLKIT, reason="mlxmolkit not installed")
class TestComputeErgBatch:
    def test_invalid_dropped(self):
        from geneva2s.adaptive import compute_erg_batch

        fp, idx = compute_erg_batch(["CCO", "not_a_smiles_xx", _DRUGLIKE[0]])
        assert fp is not None
        assert fp.shape == (2, 315)
        assert idx == [0, 2]

    def test_empty_input_returns_none(self):
        from geneva2s.adaptive import compute_erg_batch

        fp, idx = compute_erg_batch([])
        assert fp is None
        assert idx == []

    def test_all_invalid_returns_none(self):
        from geneva2s.adaptive import compute_erg_batch

        fp, idx = compute_erg_batch(["not_a", "still_bad"])
        assert fp is None
        assert idx == []


@pytest.mark.skipif(not _HAS_MLXMOLKIT, reason="mlxmolkit not installed")
class TestErgAdaptiveExplorer:
    def test_accepts_distinct_druglike(self):
        from geneva2s.adaptive import ErgAdaptiveExplorer

        def gen(n, t):
            return _DRUGLIKE[:n] if n <= len(_DRUGLIKE) else _DRUGLIKE

        # Tight cluster threshold so each distinct pharmacophore opens its own
        # cluster; very high novelty threshold so the cosine filter doesn't fire.
        e = ErgAdaptiveExplorer(
            generator_func=gen,
            cluster_threshold=0.95,
            max_per_cluster=10,
            max_freq=10,
            novelty_threshold=0.9999,
        )
        e.run_round(n_samples=len(_DRUGLIKE), verbose=False)
        # Most/all distinct druglike SMILES should be accepted.
        assert len(e.get_dataset()) >= len(_DRUGLIKE) - 2

    def test_rejects_repeats(self):
        from geneva2s.adaptive import ErgAdaptiveExplorer

        def gen(n, t): return [_DRUGLIKE[0]] * n

        e = ErgAdaptiveExplorer(generator_func=gen, max_freq=2)
        e.run_round(n_samples=5, verbose=False)
        # Only the first occurrence accepted (subsequent rejected by dup/freq cap)
        assert len(e.get_dataset()) == 1

    def test_run_adaptive_erg_mode_dispatches(self):
        from geneva2s.adaptive import run_adaptive, ErgAdaptiveExplorer

        def gen(n, t):
            return _DRUGLIKE[:n] if n <= len(_DRUGLIKE) else _DRUGLIKE

        explorer = run_adaptive(
            generator_func=gen,
            n_rounds=1,
            n_samples_per_round=len(_DRUGLIKE),
            mode="erg",
            verbose=False,
            max_freq=10,
            novelty_threshold=0.9999,
            cluster_threshold=0.95,
        )
        assert isinstance(explorer, ErgAdaptiveExplorer)
