"""Tests for geneva2s.adaptive — both InChIKey-prefix and ERG explorers."""
from __future__ import annotations

import numpy as np
import pytest

from geneva2s.adaptive import (
    AdaptiveSmilesExplorer,
    adaptive_temperature,
    compute_erg,
    compute_erg_batch,
)


# Druglike SMILES for non-trivial ERG fingerprints (small mols give zero vectors)
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
# ERG fingerprint
# ============================================================================

class TestERGFingerprint:
    def test_invalid_smiles_returns_none(self):
        assert compute_erg("not_a_smiles_xx") is None

    def test_druglike_nonzero_vector(self):
        fp = compute_erg(_DRUGLIKE[0])
        assert fp is not None
        assert fp.shape == (315,)
        # Druglike molecules with multiple pharmacophore types should produce
        # a non-zero ERG (small molecules like CCO get zero vectors)
        assert np.count_nonzero(fp) > 0

    def test_batch_filters_invalid(self):
        smis = ["CCO", _DRUGLIKE[0], "bad_smiles", _DRUGLIKE[1]]
        fps, idx_map = compute_erg_batch(smis)
        # bad_smiles dropped; CCO valid but zero-vector
        assert fps.shape[0] == 3
        assert idx_map == [0, 1, 3]


# ============================================================================
# ErgAdaptiveExplorer — only run if mlx-addons available
# ============================================================================

try:
    from mlx_addons.similarity import OnlineSingleLinkCluster  # noqa
    _HAS_MLX_ADDONS = True
except ImportError:
    _HAS_MLX_ADDONS = False


@pytest.mark.skipif(not _HAS_MLX_ADDONS, reason="mlx-addons not installed")
class TestErgAdaptiveExplorer:
    def test_accepts_distinct_druglike(self):
        from geneva2s.adaptive import ErgAdaptiveExplorer

        def gen(n, t):
            return _DRUGLIKE[:n] if n <= len(_DRUGLIKE) else _DRUGLIKE

        # Low cluster threshold so each distinct scaffold gets its own cluster
        e = ErgAdaptiveExplorer(generator_func=gen, cluster_threshold=0.9,
                                 max_per_cluster=10, max_freq=10,
                                 novelty_threshold=0.99)
        e.run_round(n_samples=len(_DRUGLIKE), verbose=False)
        # Most/all distinct druglike SMILES should be accepted (different ERGs)
        assert len(e.get_dataset()) >= len(_DRUGLIKE) - 2

    def test_rejects_repeats(self):
        from geneva2s.adaptive import ErgAdaptiveExplorer

        def gen(n, t): return [_DRUGLIKE[0]] * n

        e = ErgAdaptiveExplorer(generator_func=gen, max_freq=2)
        e.run_round(n_samples=5, verbose=False)
        # Only the first occurrence accepted (subsequent ones rejected by dup
        # and/or freq cap)
        assert len(e.get_dataset()) == 1
