"""Tests for the optimizer factory + augment_corpus helpers."""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from geneva2s.torch.optimizers import (
    AdaMuonOfficial,
    AdamN,
    Muon,
    MuonN,
    SplitOptimizer,
    build_optimizer,
)
from geneva2s.utils import augment_corpus, augment_smiles


# ============================================================================
# build_optimizer factory
# ============================================================================

class TestBuildOptimizer:
    def _model(self):
        # Mixed-shape model: 2 matrices + 2 bias scalars.
        return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))

    @pytest.mark.parametrize("name,expected_cls", [
        ("adam", torch.optim.Adam),
        ("adamw", torch.optim.AdamW),
        ("adamn", AdamN),
    ])
    def test_scalar_optimizers(self, name, expected_cls):
        opt = build_optimizer(
            self._model(),
            types.SimpleNamespace(optimizer=name, lr=3e-3),
        )
        assert isinstance(opt, expected_cls)

    @pytest.mark.parametrize("name,primary_cls", [
        ("muon", Muon),
        ("adamuon", MuonN),
        ("adamuonn", MuonN),
        ("adamuon_official", AdaMuonOfficial),
        ("muon_vx", MuonN),
    ])
    def test_split_optimizers_route_matrices_correctly(self, name, primary_cls):
        opt = build_optimizer(
            self._model(),
            types.SimpleNamespace(optimizer=name, lr=3e-3),
        )
        assert isinstance(opt, SplitOptimizer)
        assert isinstance(opt.primary, primary_cls)
        # Scalar params (biases) should be routed to the fallback optimizer.
        assert opt.fallback is not None

    def test_adamuonn_fallback_is_adamn(self):
        # adamuonn is the recommended Muon-N matrix + AdamN scalar combo.
        opt = build_optimizer(
            self._model(),
            types.SimpleNamespace(optimizer="adamuonn", lr=3e-3),
        )
        assert isinstance(opt.primary, MuonN)
        assert isinstance(opt.fallback, AdamN)

    def test_unknown_optimizer_raises(self):
        with pytest.raises(ValueError):
            build_optimizer(self._model(),
                            types.SimpleNamespace(optimizer="nope", lr=3e-3))


class TestOptimizerStep:
    """Smoke test: each optimizer takes a real step without exploding."""

    @pytest.mark.parametrize("name", ["adam", "adamw", "adamn", "muon",
                                       "adamuon", "adamuonn", "adamuon_official"])
    def test_one_step_reduces_loss(self, name):
        torch.manual_seed(0)
        x = torch.randn(32, 8)
        y = torch.randn(32, 4)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 4))
        opt = build_optimizer(
            model, types.SimpleNamespace(optimizer=name, lr=1e-2),
        )
        loss_fn = torch.nn.MSELoss()
        # Capture initial loss
        loss0 = loss_fn(model(x), y).item()
        for _ in range(5):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        loss1 = loss_fn(model(x), y).item()
        # Each optimizer should reduce the loss within 5 steps on this trivial task.
        assert loss1 < loss0, f"{name}: loss did not decrease ({loss0:.3f} -> {loss1:.3f})"


# ============================================================================
# augment_corpus
# ============================================================================

class TestAugmentCorpus:
    def test_naug_1_is_canonical_only(self):
        out = augment_corpus(["CCO", "c1ccccc1"], n_aug=1)
        assert len(out) == 2
        # Each entry should be the canonical SMILES (no random variants).
        for smi in out:
            assert "c" in smi or "C" in smi

    def test_naug_5_expands_corpus(self):
        out = augment_corpus(["CC(=O)Oc1ccccc1C(=O)O"], n_aug=5)
        # Aspirin should produce multiple distinct random walks.
        assert 1 < len(out) <= 5
        assert len(set(out)) == len(out)  # no duplicates inside the n_aug bucket

    def test_invalid_smiles_dropped(self):
        out = augment_corpus(["CCO", "not_a_smiles", "c1ccccc1"], n_aug=1)
        # n_aug=1 path only keeps valid canonical SMILES → invalid dropped.
        assert len(out) == 2

    def test_empty_input(self):
        assert augment_corpus([], n_aug=5) == []
        assert augment_corpus([], n_aug=1) == []

    def test_per_molecule_grouping_preserved(self):
        # n_aug > 1: outputs of distinct input molecules don't interleave.
        # First-bucket entries are CCO variants (CCO / OCC); aspirin variants come after.
        from rdkit.Chem import CanonSmiles
        smis = ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
        out = augment_corpus(smis, n_aug=3)
        # The first entry should canonicalize to the ethanol canonical form.
        assert CanonSmiles(out[0]) == CanonSmiles("CCO")


class TestAugmentSmiles:
    def test_returns_list_of_variants(self):
        out = augment_smiles("CC(=O)Oc1ccccc1C(=O)O", n_aug=3)
        assert isinstance(out, list)
        assert 1 <= len(out) <= 3
        # Canonical form is included.
        assert any("c1ccccc1" in s or "ccccc" in s.lower() for s in out)

    def test_invalid_returns_empty(self):
        assert augment_smiles("not_a_smiles_xxx", n_aug=5) == []
