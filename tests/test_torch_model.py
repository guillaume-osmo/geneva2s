"""PyTorch GenevaBiLSTM model tests: instantiation, forward shapes, param count."""
from __future__ import annotations

import pytest
import torch

from pathlib import Path

import numpy as np

from geneva2s.torch.model import GenevaBiLSTM, GenevaBiLSTMFused, apply_keras_init

REPO_ROOT = Path(__file__).resolve().parents[1]
PT_MODEL = REPO_ROOT / "models" / "geneva2s.pt"


class TestGenevaBiLSTM:
    def test_param_count(self):
        """Reference: Keras `model.summary()` reports 495,387 total params."""
        model = GenevaBiLSTM(vocab_size=27)
        n_params = sum(p.numel() for p in model.parameters())
        # ~497k expected (close to Keras 495,387 — small diff because PT has
        # split bias_ih+bias_hh while Keras has single bias)
        assert 490_000 < n_params < 510_000

    def test_forward_shape(self):
        model = GenevaBiLSTM(vocab_size=27)
        model.eval()
        x = torch.randint(0, 27, (4, 42), dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 27)

    def test_forward_different_batch_sizes(self):
        model = GenevaBiLSTM(vocab_size=27)
        model.eval()
        for B in [1, 8, 64]:
            x = torch.randint(0, 27, (B, 42), dtype=torch.long)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (B, 27)

    def test_n_branches(self):
        model = GenevaBiLSTM(vocab_size=27, n_branches=4)
        assert len(model.branches) == 4
        assert len(model.dropouts) == 4

    def test_keras_init_sets_forget_bias_to_one(self):
        """apply_keras_init should set forget gate bias_ih to 1, bias_hh to 0."""
        model = GenevaBiLSTM(vocab_size=27, init_keras=True)
        # bias_ih_l0 of the embedding LSTM has shape (4*128,) = (512,)
        # forget gate is the 2nd quarter: [128:256]
        bias_ih = model.embedding.bias_ih_l0.detach()
        assert torch.all(bias_ih[128:256] == 1.0)
        bias_hh = model.embedding.bias_hh_l0.detach()
        assert torch.all(bias_hh == 0.0)

    def test_default_init_is_keras(self):
        model = GenevaBiLSTM(vocab_size=27)
        # Default should apply Keras init
        bias_ih = model.embedding.bias_ih_l0.detach()
        assert torch.all(bias_ih[128:256] == 1.0)

    def test_no_keras_init_when_disabled(self):
        model = GenevaBiLSTM(vocab_size=27, init_keras=False)
        # Without keras init, forget gate bias_ih is PyTorch random init, not 1
        bias_ih = model.embedding.bias_ih_l0.detach()
        # PT default is uniform(-sqrt(1/hidden), +sqrt(1/hidden))
        assert not torch.all(bias_ih[128:256] == 1.0)


@pytest.mark.skipif(not PT_MODEL.exists(), reason="PT checkpoint not in repo")
class TestFusedEquivalence:
    """GenevaBiLSTMFused must produce numerically identical outputs to
    GenevaBiLSTM (mathematical reformulation, not a different model)."""

    def test_fused_matches_baseline(self):
        sd = torch.load(str(PT_MODEL), map_location="cpu", weights_only=True)
        m_base = GenevaBiLSTM(vocab_size=27, init_keras=False)
        m_base.load_state_dict(sd)
        m_base.eval()
        m_fused = GenevaBiLSTMFused(vocab_size=27, init_keras=False)
        m_fused.load_pt_state_dict(sd)
        m_fused.eval()
        torch.manual_seed(0)
        x = torch.randint(0, 27, (4, 42), dtype=torch.long)
        with torch.no_grad():
            o_base = m_base(x).numpy()
            o_fused = m_fused(x).numpy()
        max_diff = np.abs(o_base - o_fused).max()
        # ~4e-6 expected (float32 epsilon)
        assert max_diff < 1e-4, f"baseline vs fused max diff {max_diff}"

    def test_fused_forward_shape(self):
        m = GenevaBiLSTMFused(vocab_size=27)
        m.eval()
        x = torch.randint(0, 27, (2, 42), dtype=torch.long)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 27)
