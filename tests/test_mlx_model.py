"""MLX variant tests: instantiation, weight load, equivalence to PyTorch."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mlxnn = pytest.importorskip("mlx.nn")

from geneva2s.mlx.model import GenevaBiLSTMMLX, GenevaBiLSTMMLXFused

REPO_ROOT = Path(__file__).resolve().parents[1]
PT_MODEL = REPO_ROOT / "models" / "geneva2s.pt"


class TestGenevaBiLSTMMLX:
    def test_instantiation(self):
        m = GenevaBiLSTMMLX(vocab_size=27)
        assert m.vocab_size == 27
        assert m.n_branches == 4
        assert len(m.branches) == 4

    def test_forward_shape(self):
        m = GenevaBiLSTMMLX(vocab_size=27)
        x_np = np.random.randint(0, 27, (4, 42)).astype(np.int32)
        x = mx.array(x_np)
        out = m(x)
        mx.eval(out)
        assert out.shape == (4, 27)

    def test_forward_no_nan(self):
        m = GenevaBiLSTMMLX(vocab_size=27)
        x_np = np.random.randint(0, 27, (2, 42)).astype(np.int32)
        x = mx.array(x_np)
        out = m(x)
        mx.eval(out)
        arr = np.array(out)
        assert not np.isnan(arr).any()
        assert not np.isinf(arr).any()


@pytest.mark.skipif(not PT_MODEL.exists(), reason="PT checkpoint not in repo")
class TestPTWeightLoading:
    def test_load_pt_state_dict_runs(self):
        m = GenevaBiLSTMMLX(vocab_size=27)
        m.load_pt_checkpoint(str(PT_MODEL))
        # Forward should still work after loading
        x_np = np.random.randint(0, 27, (2, 42)).astype(np.int32)
        x = mx.array(x_np)
        out = m(x)
        mx.eval(out)
        assert out.shape == (2, 27)


@pytest.mark.skipif(not PT_MODEL.exists(), reason="PT checkpoint not in repo")
class TestPTtoMLXEquivalence:
    """With the same PT weights loaded, MLX and PyTorch forward should give
    numerically equivalent outputs (within float32 epsilon)."""

    def test_outputs_match_pytorch(self):
        torch = pytest.importorskip("torch")
        from geneva2s.torch.model import GenevaBiLSTM

        # Load same weights into both frameworks
        sd = torch.load(str(PT_MODEL), map_location="cpu", weights_only=True)
        pt_model = GenevaBiLSTM(vocab_size=27, init_keras=False)
        pt_model.load_state_dict(sd)
        pt_model.eval()

        mlx_model = GenevaBiLSTMMLX(vocab_size=27)
        mlx_model.load_pt_state_dict(sd)

        # Same input
        rng = np.random.RandomState(0)
        x_np = rng.randint(0, 27, (4, 42)).astype(np.int64)

        with torch.no_grad():
            pt_out = pt_model(torch.from_numpy(x_np)).numpy()
        mlx_out_arr = mlx_model(mx.array(x_np.astype(np.int32)))
        mx.eval(mlx_out_arr)
        mlx_out = np.array(mlx_out_arr)

        # Allow float32 tolerance — MPS LSTM kernel and MLX LSTM kernel
        # accumulate in slightly different order
        max_diff = np.abs(pt_out - mlx_out).max()
        assert max_diff < 1e-3, f"PT vs MLX max diff {max_diff}"


@pytest.mark.skipif(not PT_MODEL.exists(), reason="PT checkpoint not in repo")
class TestFusedEquivalence:
    """GenevaBiLSTMMLXFused should produce numerically identical outputs to
    GenevaBiLSTMMLX (it's a mathematical reformulation, not a different model)."""

    def test_fused_matches_baseline(self):
        m_base = GenevaBiLSTMMLX(vocab_size=27)
        m_base.load_pt_checkpoint(str(PT_MODEL))
        m_fused = GenevaBiLSTMMLXFused(vocab_size=27)
        m_fused.load_pt_checkpoint(str(PT_MODEL))

        rng = np.random.RandomState(0)
        x_np = rng.randint(0, 27, (4, 42)).astype(np.int32)
        x = mx.array(x_np)
        o_base = np.array(m_base(x)); mx.eval(o_base)
        o_fused = np.array(m_fused(x)); mx.eval(o_fused)
        max_diff = np.abs(o_base - o_fused).max()
        # ~5e-6 expected (float32 epsilon)
        assert max_diff < 1e-4, f"baseline vs fused max diff {max_diff}"
