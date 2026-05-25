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


# mlx-addons (custom Metal kernels) is optional — only test if importable
try:
    from mlx_addons.recurrent import MetalLSTM  # noqa
    _HAS_MLX_ADDONS = True
except ImportError:
    _HAS_MLX_ADDONS = False


@pytest.mark.skipif(not (_HAS_MLX_ADDONS and PT_MODEL.exists()),
                    reason="mlx-addons not installed or no PT checkpoint")
class TestMetalVariants:
    """GenevaBiLSTMMLXMetal{,Grouped} use custom Metal kernels — should match
    the baseline within float32 epsilon."""

    def test_metal_matches_baseline(self):
        from geneva2s.mlx.model import GenevaBiLSTMMLXMetal
        m_base = GenevaBiLSTMMLX(vocab_size=27)
        m_base.load_pt_checkpoint(str(PT_MODEL))
        m_metal = GenevaBiLSTMMLXMetal(vocab_size=27)
        m_metal.load_pt_checkpoint(str(PT_MODEL))
        rng = np.random.RandomState(0)
        x = mx.array(rng.randint(0, 27, (4, 42)).astype(np.int32))
        o_b = np.array(m_base(x)); mx.eval(o_b)
        o_m = np.array(m_metal(x)); mx.eval(o_m)
        assert np.abs(o_b - o_m).max() < 1e-3, "metal vs baseline too far"

    def test_metal_grouped_matches_baseline(self):
        from geneva2s.mlx.model import GenevaBiLSTMMLXMetalGrouped
        m_base = GenevaBiLSTMMLX(vocab_size=27)
        m_base.load_pt_checkpoint(str(PT_MODEL))
        m_mg = GenevaBiLSTMMLXMetalGrouped(vocab_size=27)
        m_mg.load_pt_checkpoint(str(PT_MODEL))
        rng = np.random.RandomState(0)
        x = mx.array(rng.randint(0, 27, (4, 42)).astype(np.int32))
        o_b = np.array(m_base(x)); mx.eval(o_b)
        o_mg = np.array(m_mg(x)); mx.eval(o_mg)
        assert np.abs(o_b - o_mg).max() < 1e-3, "metal-grouped vs baseline too far"

    def test_precise_mode_tighter(self):
        """Precise math toggle should give closer match to baseline than fast."""
        from geneva2s.mlx.model import GenevaBiLSTMMLXMetal
        m_base = GenevaBiLSTMMLX(vocab_size=27)
        m_base.load_pt_checkpoint(str(PT_MODEL))
        rng = np.random.RandomState(0)
        x = mx.array(rng.randint(0, 27, (4, 42)).astype(np.int32))
        o_b = np.array(m_base(x)); mx.eval(o_b)
        m_fast = GenevaBiLSTMMLXMetal(vocab_size=27, precise=False)
        m_fast.load_pt_checkpoint(str(PT_MODEL))
        m_prec = GenevaBiLSTMMLXMetal(vocab_size=27, precise=True)
        m_prec.load_pt_checkpoint(str(PT_MODEL))
        diff_fast = np.abs(o_b - np.array(m_fast(x))).max()
        diff_prec = np.abs(o_b - np.array(m_prec(x))).max()
        # precise should match the baseline as well or better than fast
        assert diff_prec <= diff_fast + 1e-7, f"precise {diff_prec:.2e} vs fast {diff_fast:.2e}"
