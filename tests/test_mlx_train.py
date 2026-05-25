"""MLX training pipeline tests: loss decreases, save/load round-trips."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx.nn")
pytest.importorskip("mlx.optimizers")

from geneva2s.mlx.model import GenevaBiLSTMMLX
from geneva2s.mlx.train import fit, load_state, save_state
from geneva2s.tokenizer import CharTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "chembl_9k_organic.smi"


@pytest.mark.skipif(not DATA.exists(), reason="data file not in repo")
def test_baseline_training_loss_decreases():
    """3-epoch training on a 500-mol subset: loss should decrease meaningfully."""
    with open(DATA) as f:
        raw = [line.strip() for line in f if line.strip()][:500]
    tok = CharTokenizer()
    text = tok.prepare_corpus(raw)
    X, y = tok.sliding_window(text)
    model = GenevaBiLSTMMLX(tok.vocab_size)
    history = fit(model, X, y, num_epochs=3, batch_size=128, verbose=False)
    assert len(history) == 3
    # loss should be lower at end than start
    assert history[-1] < history[0], f"loss did not decrease: {history}"


def test_save_load_roundtrip_safetensors(tmp_path):
    """save_state → load_state via .safetensors round-trip preserves outputs."""
    tok = CharTokenizer()
    model = GenevaBiLSTMMLX(tok.vocab_size)
    rng = np.random.RandomState(0)
    x = mx.array(rng.randint(0, tok.vocab_size, (4, 42)).astype(np.int32))
    o1 = np.array(model(x)); mx.eval(o1)

    path = tmp_path / "ckpt.safetensors"
    save_state(model, str(path))
    model2 = GenevaBiLSTMMLX(tok.vocab_size)
    load_state(model2, str(path))
    o2 = np.array(model2(x)); mx.eval(o2)
    assert np.allclose(o1, o2, atol=1e-6), \
        f"safetensors round-trip diff: {np.abs(o1 - o2).max()}"


def test_save_load_roundtrip_npz(tmp_path):
    """save_state → load_state via .npz round-trip preserves outputs."""
    tok = CharTokenizer()
    model = GenevaBiLSTMMLX(tok.vocab_size)
    rng = np.random.RandomState(0)
    x = mx.array(rng.randint(0, tok.vocab_size, (4, 42)).astype(np.int32))
    o1 = np.array(model(x)); mx.eval(o1)

    path = tmp_path / "ckpt.npz"
    save_state(model, str(path))
    model2 = GenevaBiLSTMMLX(tok.vocab_size)
    load_state(model2, str(path))
    o2 = np.array(model2(x)); mx.eval(o2)
    assert np.allclose(o1, o2, atol=1e-6), \
        f"npz round-trip diff: {np.abs(o1 - o2).max()}"


def test_save_state_pt_format(tmp_path):
    """save_state with .pt extension produces a PyTorch-loadable state_dict."""
    torch = pytest.importorskip("torch")
    from geneva2s.torch.model import GenevaBiLSTM

    tok = CharTokenizer()
    mlx_model = GenevaBiLSTMMLX(tok.vocab_size)
    path = tmp_path / "from_mlx.pt"
    save_state(mlx_model, str(path))

    # Should load cleanly into the PyTorch GenevaBiLSTM
    pt_model = GenevaBiLSTM(tok.vocab_size, init_keras=False)
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    pt_model.load_state_dict(sd, strict=True)


# mlx-addons is optional — only test metal training if importable
try:
    from mlx_addons.recurrent import MetalLSTM  # noqa
    _HAS_MLX_ADDONS = True
except ImportError:
    _HAS_MLX_ADDONS = False


@pytest.mark.skipif(not (_HAS_MLX_ADDONS and DATA.exists()),
                    reason="mlx-addons not installed or no data")
def test_metal_training_loss_decreases():
    """3-epoch training through MetalLSTM (using the VJP kernel)."""
    from geneva2s.mlx.model import GenevaBiLSTMMLXMetal
    with open(DATA) as f:
        raw = [line.strip() for line in f if line.strip()][:500]
    tok = CharTokenizer()
    text = tok.prepare_corpus(raw)
    X, y = tok.sliding_window(text)
    model = GenevaBiLSTMMLXMetal(tok.vocab_size)
    history = fit(model, X, y, num_epochs=3, batch_size=128, verbose=False)
    assert len(history) == 3
    assert history[-1] < history[0], f"metal loss did not decrease: {history}"
