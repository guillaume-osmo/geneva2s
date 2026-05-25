"""Test Keras → PyTorch weight conversion: shapes, key matching, load success."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from geneva2s.torch.convert_keras import convert
from geneva2s.torch.model import GenevaBiLSTM


REPO_ROOT = Path(__file__).resolve().parents[1]
KERAS_MODEL = REPO_ROOT / "models" / "geneva2s.keras"


@pytest.mark.skipif(not KERAS_MODEL.exists(), reason="Keras model not in repo")
class TestKerasConversion:
    def test_convert_produces_state_dict(self, tmp_path):
        out_pt = tmp_path / "converted.pt"
        sd = convert(str(KERAS_MODEL), str(out_pt))
        assert out_pt.exists()
        assert isinstance(sd, dict)
        assert len(sd) == 26  # 8 embedding (4 each direction) + 16 branches + 2 head

    def test_converted_weights_load_into_model(self, tmp_path):
        out_pt = tmp_path / "converted.pt"
        convert(str(KERAS_MODEL), str(out_pt))
        model = GenevaBiLSTM(vocab_size=27, init_keras=False)
        state = torch.load(str(out_pt), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)  # raises if keys/shapes mismatch

    def test_loaded_model_forward_runs(self, tmp_path):
        out_pt = tmp_path / "converted.pt"
        convert(str(KERAS_MODEL), str(out_pt))
        model = GenevaBiLSTM(vocab_size=27, init_keras=False)
        state = torch.load(str(out_pt), map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        x = torch.randint(0, 27, (2, 42), dtype=torch.long)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 27)
        assert not torch.isnan(out).any()
