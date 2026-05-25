"""TensorFlow / Keras variant tests. Optional — skipped if TF not installed.
Use TF 2.15.1 + tensorflow-metal 1.1.0 on Apple Silicon."""
from __future__ import annotations

from pathlib import Path

import pytest

tf = pytest.importorskip("tensorflow")

from geneva2s.tf.model import build_model, load_keras_model

REPO_ROOT = Path(__file__).resolve().parents[1]
KERAS_MODEL = REPO_ROOT / "models" / "geneva2s.keras"


class TestBuildModel:
    def test_default_shapes(self):
        model = build_model()
        assert model.input_shape == (None, 42, 27)
        assert model.output_shape == (None, 27)

    def test_param_count(self):
        model = build_model()
        # Same architecture as PyTorch port — should match Keras summary
        n = model.count_params()
        assert 490_000 < n < 510_000


@pytest.mark.skipif(not KERAS_MODEL.exists(), reason="Keras model not in repo")
class TestLoadKerasModel:
    def test_loads_without_error(self):
        """Only works in TF 2.15.x — the .keras file was saved with Keras 2."""
        try:
            model = load_keras_model(str(KERAS_MODEL))
        except (TypeError, ImportError) as e:
            pytest.skip(f"Keras version mismatch ({e}); use TF 2.15.x to load this file")
        assert model.input_shape == (None, 42, 27)
        assert model.output_shape == (None, 27)
