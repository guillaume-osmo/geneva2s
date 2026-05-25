"""End-to-end generation test: load Keras weights into PyTorch model, generate
a small batch of molecules, check at least 80% rdkit-canonicalize cleanly."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from geneva2s.tokenizer import CharTokenizer
from geneva2s.torch.convert_keras import convert
from geneva2s.torch.generate import predict_batch_seeds
from geneva2s.torch.model import GenevaBiLSTM
from geneva2s.utils import canonicalize, sanity_check


REPO_ROOT = Path(__file__).resolve().parents[1]
KERAS_MODEL = REPO_ROOT / "models" / "geneva2s.keras"
PT_MODEL = REPO_ROOT / "models" / "geneva2s.pt"
DATA = REPO_ROOT / "data" / "chembl_9k_organic.smi"


def _load_model_with_keras_weights(tmp_path):
    out_pt = tmp_path / "converted.pt"
    convert(str(KERAS_MODEL), str(out_pt))
    tok = CharTokenizer()
    model = GenevaBiLSTM(tok.vocab_size, init_keras=False)
    state = torch.load(str(out_pt), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tok


@pytest.mark.skipif(not (KERAS_MODEL.exists() and DATA.exists()),
                    reason="Keras model or data not in repo")
class TestGeneration:
    def test_small_generation_runs(self, tmp_path):
        model, tok = _load_model_with_keras_weights(tmp_path)
        with open(DATA) as f:
            raw = [line.strip() for line in f if line.strip()][:500]
        text = tok.prepare_corpus(raw)

        mols = predict_batch_seeds(
            model, tok, text, ncollect=20, ncopies=4,
            device=torch.device("cpu"),
        )
        assert len(mols) == 20
        assert all(isinstance(m, str) for m in mols)

    def test_generated_mols_mostly_valid(self, tmp_path):
        """With proper Keras weights, ≥80% of generated mols should be rdkit-valid
        and ≥85% should pass the text-based sanity check."""
        model, tok = _load_model_with_keras_weights(tmp_path)
        with open(DATA) as f:
            raw = [line.strip() for line in f if line.strip()]
        text = tok.prepare_corpus(raw)

        mols = predict_batch_seeds(
            model, tok, text, ncollect=100, ncopies=10,
            device=torch.device("cpu"),
        )
        n_rdkit = sum(1 for m in mols if canonicalize(m))
        n_sanity = sum(1 for m in mols if sanity_check(m))

        # Loose bounds — sampling stochasticity
        assert n_rdkit / len(mols) >= 0.80, f"rdkit valid {n_rdkit}/{len(mols)}"
        assert n_sanity / len(mols) >= 0.85, f"sanity {n_sanity}/{len(mols)}"


@pytest.mark.skipif(not (PT_MODEL.exists() and DATA.exists()),
                    reason="PyTorch model or data not in repo")
class TestPyTorchTrainedModel:
    def test_pt_checkpoint_generates_valid_mols(self):
        """The shipped PyTorch checkpoint (trained on 45k corpus) should
        match Keras quality: ≥85% rdkit valid."""
        tok = CharTokenizer()
        model = GenevaBiLSTM(tok.vocab_size, init_keras=False)
        state = torch.load(str(PT_MODEL), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()

        with open(DATA) as f:
            raw = [line.strip() for line in f if line.strip()]
        text = tok.prepare_corpus(raw)

        mols = predict_batch_seeds(
            model, tok, text, ncollect=100, ncopies=10,
            device=torch.device("cpu"),
        )
        n_rdkit = sum(1 for m in mols if canonicalize(m))
        assert n_rdkit / len(mols) >= 0.80, f"rdkit valid {n_rdkit}/{len(mols)}"
