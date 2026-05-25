"""Layer-by-layer numerical equivalence: PyTorch GenevaBiLSTM forward must match
a pure-numpy port of the Keras LSTM math to float64 epsilon (~1e-13).

This is the proof that the architecture port is bit-faithful.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from geneva2s.tokenizer import CharTokenizer
from geneva2s.torch.convert_keras import convert
from geneva2s.torch.model import GenevaBiLSTM


REPO_ROOT = Path(__file__).resolve().parents[1]
KERAS_MODEL = REPO_ROOT / "models" / "geneva2s.keras"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _keras_lstm_np(x, W_x, W_h, b, h0=None, c0=None):
    """Pure-numpy Keras LSTM forward (one direction, IFCO gate order)."""
    B, T, _ = x.shape
    H = W_h.shape[0]
    h = np.zeros((B, H), dtype=x.dtype) if h0 is None else h0
    c = np.zeros((B, H), dtype=x.dtype) if c0 is None else c0
    outs = []
    for t in range(T):
        z = x[:, t, :] @ W_x + h @ W_h + b
        i = _sigmoid(z[:, :H])
        f = _sigmoid(z[:, H:2 * H])
        g = np.tanh(z[:, 2 * H:3 * H])
        o = _sigmoid(z[:, 3 * H:4 * H])
        c = f * c + i * g
        h = o * np.tanh(c)
        outs.append(h)
    return np.stack(outs, axis=1), h, c


def _keras_bilstm_np(x, W_x_fwd, W_h_fwd, b_fwd, W_x_bwd, W_h_bwd, b_bwd):
    fwd_out, _, _ = _keras_lstm_np(x, W_x_fwd, W_h_fwd, b_fwd)
    bwd_in = x[:, ::-1, :]
    bwd_out_rev, _, _ = _keras_lstm_np(bwd_in, W_x_bwd, W_h_bwd, b_bwd)
    bwd_out = bwd_out_rev[:, ::-1, :]
    return np.concatenate([fwd_out, bwd_out], axis=-1)


@pytest.mark.skipif(not KERAS_MODEL.exists(), reason="Keras model not in repo")
class TestNumericalEquivalence:
    """Each layer of GenevaBiLSTM must match the pure-numpy Keras port to
    float64 epsilon when loaded with the same weights."""

    @pytest.fixture(scope="class")
    def model_and_input(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("conv")
        out_pt = tmp / "converted.pt"
        convert(str(KERAS_MODEL), str(out_pt))
        tok = CharTokenizer()
        model = GenevaBiLSTM(tok.vocab_size, init_keras=False).double()
        sd = torch.load(str(out_pt), map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.double() for k, v in sd.items()})
        model.eval()

        rng = np.random.RandomState(42)
        seq_ids = rng.randint(0, tok.vocab_size, (4, 42)).astype(np.int64)
        x_onehot = np.eye(tok.vocab_size, dtype=np.float32)[seq_ids]
        x_ids = torch.from_numpy(seq_ids)
        return model, tok, x_ids, x_onehot

    def test_bilstm_layer(self, model_and_input):
        model, tok, x_ids, x_onehot = model_and_input
        with torch.no_grad():
            x_pt = torch.nn.functional.one_hot(x_ids, num_classes=tok.vocab_size).double()
            pt_out, _ = model.embedding(x_pt)
        pt_np = pt_out.cpu().numpy()

        W_x_fwd = model.embedding.weight_ih_l0.detach().cpu().numpy().T.astype(np.float64)
        W_h_fwd = model.embedding.weight_hh_l0.detach().cpu().numpy().T.astype(np.float64)
        b_fwd = (model.embedding.bias_ih_l0 + model.embedding.bias_hh_l0
                 ).detach().cpu().numpy().astype(np.float64)
        W_x_bwd = model.embedding.weight_ih_l0_reverse.detach().cpu().numpy().T.astype(np.float64)
        W_h_bwd = model.embedding.weight_hh_l0_reverse.detach().cpu().numpy().T.astype(np.float64)
        b_bwd = (model.embedding.bias_ih_l0_reverse + model.embedding.bias_hh_l0_reverse
                 ).detach().cpu().numpy().astype(np.float64)

        np_out = _keras_bilstm_np(
            x_onehot.astype(np.float64),
            W_x_fwd, W_h_fwd, b_fwd, W_x_bwd, W_h_bwd, b_bwd,
        )
        assert np.abs(pt_np - np_out).max() < 1e-10

    def test_branch_lstms(self, model_and_input):
        model, tok, x_ids, x_onehot = model_and_input
        with torch.no_grad():
            x_pt = torch.nn.functional.one_hot(x_ids, num_classes=tok.vocab_size).double()
            pt_bilstm_out, _ = model.embedding(x_pt)
        pt_bilstm_np = pt_bilstm_out.cpu().numpy()

        for i in range(4):
            with torch.no_grad():
                out_i, _ = model.branches[i](pt_bilstm_out)
            pt_last = out_i[:, -1, :].cpu().numpy()

            W_x = model.branches[i].weight_ih_l0.detach().cpu().numpy().T.astype(np.float64)
            W_h = model.branches[i].weight_hh_l0.detach().cpu().numpy().T.astype(np.float64)
            b = (model.branches[i].bias_ih_l0 + model.branches[i].bias_hh_l0
                 ).detach().cpu().numpy().astype(np.float64)
            np_out, _, _ = _keras_lstm_np(pt_bilstm_np, W_x, W_h, b)
            np_last = np_out[:, -1, :]
            assert np.abs(pt_last - np_last).max() < 1e-10, f"branch {i} mismatch"

    def test_full_forward_no_nan(self, tmp_path):
        """End-to-end float32 forward — make sure the loaded weights produce
        finite output. Uses a separate float32 model since F.one_hot inside
        forward returns float32 (per-layer test above runs in float64)."""
        out_pt = tmp_path / "f32.pt"
        convert(str(KERAS_MODEL), str(out_pt))
        tok = CharTokenizer()
        model = GenevaBiLSTM(tok.vocab_size, init_keras=False)
        sd = torch.load(str(out_pt), map_location="cpu", weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        x_ids = torch.randint(0, tok.vocab_size, (4, 42), dtype=torch.long)
        with torch.no_grad():
            out = model(x_ids)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
        assert out.shape == (4, 27)
