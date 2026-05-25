"""PyTorch port of the GENEVA²S architecture (BaseModel in the original).

Layers=[128, 64], Bidirectional=[True, False], minimodels=4, merge=0 (concat).
One-hot input (no embedding), biLSTM first layer, 4 parallel unidirectional
second-layer LSTM branches, per-branch Dropout(0.3), concatenated to a 256-dim
feature, then Dense(vocab) head. Softmax is applied by CrossEntropyLoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_keras_init(module: nn.Module) -> None:
    """Match Keras defaults: GlorotUniform kernel, Orthogonal recurrent kernel,
    Zeros bias, unit_forget_bias=True (forget gate bias initialized to 1).
    PyTorch LSTM gate order is IFGO — same as Keras (input/forget/cell/output).
    """
    for m in module.modules():
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                    if "bias_ih" in name:
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size:2 * hidden_size].fill_(1.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class GenevaBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden=(128, 64),
        bidirectional=(True, False),
        n_branches: int = 4,
        dropout: float = 0.3,
        init_keras: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.n_branches = n_branches

        self.embedding = nn.LSTM(
            vocab_size, hidden[0], batch_first=True, bidirectional=bidirectional[0]
        )
        in2 = hidden[0] * (2 if bidirectional[0] else 1)

        self.branches = nn.ModuleList([
            nn.LSTM(in2, hidden[1], batch_first=True, bidirectional=bidirectional[1])
            for _ in range(n_branches)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout) for _ in range(n_branches)
        ])

        out_per_branch = hidden[1] * (2 if bidirectional[1] else 1)
        self.head = nn.Linear(out_per_branch * n_branches, vocab_size)

        if init_keras:
            apply_keras_init(self)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """x_ids: (B, T) int. Converts to one-hot internally to match original."""
        x = F.one_hot(x_ids, num_classes=self.vocab_size).float()
        e, _ = self.embedding(x)
        branch_outs = []
        for lstm, drop in zip(self.branches, self.dropouts):
            o, _ = lstm(e)
            o = drop(o[:, -1, :])
            branch_outs.append(o)
        return self.head(torch.cat(branch_outs, dim=-1))


class GenevaBiLSTMFused(nn.Module):
    """Fused twin of GenevaBiLSTM: the 4 second-layer branches are collapsed
    into a single grouped LSTM step (one big input-projection matmul + an
    einsum recurrent step). Bit-identical to GenevaBiLSTM (max diff ~5e-6,
    float32 epsilon).

    Speed profile on Apple Silicon MPS (vs nn.LSTM-based baseline):
    - batch ≤ 64:   SLOWER (Python time-loop overhead beats nn.LSTM's fused scan)
    - batch ≥ 256:  ~30% FASTER (compute dominates, single big matmul wins)

    Use for large-batch offline inference; use the baseline GenevaBiLSTM for
    training (where batches are usually ≤256). The MLX-fused counterpart wins
    at every batch size; PyTorch's nn.LSTM is too well-optimized to beat in
    the small-batch regime."""

    def __init__(
        self,
        vocab_size: int,
        hidden=(128, 64),
        n_branches: int = 4,
        init_keras: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_branches = n_branches

        self.embedding = nn.LSTM(vocab_size, hidden[0], batch_first=True, bidirectional=True)
        in2 = hidden[0] * 2
        H2 = hidden[1]
        G = n_branches
        # Stacked branch weights as nn.Parameters
        self.branch_Wx = nn.Parameter(torch.empty(G, 4 * H2, in2))
        self.branch_Wh = nn.Parameter(torch.empty(G, 4 * H2, H2))
        self.branch_b = nn.Parameter(torch.empty(G, 4 * H2))
        self.head = nn.Linear(H2 * G, vocab_size)

        if init_keras:
            # Glorot for Wx, Orthogonal for Wh, zero bias except forget=1 in bias_ih portion
            for g in range(G):
                nn.init.xavier_uniform_(self.branch_Wx[g])
                nn.init.orthogonal_(self.branch_Wh[g])
            nn.init.zeros_(self.branch_b)
            # Forget gate is the 2nd quarter of (4*H2,); set to 1 for unit_forget_bias
            self.branch_b.data[:, H2:2 * H2] = 1.0
            apply_keras_init(self)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(x_ids, num_classes=self.vocab_size).float()
        e, _ = self.embedding(x)                                  # (B, T, 256)
        B, T, D_in = e.shape
        G = self.n_branches
        H = self.hidden[1]
        # ONE big input-projection matmul instead of 4
        Wx_flat = self.branch_Wx.reshape(-1, D_in)
        e_proj = (e @ Wx_flat.T).reshape(B, T, G, 4 * H)
        h = torch.zeros(B, G, H, device=e.device, dtype=e.dtype)
        c = torch.zeros(B, G, H, device=e.device, dtype=e.dtype)
        for t in range(T):
            gates_h = torch.einsum("bgh,gih->bgi", h, self.branch_Wh)
            gates = e_proj[:, t, :, :] + gates_h + self.branch_b
            i = torch.sigmoid(gates[..., :H])
            f = torch.sigmoid(gates[..., H:2 * H])
            g_ = torch.tanh(gates[..., 2 * H:3 * H])
            o = torch.sigmoid(gates[..., 3 * H:4 * H])
            c = f * c + i * g_
            h = o * torch.tanh(c)
        return self.head(h.reshape(B, G * H))

    def load_pt_state_dict(self, sd: dict) -> None:
        """Load weights from a GenevaBiLSTM state_dict (auto-stacks branches)."""
        for k in ("embedding.weight_ih_l0", "embedding.weight_hh_l0",
                  "embedding.bias_ih_l0", "embedding.bias_hh_l0",
                  "embedding.weight_ih_l0_reverse", "embedding.weight_hh_l0_reverse",
                  "embedding.bias_ih_l0_reverse", "embedding.bias_hh_l0_reverse"):
            getattr(self.embedding, k.split(".", 1)[1]).data = sd[k]
        self.branch_Wx.data = torch.stack(
            [sd[f"branches.{i}.weight_ih_l0"] for i in range(self.n_branches)]
        )
        self.branch_Wh.data = torch.stack(
            [sd[f"branches.{i}.weight_hh_l0"] for i in range(self.n_branches)]
        )
        self.branch_b.data = torch.stack([
            sd[f"branches.{i}.bias_ih_l0"] + sd[f"branches.{i}.bias_hh_l0"]
            for i in range(self.n_branches)
        ])
        self.head.weight.data = sd["head.weight"]
        self.head.bias.data = sd["head.bias"]
