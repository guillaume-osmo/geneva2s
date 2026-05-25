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
