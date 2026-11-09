"""Small sequence-to-one SoC estimators sized for a Raspberry Pi.

Input:  (batch, window, 3) normalised [V, I, T]
Output: (batch,) SoC estimate at the last step of the window
"""

from __future__ import annotations

import math

import torch
from torch import nn


class SoCTransformer(nn.Module):
    """Transformer encoder (Vaswani et al., NeurIPS 2017) with pre-layer norm,
    fixed sinusoidal positions, and a linear head on the final token.

    The input is split into non-overlapping patches of ``patch`` samples, each
    embedded by one linear map (a strided convolution), as in PatchTST (Nie et
    al., ICLR 2023). With patch = 5 a 300-sample window becomes 60 tokens, which
    cuts attention cost by 25x compared with one token per sample and keeps
    inference light enough for a Raspberry Pi. patch = 1 gives one token per sample.
    """

    def __init__(self, n_in=3, d_model=32, n_heads=4, n_layers=2, d_ff=64, window=300, dropout=0.1, patch=5):
        super().__init__()
        if window % patch:
            raise ValueError("window must be a multiple of patch")
        self.patch = patch
        self.inp = nn.Conv1d(n_in, d_model, kernel_size=patch, stride=patch)
        n_tok = window // patch
        pe = torch.zeros(n_tok, d_model)
        pos = torch.arange(n_tok).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2)).transpose(1, 2)
        h = h + self.pe[: h.shape[1]]
        h = self.encoder(h)
        return self.head(self.norm(h[:, -1])).squeeze(-1)


class SoCLSTM(nn.Module):
    """Single-layer LSTM baseline (cf. Chemali et al., IEEE TIE 2018)."""

    def __init__(self, n_in=3, hidden=32, n_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, n_layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


MODELS = {"transformer": SoCTransformer, "lstm": SoCLSTM}


def build_model(name: str, **kwargs) -> nn.Module:
    return MODELS[name](**kwargs)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
