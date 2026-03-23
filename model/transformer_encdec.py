"""
Transformer Encoder-Decoder model for PI forecasting with covariates.

The model accept:
    x = (lag_regressor, future_regressor)
      lag_regressor  : (B, num_lag_vars,    num_lag)
      future_regressor: (B, num_future_vars, num_ahead)
  and return:
    pi     : (B, num_ahead * 2) - [lower_1, upper_1, lower_2, upper_2, ...]
    points : (B, num_ahead)

Uses sinusoidal positional encoding and a standard Transformer encoder-decoder
"""

import math
from typing import Tuple

import torch
import torch.nn as nn

from model.lstm_encdec import _collect_predictions, _InputNorm, _OutputHead


class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderDecoder(nn.Module):
    """
    Transformer-based encoder-decoder for prediction interval forecasting.

    Encoder: processes lag covariates through a stack of Transformer encoder layers.
    Decoder: processes future covariates through a stack of Transformer decoder layers,
             attending to encoder outputs via cross-attention.
    At each decoder time-step an output head predicts (point, lower, upper).

    The decoder uses a causal mask so that position t can only attend to
    positions <= t in the future covariate sequence.
    """

    def __init__(
        self,
        num_lag_vars: int = 3,
        num_future_vars: int = 3,
        num_ahead: int = 16,
        d_model: int = 128,
        num_enc_layers: int = 3,
        num_dec_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_lag_len: int = 512,
    ):
        super().__init__()
        self.num_ahead = num_ahead
        self.d_model = d_model

        self.lag_norm = _InputNorm(num_lag_vars)
        self.fut_norm = _InputNorm(num_future_vars)

        self.enc_input_proj = nn.Linear(num_lag_vars, d_model)
        self.dec_input_proj = nn.Linear(num_future_vars, d_model)

        self.enc_pe = _SinusoidalPE(d_model, max_len=max_lag_len, dropout=dropout)
        self.dec_pe = _SinusoidalPE(d_model, max_len=num_ahead + 16, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_enc_layers,
            enable_nested_tensor=False,
        )

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            dec_layer,
            num_layers=num_dec_layers,
        )

        self.head = _OutputHead(d_model, dropout)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:

        lag, fut = x  # (B, C, T), where C = num_lag_vars or num_future_vars

        lag = self.lag_norm(lag).transpose(1, 2)  # (B, T_lag, C_lag)
        fut = self.fut_norm(fut).transpose(1, 2)  # (B, T_ahead, C_fut)

        # Project to d_model and add positional encoding
        enc_in = self.enc_pe(self.enc_input_proj(lag))  # (B, T_lag, d_model)
        dec_in = self.dec_pe(self.dec_input_proj(fut))  # (B, T_ahead, d_model)

        memory = self.encoder(enc_in)  # (B, T_lag, d_model)

        tgt_mask = self._causal_mask(self.num_ahead, dec_in.device)

        dec_out = self.decoder(
            tgt=dec_in,
            memory=memory,
            tgt_mask=tgt_mask,
        )  # (B, T_ahead, d_model)

        # Prediction
        points_list, delta_lowers_list, delta_uppers_list = [], [], []
        for t in range(self.num_ahead):
            pt, delta_lo, delta_up = self.head(dec_out[:, t, :])
            points_list.append(pt)
            delta_lowers_list.append(delta_lo)
            delta_uppers_list.append(delta_up)

        return _collect_predictions(points_list, delta_lowers_list, delta_uppers_list)
