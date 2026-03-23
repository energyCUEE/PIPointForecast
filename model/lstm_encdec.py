"""
LSTM Encoder-Decoder models for PI forecasting with covariates.

The model accept:
    x = (lag_regressor, future_regressor)
      lag_regressor  : (B, num_lag_vars,    num_lag)
      future_regressor: (B, num_future_vars, num_ahead)
  and return:
    pi     : (B, num_ahead * 2) - [lower_1, upper_1, lower_2, upper_2, ...]
    points : (B, num_ahead)
"""

from typing import Tuple

import torch
import torch.nn as nn

from model.layers import _collect_predictions, _InputNorm, _OutputHead


class LSTMEncoderDecoder(nn.Module):
    """
    Encoder LSTM  (lstm1): processes lag regressor ->  final (h, c)
    Decoder LSTM  (lstm2): processes future regressor, initialized by encoder (h, c)
    At each decoder step an output head predicts (point, lower, upper).
    """

    def __init__(
        self,
        num_lag_vars: int = 3,
        num_future_vars: int = 3,
        num_ahead: int = 16,
        h_dim: int = 128,
        num_enc_layers: int = 1,
        num_dec_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_ahead = num_ahead
        self.h_dim = h_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        # Input normalisations (raw covariates)
        self.lag_norm = _InputNorm(num_lag_vars)
        self.fut_norm = _InputNorm(num_future_vars)

        # Encoder
        self.encoder = nn.LSTM(
            input_size=num_lag_vars,
            hidden_size=h_dim,
            num_layers=num_enc_layers,
            batch_first=True,
            dropout=dropout if num_enc_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Bridge: adapt encoder layers -> decoder layers if they differ
        if num_enc_layers != num_dec_layers:
            self.h_bridge = nn.Linear(h_dim, h_dim)
            self.c_bridge = nn.Linear(h_dim, h_dim)
        else:
            self.h_bridge = self.c_bridge = None

        # Decoder
        self.decoder = nn.LSTM(
            input_size=num_future_vars,
            hidden_size=h_dim,
            num_layers=num_dec_layers,
            batch_first=True,
            dropout=dropout if num_dec_layers > 1 else 0.0,
        )

        self.head = _OutputHead(h_dim, dropout)

    def _bridge_state(self, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adapt encoder hidden state for decoder when layer counts differ."""
        if self.h_bridge is not None:
            h_last = h[-1]  # (B, h_dim)
            c_last = c[-1]
            h_dec = (
                self.h_bridge(h_last).unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
            )
            c_dec = (
                self.c_bridge(c_last).unsqueeze(0).expand(self.num_dec_layers, -1, -1).contiguous()
            )
            return h_dec, c_dec
        return h, c

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        lag, fut = x

        # (B, C, T), where C = num_lag_vars or num_future_vars
        lag = self.lag_norm(lag).transpose(1, 2)
        fut = self.fut_norm(fut).transpose(1, 2)

        _, (h_enc, c_enc) = self.encoder(lag)  # h: (n_layers, B, h_dim)

        # Bridge
        h_dec, c_dec = self._bridge_state(h_enc, c_enc)

        dec_out, _ = self.decoder(fut, (h_dec, c_dec))  # (B, T_ahead, h_dim)

        points_list, delta_lowers_list, delta_uppers_list = [], [], []
        for t in range(self.num_ahead):
            pt, delta_lo, delta_up = self.head(dec_out[:, t, :])
            points_list.append(pt)
            delta_lowers_list.append(delta_lo)
            delta_uppers_list.append(delta_up)

        return _collect_predictions(points_list, delta_lowers_list, delta_uppers_list)
