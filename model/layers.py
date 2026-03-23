from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class _InputNorm(nn.Module):
    """Per-channel BatchNorm applied on the time dimension of raw inputs.

    Input : (B, C, T)
    Output: (B, C, T)  — each channel independently normalised across B * T.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class _OutputHead(nn.Module):
    """Shared MLP that maps h_dim -> (point, delta_l, delta_u)."""

    def __init__(self, h_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim // 2, 3),  # point, delta_l, delta_u
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(h)  # (B, 3)
        point, delta_l, delta_u = out.unbind(dim=-1)
        return point, delta_l, delta_u


def _collect_predictions(
    points_list, delta_lowers_list, delta_uppers_list
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack per-step lists into (pi, points)."""
    points = torch.stack(points_list, dim=1)  # (B, T)
    delta_lowers = torch.stack(delta_lowers_list, dim=1)
    delta_uppers = torch.stack(delta_uppers_list, dim=1)

    lowers = points - F.softplus(delta_lowers)
    uppers = points + F.softplus(delta_uppers)

    pi = torch.stack([lowers, uppers], dim=-1).flatten(start_dim=1)  # (B, T*2)
    return pi, points
