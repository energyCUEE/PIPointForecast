"""
Solar Point PI LSTMNet: A multi-step LSTM-based model for point and interval forecasting with covariates.

The model accept:
    x = (lag_regressor, future_regressor)
      lag_regressor  : (B, num_lag_vars,    num_lag)
      future_regressor: (B, num_future_vars, num_ahead)
  and return:
    pi     : (B, num_ahead * 2) - [lower_1, upper_1, lower_2, upper_2, ...]
    points : (B, num_ahead)
"""

import torch
import torch.nn.functional as F
from torch import nn

from model.layers import _InputNorm


class SolarPointPI_LSTMNet(nn.Module):
    ##### Input Tensor Format #####
    # X = [lag_features (num_total_lag_cols), future_features (num_total_future_cols)]
    #
    # Example:
    #   - Lagged regressors: I, CI
    #   - Future regressors: Iclr, Inwp, HI
    #
    # The input tensor should be arranged as:
    # X = [
    #       I(t-L+1), ..., I(t),
    #       CI(t-L+1), ..., CI(t),
    #       Iclr(t+1), Inwp(t+1), HI(t+1),
    #       ...,
    #       Iclr(t+H), Inwp(t+H), HI(t+H)
    #     ]
    # lag_features are used for the LSTM part (temporal dependencies)
    # future_features are split across each prediction step (exogenous regressors)

    def __init__(
        self,
        num_lag_features=3,
        num_future_features=3,
        num_lstm_cells=1,
        lstm_hidden_size=100,
        submodel_hidden_size=[100, 100],
        predicted_step=16,
        batchnorm=False,
    ):
        super(SolarPointPI_LSTMNet, self).__init__()

        # Allow both int and list for hidden size (convert to list if int is given)
        if isinstance(submodel_hidden_size, int):
            submodel_hidden_size = [submodel_hidden_size]

        # Save configuration of the model for reference/logging
        self.config = {
            # 'num_total_lag_cols': num_total_lag_cols,
            # 'num_total_future_cols': num_total_future_cols,
            "num_lag_features": num_lag_features,
            "num_future_features": num_future_features,
            "num_lstm_cells": num_lstm_cells,
            "submodel_hidden_size": submodel_hidden_size,
            "lstm_hidden_size": lstm_hidden_size,
            "predicted_step": predicted_step,
            "batchnorm": batchnorm,
        }

        ##### LSTM PART (shared across all steps) #####
        # self.num_total_lag_cols = num_total_lag_cols   # total lagged input features

        # lag: [I, CI_R, CI_CM]  # num_lag_features = 3
        self.num_lag_features = num_lag_features  # number of lagged variables (auto + exogenous)
        self.lstm_hidden_size = lstm_hidden_size  # hidden units in LSTM
        self.num_lstm_cells = num_lstm_cells  # number of stacked LSTM layers
        self.batchnorm = batchnorm

        # Input normalisations (raw covariates)
        self.lag_norm = _InputNorm(num_lag_features)
        self.fut_norm = _InputNorm(num_future_features)

        # LSTM layer: processes lag features over time
        self.lstm = nn.LSTM(
            input_size=self.num_lag_features,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.num_lstm_cells,
            batch_first=True,  # input/output shape (batch, seq, feature)
        )

        # Batch Normalization applied to LSTM outputs (Identity if batchnorm is False)
        self.bn_lstm = nn.BatchNorm1d(self.lstm_hidden_size) if batchnorm else nn.Identity()

        ##### SUBMODEL PART (step-specific fully connected layers) #####
        # self.num_total_future_cols = num_total_future_cols   # total future regressor inputs

        self.submodel_hidden_size = submodel_hidden_size  # list of hidden units per layer
        self.predicted_step = predicted_step  # horizon steps ahead
        self.num_future_features = num_future_features  # number of future regressors per step

        self.submodel_layers = nn.ModuleList()  # list of hidden layer stacks per step
        self.bn_submodel_layers = nn.ModuleList()  # list of BN layers per step

        # Build a separate submodel for each prediction step
        for _ in range(predicted_step):
            layers = nn.ModuleList()  # hidden layers for this step
            bns = nn.ModuleList()  # batch norms for this step

            # First hidden layer: input size = [LSTM output + future features]
            layers.append(
                nn.Linear(
                    self.lstm_hidden_size + self.num_future_features, self.submodel_hidden_size[0]
                )
            )
            bns.append(nn.BatchNorm1d(self.submodel_hidden_size[0]) if batchnorm else nn.Identity())

            # Remaining hidden layers: chain from list of submodel_hidden_size
            for j in range(1, len(self.submodel_hidden_size)):
                layers.append(
                    nn.Linear(self.submodel_hidden_size[j - 1], self.submodel_hidden_size[j])
                )
                bns.append(
                    nn.BatchNorm1d(self.submodel_hidden_size[j]) if batchnorm else nn.Identity()
                )

            # Add this step’s hidden layers into global list
            self.submodel_layers.append(layers)
            self.bn_submodel_layers.append(bns)

        # Output layers: one per step
        # Each outputs 3 values -> [delta_lower, delta_upper, point forecast]
        self.output_layers = nn.ModuleList(
            [nn.Linear(self.submodel_hidden_size[-1], 3) for _ in range(predicted_step)]
        )

        # Non-linear activation
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tuple of (lag_features, future_features) tensors

        Returns:
            pi: Prediction intervals (N, 2 * predicted_step) -> [lower, upper] pairs
            point: Point forecasts (N, predicted_step)
        """

        ##### SPLIT INPUTS #####
        # common_input = x[:, :self.num_total_lag_cols]   # lag features
        # future_input = x[:, self.num_total_lag_cols:]   # future regressor features
        # batch_size = common_input.size(0)

        lag, fut = x
        lag = self.lag_norm(lag).transpose(1, 2)  # (B, T_lag, C_lag)
        fut = self.fut_norm(fut).transpose(1, 2)  # (B, T_ahead, C_fut)

        ##### LSTM PROCESSING #####
        # Reshape lag features to 3D: (batch, seq_len, num_lag_features)
        # num_each_feature_lag = self.num_total_lag_cols // self.num_lag_features
        # common_input = common_input.view(batch_size, self.num_lag_features, num_each_feature_lag).transpose(1, -1)

        # Pass through LSTM (seq modeling)
        lstm_out, _ = self.lstm(lag)  # (batch, seq_len, hidden_size)

        # Take only last timestep output
        common_input = lstm_out[:, -1, :]  # (batch, hidden_size)

        # BatchNorm + ReLU
        common_input = self.relu(self.bn_lstm(common_input))

        ##### SPLIT FUTURE INPUT PER STEP #####
        # Each sublist corresponds to features for that step
        # future_eachstep = [
        #     future_input[:, i * self.num_future_input:(i + 1) * self.num_future_input]
        #     for i in range(self.predicted_step)
        # ]

        outputs = []

        ##### SUBMODELS (per step) #####
        for i in range(self.predicted_step):
            # Concatenate LSTM output with future features for step i
            step_input = torch.cat((common_input, fut[:, i, :]), dim=1)

            # Pass through hidden layers with BatchNorm + ReLU
            for layer, bn in zip(self.submodel_layers[i], self.bn_submodel_layers[i]):
                step_input = self.relu(bn(layer(step_input)))

            # Output: [delta_l, delta_u, point]
            step_output = self.output_layers[i](step_input)
            outputs.append(step_output)

        ##### FINAL OUTPUT FORMATTING #####
        # Concatenate all steps (N, (delta_l, delta_u, point) * predicted_step)
        final_output = torch.cat(outputs, dim=1)

        # Extract PI deltas and point forecasts
        delta_l = final_output[:, 0::3]  # (batch, predicted_step)
        delta_u = final_output[:, 1::3]  # (batch, predicted_step)
        point = final_output[:, 2::3]  # (batch, predicted_step)

        lower = point - F.softplus(delta_l)
        upper = point + F.softplus(delta_u)

        pi = torch.stack([lower, upper], dim=2).reshape(lower.size(0), -1)

        return pi, point
