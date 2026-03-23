import numpy as np
import torch
from torch import nn

# This module provides: Loss function classes for prediction interval training.


class PIloss_split:
    def __init__(
        self,
        dev_picp=None,
        desired_picp_night=None,
        threshold=-0.678,
        mul_factor=10,
        delta=0.1,
        soften=50,
        split_piwidth=False,
        piwidth="pinaw",
        k=None,
        lmbda=None,
        return_seploss=False,
    ):
        self.soften = soften
        self.delta = delta
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_seploss = return_seploss
        self.mul_factor = mul_factor
        self.split_piwidth = split_piwidth

        if desired_picp_night is None:
            self.desired_picp_night = 1 - self.delta
        else:
            self.desired_picp_night = desired_picp_night

        # Compute adaptive t based on deviation from PICP
        t_val = self.adaptive_t(dev_picp, mul_factor)
        self.t = torch.tensor(t_val).to(
            self.device
        )  # t can be tensor with the same shape of predicted_step

        # Choose method for PI width
        self.piwidth = piwidth.lower()
        if self.piwidth not in ["pinaw", "sumk"]:
            raise ValueError("piwidth must be either 'pinaw' or 'sumk'")

        # Extra params for 'sumk'
        if self.piwidth == "sumk":
            if k is None or lmbda is None:
                raise ValueError("When piwidth='sumk', both k and lmbda must be provided")
            self.k = k
            self.lmbda = lmbda
        else:
            self.k = None
            self.lmbda = None

    def get_config(self):
        return {
            "mul_factor": self.mul_factor,
            "delta": self.delta,
            "soften": self.soften,
            "piwidth": self.piwidth,
            "k": self.k,
            "lmbda": self.lmbda,
        }

    @staticmethod
    def adaptive_t(dev_picp, mul_factor=10):
        """
        Compute an adaptive scaling factor t based on deviation from PICP.

        Parameters
        ----------
        dev_picp : array-like or None
            Deviation of Prediction Interval Coverage Probability (PICP) from the desired confidence level.
            Could be a NumPy array or None.
        mul_factor : float, default=10
            Multiplication factor that controls the scaling strength.

        Returns
        -------
        t : ndarray
            Adaptive scaling factors with the same shape as dev_picp.
        """

        # Case 1: If input is None or all elements are None → return array of ones
        if dev_picp is None or np.all(dev_picp == None):
            return np.ones_like(dev_picp, dtype=float)

        else:
            # Case 2: Compute adaptive factor
            # - Larger |dev_picp| (big deviation) → smaller factor
            # - Smaller |dev_picp| (close to desired coverage) → larger factor
            # - Prevent division by zero with +1e-3
            # - Cap the factor at 100 to avoid extreme growth
            t = np.minimum(100, mul_factor / (abs(dev_picp) + 1e-3))
            return t

    def update_t(self, dev_picp):
        """Update t dynamically during training."""
        self.dev_picp = dev_picp
        t_val = self.adaptive_t(dev_picp, self.mul_factor)
        self.t = torch.tensor(t_val, dtype=torch.float32, device=self.device)
        return self.t  # <-- return new t for logging if needed

    def log_barrier_function(self, z):
        threshold = -(1 / self.t**2)

        # For each element in z:
        #   if z <= threshold -> logbarrier = -(1/self.t) * log(-z)
        #   else              -> logbarrier = self.t*z - (1/self.t)*log(1/self.t**2) + 1/self.t

        mask = z <= threshold
        log_term = torch.zeros_like(z)
        linear_term = torch.zeros_like(z)

        # ----- Safe log-barrier branch -----
        if mask.any():
            safe_neg_z = (-z[mask]).clamp(min=1e-12)  # avoid log(0)
            log_term[mask] = -(1.0 / self.t[mask]) * torch.log(safe_neg_z)

        # ----- Linear branch -----
        if (~mask).any():
            linear_term[~mask] = (
                self.t[~mask] * z[~mask]
                - (1.0 / self.t[~mask]) * torch.log(1.0 / (self.t[~mask] ** 2))
                + 1.0 / self.t[~mask]
            )

        logbarrier_loss = log_term + linear_term

        return logbarrier_loss

    def smooth_day_night_mask(self, y):
        # approx. 1 when (y < threshold) (night)
        smooth_mask_night = torch.maximum(
            torch.zeros(1).to(self.device), torch.tanh(self.soften * (self.threshold - y))
        )
        # approx. 1 when (y > threshold) (day)
        smooth_mask_day = torch.maximum(
            torch.zeros(1).to(self.device), torch.tanh(self.soften * (y - self.threshold))
        )

        return smooth_mask_night, smooth_mask_day

    def picp_smooth_split(self, y, k_soft):

        smooth_mask_night, smooth_mask_day = self.smooth_day_night_mask(y)

        k_soft_night = smooth_mask_night * k_soft
        k_soft_day = smooth_mask_day * k_soft

        picp_soft_night = (smooth_mask_night * k_soft).sum(dim=0) / (
            smooth_mask_night.sum(dim=0) + 1e-3
        )
        picp_soft_day = (smooth_mask_day * k_soft).sum(dim=0) / (smooth_mask_day.sum(dim=0) + 1e-3)

        return picp_soft_night, picp_soft_day

    def pinaw_function(self, widths, y_range=None):
        N = torch.tensor(widths.shape[0]).to(self.device)
        if y_range is None:
            y_range = torch.tensor(1).to(self.device)

        pinaw = torch.norm(widths, p=1, dim=0) / (N * y_range + 1e-3)

        return pinaw

    def sumk_width_function(self, widths, y_range=None):
        if y_range is None:
            y_range = torch.tensor(1).to(self.device)
        num_k_largest = int(np.floor(self.k * widths.shape[0]))
        num_k_lowest = int(widths.shape[0]) - num_k_largest
        # Prevent the denominator to become zero
        n_k_largest = num_k_largest + 1e-3
        n_k_lowest = num_k_lowest + 1e-3
        sum_k_largest_PIwidth = torch.sum(
            torch.topk(widths.abs(), num_k_largest, dim=0, largest=True)[0], axis=0
        )
        sum_k_lowest_PIwidth = torch.sum(
            torch.topk(widths.abs(), num_k_lowest, dim=0, largest=False)[0], axis=0
        )
        sumk_width = (
            sum_k_largest_PIwidth / n_k_largest + self.lmbda * sum_k_lowest_PIwidth / n_k_lowest
        ) / (y_range + 1e-3)

        return sumk_width

    def piwidth_smooth_split(self, y, k_soft, widths, y_range=None):
        if y_range is None:
            y_range = torch.tensor(1).to(self.device)

        smooth_mask_night, smooth_mask_day = self.smooth_day_night_mask(y)

        widths_night = smooth_mask_night * widths
        widths_day = smooth_mask_day * widths

        pinaw_night = self.pinaw_function(widths_night, y_range)
        sumk_day = self.sumk_width_function(widths_day, y_range)

        return pinaw_night, sumk_day

    def __call__(self, y, pi):
        """
        Parameters
        ----------
        y  : ground truth tensor, shape (N, D)
        pi : prediction intervals, shape (N, 2*D)
             pi[:, 0::2] = lower bounds
             pi[:, 1::2] = upper bounds
        """
        N = torch.tensor(y.shape[0]).to(self.device)
        # Calculate y_range for scaling adjustment
        # quantile requires float32 or double
        y_calc = y.float() if y.dtype in [torch.bfloat16, torch.float16] else y
        y_range = (
            (torch.quantile(y_calc[:, 0], 0.95) - torch.quantile(y_calc[:, 0], 0.05))
            .to(self.device)
            .to(y.dtype)
        )

        # Extract lower, upper bound from pi
        y_pred_lower = pi[:, 0::2].to(self.device)
        y_pred_upper = pi[:, 1::2].to(self.device)
        y = y.to(self.device)

        ## PICP constraint using log-barrier
        # Evaluate smoothing PICP using tanh approximation
        k_soft = (1 / 2) * torch.maximum(
            torch.zeros(1).to(self.device),
            torch.tanh(self.soften * (y - y_pred_lower))
            + torch.tanh(self.soften * (y_pred_upper - y)),
        )

        picp_soft_night, picp_soft_day = self.picp_smooth_split(y, k_soft)

        z_day = (1 - self.delta) - picp_soft_day
        z_night = self.desired_picp_night - picp_soft_night

        logbarrier_loss_night = self.log_barrier_function(z_night)
        logbarrier_loss_day = self.log_barrier_function(z_day)
        logbarrier_loss = logbarrier_loss_night + logbarrier_loss_day

        ## PI width loss
        widths = y_pred_upper - y_pred_lower

        if self.split_piwidth:
            pinaw_night, sumk_day = self.piwidth_smooth_split(y, k_soft, widths, y_range)
            PIwidth = pinaw_night + sumk_day

        else:
            # In case of PINAW
            if self.piwidth == "pinaw":
                pinaw = self.pinaw_function(widths, y_range)
                PIwidth = pinaw
            # In case of Sum-k width
            elif self.piwidth == "sumk":
                sumk_width = self.sumk_width_function(widths, y_range)
                PIwidth = sumk_width

        # Aggregate the PI loss = PIwidth + logbarrier_loss
        pi_loss = PIwidth + logbarrier_loss

        # Return every loss, pi_loss (total loss), PI width, picp_loss
        if self.return_seploss:
            return pi_loss, PIwidth, logbarrier_loss

        # Return PI loss in 1D array (predicted_step, )
        else:
            return pi_loss


class Regressionloss:
    def __init__(self):
        # Automatically choose GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_config(self):
        return {}

    def __call__(self, y, yhat):
        # Number of samples in the batch
        N = yhat.shape[0]

        # Move tensors to the same device (GPU/CPU)
        y = y.to(self.device)
        yhat = yhat.to(self.device)

        # Compute the range of target values using the 95th and 5th quantiles
        # This is more robust to outliers than using max-min
        # Cast to float32 for quantile calculation as it doesn't support bfloat16
        y_float = y.float()
        y_range = torch.quantile(y_float[:, 0], 0.95) - torch.quantile(y_float[:, 0], 0.05)
        y_range = y_range.to(y.dtype)

        # Compute L1 loss (sum of absolute errors) along the batch dimension
        # Result shape: (predicted_step,)

        l1_loss = torch.norm(yhat - y, p=1, dim=0)

        # Normalize the loss by both number of samples and target range
        # Add small epsilon (0.001) to prevent division by zero
        reg_loss = l1_loss / (N * y_range + 1e-8)

        # Return regression loss in 1D array with shape (predicted_step,)
        return reg_loss


class QDplusloss(nn.Module):
    def __init__(self, delta=0.1, lambda_1=0.995, lambda_2=0.2, ksi=10, soften=100):
        """
        Quantile-based Dual (QD+) loss function for joint point + PI estimation.

        Parameters
        ----------
        delta : float
            Target miscoverage rate (1 - desired PICP).
        lambda_1, lambda_2 : float
            Balancing weights for PICP, PI-width, and MSE.
        ksi : float
            Penalty term coefficient for out-of-bound predictions.
        soften : float
            Softening factor for sigmoid approximation (higher = sharper boundary).
        """
        super(QDplusloss, self).__init__()
        self.soften = soften
        self.delta = delta
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.ksi = ksi
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_config(self):
        """Return configuration dictionary for checkpoint saving."""
        return {
            "delta": self.delta,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "ksi": self.ksi,
            "soften": self.soften,
        }

    def loss_picp(self, y, y_pred_lower, y_pred_upper):
        """Soft PICP (Prediction Interval Coverage Probability) loss."""
        # Soft coverage indicators using sigmoid approximation
        k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
        k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
        k_soft = k_soft_u * k_soft_l

        # Soft coverage probability
        picp_soft = torch.mean(k_soft, dim=0)

        # Coverage loss (penalize if below target coverage)
        loss_picp_ = torch.square(
            torch.max(torch.zeros(1, device=self.device), (1 - self.delta) - picp_soft)
        )
        return loss_picp_

    def loss_piaw_capt(self, y, y_pred_lower, y_pred_upper):
        """PI width (PINAW) for covered samples only (non-normalized)."""
        # Hard coverage indicators (binary inside/outside)
        k_hard_u = torch.max(torch.sign(y_pred_upper - y), torch.zeros_like(y, device=self.device))
        k_hard_l = torch.max(torch.sign(y - y_pred_lower), torch.zeros_like(y, device=self.device))
        k_hard = k_hard_u * k_hard_l

        # Count covered samples
        c_hard = torch.sum(k_hard)

        # Average width among covered samples
        loss_piaw_capt = torch.sum((y_pred_upper - y_pred_lower) * k_hard) / (c_hard + 1e-3)
        return loss_piaw_capt

    def loss_mse(self, y, yhat):
        """Mean squared error (MSE)."""
        return torch.mean((yhat - y) ** 2, dim=0)

    def penalty_function(self, yhat, y_pred_lower, y_pred_upper):
        """Penalty for predictions outside the interval."""
        m_l = torch.relu(y_pred_lower - yhat)  # penalty if lower > point
        m_u = torch.relu(yhat - y_pred_upper)  # penalty if upper < point
        return torch.mean(m_l + m_u, dim=0)

    def forward(self, y, pi, yhat):
        """
        Compute total QD+ loss.

        Parameters
        ----------
        y : tensor
            Ground truth values, shape (N, D)
        pi : tensor
            Prediction intervals, shape (N, 2*D)
            - Lower bound = pi[:, 0::2]
            - Upper bound = pi[:, 1::2]
        yhat : tensor
            Point predictions, shape (N, D)
        """
        y, pi, yhat = y.to(self.device), pi.to(self.device), yhat.to(self.device)

        # Extract lower and upper bounds
        y_pred_lower = pi[:, 0::2]
        y_pred_upper = pi[:, 1::2]

        # --- Compute individual losses ---
        loss_picp_ = self.loss_picp(y, y_pred_lower, y_pred_upper)
        loss_piaw_capt_ = self.loss_piaw_capt(y, y_pred_lower, y_pred_upper)
        loss_mse_ = self.loss_mse(y, yhat)
        loss_penalty = self.penalty_function(yhat, y_pred_lower, y_pred_upper)

        # --- Combine into total QD+ loss ---
        loss_qdp = (
            (1 - self.lambda_1) * (1 - self.lambda_2) * loss_piaw_capt_
            + self.lambda_1 * (1 - self.lambda_2) * loss_picp_
            + self.lambda_2 * loss_mse_
            + self.ksi * loss_penalty
        )

        return loss_qdp


class IPIVloss(nn.Module):
    def __init__(self, lmbda=15, delta=0.1, beta=0.5, soften=100):
        """
        Parameters
        ----------
        lmbda : float
            Scaling factor for the PICP regularization term.
        delta : float
            Target miscoverage rate (1 - desired PICP).
        beta : float
            Weight balancing PI-related loss and point regression loss.
        soften : float
            Softness of sigmoid in coverage approximation (larger -> sharper boundary).
        """
        super(IPIVloss, self).__init__()
        self.lmbda = lmbda
        self.delta = delta
        self.beta = beta
        self.soften = soften
        # Automatically choose CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_config(self):
        """Return loss configuration for checkpoint logging."""
        return {"lmbda": self.lmbda, "delta": self.delta, "beta": self.beta, "soften": self.soften}

    def loss_picp(self, y, y_pred_lower, y_pred_upper):
        """Soft PICP (Prediction Interval Coverage Probability) loss."""
        # Soft coverage indicators using sigmoid approximation
        k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
        k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
        k_soft = k_soft_u * k_soft_l

        # Soft coverage probability
        picp_soft = torch.mean(k_soft, dim=0)

        # Coverage loss (penalize if below target coverage)
        loss_picp_ = torch.square(
            torch.max(torch.zeros(1, device=self.device), (1 - self.delta) - picp_soft)
        )
        return loss_picp_

    def loss_piaw_capt(self, y, y_pred_lower, y_pred_upper):
        """PI width (PINAW) for covered samples only (non-normalized)."""
        # Hard coverage indicators (binary inside/outside)
        k_hard_u = torch.max(torch.sign(y_pred_upper - y), torch.zeros_like(y, device=self.device))
        k_hard_l = torch.max(torch.sign(y - y_pred_lower), torch.zeros_like(y, device=self.device))
        k_hard = k_hard_u * k_hard_l

        # Count covered samples
        c_hard = torch.sum(k_hard)

        # Average width among covered samples
        loss_piaw_capt = torch.sum((y_pred_upper - y_pred_lower) * k_hard) / (c_hard + 1e-3)
        return loss_piaw_capt

    def loss_mse(self, y, yhat):
        """Mean squared error (MSE)."""
        return torch.mean((yhat - y) ** 2, dim=0)

    def forward(self, y, pi, yhat):
        """
        Compute IPIV loss combining interval and point prediction terms.

        Parameters
        ----------
        y  : ground truth tensor, shape (N, D)
        pi : prediction intervals, shape (N, 2*D)
             pi[:, 0::2] = lower bounds
             pi[:, 1::2] = upper bounds
        yhat : tensor
             Point predictions, shape (N, D)
        """
        N = torch.tensor(y.shape[0]).to(self.device)
        y, pi, yhat = y.to(self.device), pi.to(self.device), yhat.to(self.device)

        # Extract lower and upper bounds
        y_pred_lower = pi[:, 0::2]
        y_pred_upper = pi[:, 1::2]

        # --- Compute individual losses ---
        loss_picp_ = self.loss_picp(y, y_pred_lower, y_pred_upper)
        loss_piaw_capt_ = self.loss_piaw_capt(y, y_pred_lower, y_pred_upper)
        loss_pi = loss_piaw_capt_ + torch.sqrt(N) * self.lmbda * loss_picp_

        loss_mse_ = self.loss_mse(y, yhat)

        loss_ipiv = self.beta * loss_pi + (1 - self.beta) * loss_mse_

        return loss_ipiv


class PIPointQuantileloss(nn.Module):
    def __init__(self, delta=0.1, quantile_point=0.5):
        """
        Quantile PI + Point Loss (Pinball Loss version)
        ------------------------------------------------
        Used for models that output:
            - lower quantile (e.g., 5%)
            - median / point forecast (e.g., 50%)
            - upper quantile (e.g., 95%)

        This loss combines pinball losses for all three components:
        lower bound, point prediction, and upper bound.

        Parameters
        ----------
        delta : float
            Total miscoverage probability (1 - desired_PICP).
            Example: delta=0.1 → lower=0.05, upper=0.95.
        quantile_point : float
            Quantile for the point forecast (default: median, 0.5).
        """
        super(PIPointQuantileloss, self).__init__()

        self.delta = delta
        self.quantile_point = quantile_point

        # Define lower/upper quantile levels from delta
        self.quantile_lower = self.delta / 2
        self.quantile_upper = 1 - self.quantile_lower

        # Automatically select device (CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_config(self):
        """Return configuration dictionary for checkpoint saving."""
        return {
            "delta": self.delta,
            "quantile_lower": self.quantile_lower,
            "quantile_point": self.quantile_point,
            "quantile_upper": self.quantile_upper,
        }

    def forward(self, y, pi, yhat):
        """
        Compute combined Quantile Regression (Pinball) loss for
        lower, upper, and point predictions.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth tensor, shape (N, D)
        pi : torch.Tensor
            Prediction interval tensor, shape (N, 2*D)
            - lower bound = pi[:, 0::2]
            - upper bound = pi[:, 1::2]
        yhat : torch.Tensor
            Point prediction tensor, shape (N, D)

        Returns
        -------
        pinball : torch.Tensor
            1D tensor of per-step mean pinball losses, shape (D,)
        """
        # Ensure input tensors are on the same device (CUDA or CPU)
        y, pi, yhat = y.to(self.device), pi.to(self.device), yhat.to(self.device)

        # Extract lower and upper bounds
        y_pred_lower = pi[:, 0::2]
        y_pred_upper = pi[:, 1::2]

        # Predefine a zero tensor for broadcasting in pinball computation
        zeros = torch.zeros_like(y).to(self.device)

        # --------------------------
        # Compute pinball losses
        # --------------------------

        # Lower quantile pinball loss (e.g., q=0.05)
        pinball_lower = self.quantile_lower * torch.maximum(zeros, y - y_pred_lower) + (
            1 - self.quantile_lower
        ) * torch.maximum(zeros, -(y - y_pred_lower))

        # Upper quantile pinball loss (e.g., q=0.95)
        pinball_upper = self.quantile_upper * torch.maximum(zeros, y - y_pred_upper) + (
            1 - self.quantile_upper
        ) * torch.maximum(zeros, -(y - y_pred_upper))

        # Point forecast pinball loss (e.g., q=0.5)
        pinball_point = self.quantile_point * torch.maximum(zeros, y - yhat) + (
            1 - self.quantile_point
        ) * torch.maximum(zeros, -(y - yhat))

        # Total loss per element
        pinball_i = pinball_lower + pinball_point + pinball_upper

        # Total loss per element
        pinball = torch.mean(pinball_i, dim=0)

        # Output: 1D tensor of shape (num_steps,)
        return pinball


class EMQloss(nn.Module):
    def __init__(self, delta=0.1, lmbda=0.1, quantile_range=[-0.01, 0.0, 0.01], soften=100):
        """
        Enhanced Multi-Quantile (EMQ) Loss Function
        -------------------------------------------
        Used for simultaneous point forecast and prediction interval (PI) estimation.
        This loss combines multiple objectives:
            - Quantile regression (pinball) loss for lower, upper, and point predictions
            - Calibration consistency to penalize out-of-PI samples
            - Scaled PI width regularization (adaptive interval widening)
            - Non-crossing constraint between bounds and mean predictions

        Parameters
        ----------
        delta : float, default=0.1
            Significance level for the prediction interval (PI).
            The desired coverage probability is (1 - delta).

        lmbda : float, default=0.995
            Weight coefficient for the non-crossing penalty term.
            Higher values emphasize monotonic interval behavior.

        quantile_range : list of float, default=[-0.01, 0.0, 0.01]
            Small perturbations added to the lower and upper quantile levels.
            Used to stabilize quantile regression and improve calibration.

        soften : int, default=100
            Smoothing factor for the sigmoid approximation in soft coverage computation.
            Larger values approximate a hard indicator function.

        Attributes
        ----------
        quantile_point : float
            The median quantile (0.5) used for point forecasts.

        quantile_lower : float
            Lower quantile level computed as delta / 2.

        quantile_upper : float
            Upper quantile level computed as 1 - delta / 2.

        device : torch.device
            Automatically detects and assigns computation to CUDA (GPU) if available, otherwise CPU.
        """

        super(EMQloss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.delta = delta
        self.soften = soften
        self.lmbda = lmbda
        self.quantile_range = torch.tensor(quantile_range).to(self.device)

        self.quantile_point = 0.5
        # Define lower/upper quantile levels from delta
        self.quantile_lower = self.delta / 2
        self.quantile_upper = 1 - self.quantile_lower

    def get_config(self):
        """Return configuration dictionary for checkpoint saving."""
        return {
            "delta": self.delta,
            "lmbda": self.lmbda,
            "quantile_range": self.quantile_range.tolist(),
            "soften": self.soften,
        }

    # ============================================================
    #                   Sub-loss components
    # ============================================================

    def loss_quantiles(self, y, yhat, y_pred_lower, y_pred_upper):
        """
        Compute quantile (pinball) losses for lower, upper, and point forecasts.
        Supports small quantile perturbations via quantile_range.
        """
        # Ensure inputs are on the same device
        y, yhat, y_pred_lower, y_pred_upper = (
            y.to(self.device),
            yhat.to(self.device),
            y_pred_lower.to(self.device),
            y_pred_upper.to(self.device),
        )

        zeros = torch.zeros_like(y).to(self.device)
        # ----- Pinball loss for point forecast (q = 0.5) -----
        pinball_point = self.quantile_point * torch.maximum(zeros, y - yhat) + (
            1 - self.quantile_point
        ) * torch.maximum(zeros, -(y - yhat))

        # Expand quantile offsets
        quantile_range_lower = (self.quantile_lower + self.quantile_range).view(1, 1, -1)
        quantile_range_upper = (self.quantile_upper + self.quantile_range).view(1, 1, -1)

        # Expand dimensions for broadcasting
        y = y.unsqueeze(-1)
        y_pred_lower = y_pred_lower.unsqueeze(-1)
        y_pred_upper = y_pred_upper.unsqueeze(-1)
        zeros = torch.zeros_like(y).to(self.device)

        # ----- Lower quantile pinball -----

        pinball_lower = torch.mean(
            quantile_range_lower * torch.maximum(zeros, y - y_pred_lower)
            + (1 - quantile_range_lower) * torch.maximum(zeros, -(y - y_pred_lower)),
            dim=-1,
        )

        # ----- Upper quantile pinball -----
        pinball_upper = torch.mean(
            quantile_range_upper * torch.maximum(zeros, y - y_pred_upper)
            + (1 - quantile_range_upper) * torch.maximum(zeros, -(y - y_pred_upper)),
            dim=-1,
        )

        # Combine all three pinball terms
        pinball_i = pinball_lower + pinball_point + pinball_upper
        pinball = torch.mean(pinball_i, dim=0)  # average over samples

        return pinball

    def loss_calibration(self, y, y_pred_lower, y_pred_upper):
        """
        Calibration loss:
        Penalizes samples outside the PI based on deviation from boundaries.
        """
        y, y_pred_lower, y_pred_upper = (
            y.to(self.device),
            y_pred_lower.to(self.device),
            y_pred_upper.to(self.device),
        )
        lc = ((y_pred_lower - y) * (y < y_pred_lower) + (y - y_pred_upper) * (y > y_pred_upper)) / (
            self.delta / 2
        )

        return torch.mean(lc, dim=0)

    def loss_piaw_scaled(self, y, y_pred_lower, y_pred_upper):
        """
        Scaled PI width (PINAW) term.
        Adjusts penalty based on soft coverage.
        """
        y, y_pred_lower, y_pred_upper = (
            y.to(self.device),
            y_pred_lower.to(self.device),
            y_pred_upper.to(self.device),
        )

        # Soft coverage indicators using sigmoid approximation
        k_soft_u = torch.sigmoid(self.soften * (y_pred_upper - y))
        k_soft_l = torch.sigmoid(self.soften * (y - y_pred_lower))
        k_soft = k_soft_u * k_soft_l

        # Soft coverage probability
        picp_soft = torch.mean(k_soft, dim=0)

        desired_picp = 1 - self.delta
        scaling_factor = 1 + (picp_soft < desired_picp) * (
            (desired_picp - picp_soft) / desired_picp
        )

        loss_piaw_scaled = torch.mean(scaling_factor * (y_pred_upper - y_pred_lower), dim=0)

        return loss_piaw_scaled

    def loss_noncrossing(self, yhat, y_pred_lower, y_pred_upper):
        """
        Non-crossing penalty:
        Prevents lower > yhat or yhat > upper by applying exponential penalty.
        """
        yhat, y_pred_lower, y_pred_upper = (
            yhat.to(self.device),
            y_pred_lower.to(self.device),
            y_pred_upper.to(self.device),
        )
        lnc = torch.exp(torch.relu(y_pred_lower - yhat)) + torch.exp(
            torch.relu(yhat - y_pred_upper)
        )

        return torch.mean(lnc, dim=0)

    # ============================================================
    #                     Forward pass
    # ============================================================

    def forward(self, y, pi, yhat):
        """
        Compute the total EMQ loss.

        Parameters
        ----------
        y : tensor, shape (N, D)
            Ground truth values.
        pi : tensor, shape (N, 2*D)
            Prediction interval tensor:
                - Lower bound = pi[:, 0::2]
                - Upper bound = pi[:, 1::2]
        yhat : tensor, shape (N, D)
            Point forecast predictions.
        """
        y, pi, yhat = y.to(self.device), pi.to(self.device), yhat.to(self.device)

        # Extract lower and upper bounds
        y_pred_lower = pi[:, 0::2]
        y_pred_upper = pi[:, 1::2]

        # Compute sub-losses
        loss_piaw_scaled_ = self.loss_piaw_scaled(y, y_pred_lower, y_pred_upper)
        loss_calibration = self.loss_calibration(y, y_pred_lower, y_pred_upper)
        loss_noncrossing = self.loss_noncrossing(yhat, y_pred_lower, y_pred_upper)
        loss_quantiles = self.loss_quantiles(y, yhat, y_pred_lower, y_pred_upper)

        # Combine all losses into total EMQ loss
        loss_emq = (
            loss_quantiles + self.lmbda * loss_noncrossing + loss_calibration + loss_piaw_scaled_
        )

        return loss_emq


class QuantileRegressionloss(nn.Module):
    def __init__(self, quantiles=[0.05, 0.5, 0.95]):
        """
        Quantile Regression (Pinball) Loss Function
        -------------------------------------------
        Used to train models that predict multiple quantile levels (e.g., 5%, 50%, 95%)
        for uncertainty-aware regression.

        Parameters
        ----------
        quantiles : list or float
            List of quantile levels (e.g., [0.05, 0.5, 0.95]) or a single float.
            Determines which quantiles the model should predict.
        """
        super(QuantileRegressionloss, self).__init__()

        # Automatically select device (CUDA if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure quantiles are always in list form
        if isinstance(quantiles, (int, float)):
            quantiles = [quantiles]

        # Convert quantile list to tensor and move it to the target device
        self.quantiles = torch.as_tensor(quantiles, dtype=torch.float32, device=self.device)
        self.num_quantiles = len(quantiles)

    def get_config(self):
        """Return configuration dictionary for checkpoint saving."""
        return {"quantiles": self.quantiles}

    def forward(self, y, y_pred):
        """
        Compute the quantile regression (pinball) loss.

        Parameters
        ----------
        y : torch.Tensor
            Ground truth tensor of shape (N, num_steps)
        y_pred : torch.Tensor
            Predicted quantiles tensor of shape (N, num_steps * num_quantiles)

        Returns
        -------
        pinball : torch.Tensor
            Mean quantile regression loss averaged over quantiles and steps.
        """
        # Ensure input tensors are on the same device (CUDA or CPU)
        y = y.to(self.device)
        y_pred = y_pred.to(self.device)
        self.quantiles = self.quantiles.to(self.device)  # Re-sync in case of device mismatch

        # Reshape y_pred: (N, num_steps * num_quantiles) → (N, num_quantiles, num_steps)
        y_pred = y_pred.view(
            y_pred.shape[0], y_pred.shape[1] // self.num_quantiles, self.num_quantiles
        )
        y_pred = y_pred.transpose(1, 2)
        # Shape now: (N, num_quantiles, num_steps)
        # Example: if N=64, num_steps=10, num_quantiles=3 → (64, 3, 10)

        # Expand y to match y_pred dimensions for broadcasting
        # y: (N, num_steps) → (N, num_quantiles, num_steps)
        y_expand = y.detach().unsqueeze(1).repeat(1, self.num_quantiles, 1)

        # Compute error (residual)
        error = y_expand - y_pred

        # Expand quantile tensor for broadcasting: (num_quantiles,) → (1, num_quantiles, 1)
        quantiles_expand = self.quantiles.view(1, -1, 1)

        # Compute pinball loss:
        # For positive errors → q * error
        # For negative errors → (q - 1) * error
        pinball_i = torch.maximum(quantiles_expand * error, (quantiles_expand - 1) * error)

        # Average over samples N, and quantiles
        pinball = torch.mean(pinball_i, dim=(0, 1))

        # Output: 1D tensor of shape (num_steps,)
        return pinball
