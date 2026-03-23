import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch


class EvalDFBuilder:
    def __init__(
        self,
        siteidcol="siteID",
        datetimecol="Datetime",
        num_max_workers=40,
        verbose=False,
    ):
        self.siteidcol = siteidcol
        self.datetimecol = datetimecol
        self.num_max_workers = num_max_workers
        self.verbose = verbose

    def process_eval_site(
        self,
        site_id,
        df_infer,
        operation_cols,
        numstepahead=3,
        data_resolution=15,
        verbose=None,
    ):
        if verbose is None:
            verbose = self.verbose

        if verbose == True:
            print(f"Processing for site: {site_id}")

        df_site = df_infer[df_infer[self.siteidcol] == site_id].copy()
        new_cols = {}

        for col_oper in operation_cols:
            for i in range(numstepahead):
                col = f"{col_oper}_ahead{int(data_resolution * (i + 1))}"
                new_col = f"{col_oper}(t|t-{int(data_resolution * (i + 1))})"

                # If source column does not exist (e.g. yhat=None or upper/lower=None), skip
                if col not in df_site.columns:
                    continue

                shifted = (
                    df_site[col]
                    .shift(periods=(i + 1), freq=f"{int(data_resolution)}min")
                    .reindex(df_site.index)
                )
                new_cols[new_col] = shifted

        df_site = pd.concat([df_site, pd.DataFrame(new_cols, index=df_site.index)], axis=1)

        if verbose == True:
            print("Finished concatenating df.")

        return df_site

    def extract_resolution_minute(self, df):
        resolution = df.index.to_series().diff().dropna().mode()[0]

        resolution_minutes = resolution.total_seconds() / 60

        return resolution_minutes

    def extract_num_step_ahead(self, upper, lower, yhat):
        # Allow any of (upper, lower, yhat) to be None
        if (yhat is None) and (upper is None) and (lower is None):
            raise ValueError("At least one of yhat, upper, or lower must be provided.")

        # Determine num_step_ahead from the first available array
        if yhat is not None:
            num_step_ahead = yhat.shape[1] if np.asarray(yhat).ndim == 2 else 1
        elif upper is not None:
            num_step_ahead = upper.shape[1] if np.asarray(upper).ndim == 2 else 1
        else:
            num_step_ahead = lower.shape[1] if np.asarray(lower).ndim == 2 else 1

        return num_step_ahead

    def build_inference_df(self, df, upper=None, lower=None, yhat=None):
        num_step_ahead = self.extract_num_step_ahead(upper, lower, yhat)
        data_resolution = self.extract_resolution_minute(df)

        df_infer = df.copy()

        if upper is not None:
            upper_col = [
                f"upper_ahead{int(data_resolution * (i + 1))}" for i in range(num_step_ahead)
            ]
            df_infer[upper_col] = upper

        if lower is not None:
            lower_col = [
                f"lower_ahead{int(data_resolution * (i + 1))}" for i in range(num_step_ahead)
            ]
            df_infer[lower_col] = lower

        if yhat is not None:
            yhat_col = [
                f"Ihat_ahead{int(data_resolution * (i + 1))}" for i in range(num_step_ahead)
            ]
            df_infer[yhat_col] = yhat

        return df_infer

    def build_evaluation_df(
        self,
        df_infer,
        upper=None,
        lower=None,
        yhat=None,
        num_max_workers=None,
        verbose=None,
    ):
        if num_max_workers is None:
            num_max_workers = self.num_max_workers
        if verbose is None:
            verbose = self.verbose

        # Require the df that contains columns 'siteID', 'skycondition', 'I'

        num_step_ahead = self.extract_num_step_ahead(upper, lower, yhat)
        data_resolution = self.extract_resolution_minute(df_infer)

        site_ids = sorted(df_infer[self.siteidcol].unique())
        operation_cols = ["Ihat", "upper", "lower"]
        df_eval = []

        # --- Parallel processing across sites ---
        with ThreadPoolExecutor(max_workers=num_max_workers) as executor:
            futures = [
                executor.submit(
                    self.process_eval_site,
                    site_id,
                    df_infer,
                    operation_cols,
                    num_step_ahead,
                    data_resolution,
                    verbose=verbose,
                )
                for site_id in site_ids
            ]

            for future in as_completed(futures):
                df_eval.append(future.result())

        # --- Combine all sites ---
        df_eval = pd.concat(df_eval, axis=0)
        df_eval = df_eval.sort_values(by=[self.siteidcol, self.datetimecol]).copy()

        # Keep only columns that exist (because some inputs may be None)
        keep_cols = [self.siteidcol, "skycondition", "I"]

        if upper is not None:
            upper_eval_col = [
                f"upper(t|t-{int(data_resolution * (i + 1))})" for i in range(num_step_ahead)
            ]
            keep_cols += [c for c in upper_eval_col if c in df_eval.columns]

        if lower is not None:
            lower_eval_col = [
                f"lower(t|t-{int(data_resolution * (i + 1))})" for i in range(num_step_ahead)
            ]
            keep_cols += [c for c in lower_eval_col if c in df_eval.columns]

        if yhat is not None:
            yhat_eval_col = [
                f"Ihat(t|t-{int(data_resolution * (i + 1))})" for i in range(num_step_ahead)
            ]
            keep_cols += [c for c in yhat_eval_col if c in df_eval.columns]

        df_eval = df_eval[keep_cols].copy()

        # Return df in evaluation mode
        return df_eval

    def extract_daytime_arrays(self, df_eval, start_time="06:00", end_time="18:00"):
        df_eval_daytime = df_eval.between_time(start_time, end_time)

        ihat_cols = sorted(
            [c for c in df_eval.columns if c.startswith("Ihat(t|")],
            key=lambda x: int(x.split("t-")[-1].rstrip(")")),
        )
        upper_cols = sorted(
            [c for c in df_eval.columns if c.startswith("upper(t|")],
            key=lambda x: int(x.split("t-")[-1].rstrip(")")),
        )
        lower_cols = sorted(
            [c for c in df_eval.columns if c.startswith("lower(t|")],
            key=lambda x: int(x.split("t-")[-1].rstrip(")")),
        )

        yhat_eval_daytime = None
        upper_eval_daytime = None
        lower_eval_daytime = None
        num_step_ahead = None

        if len(ihat_cols) > 0:
            num_step_ahead = len(ihat_cols)
            yhat_eval_daytime = df_eval_daytime[ihat_cols].to_numpy()

        if len(upper_cols) > 0:
            num_step_ahead = len(upper_cols) if num_step_ahead is None else num_step_ahead
            upper_eval_daytime = df_eval_daytime[upper_cols].to_numpy()

        if len(lower_cols) > 0:
            num_step_ahead = len(lower_cols) if num_step_ahead is None else num_step_ahead
            lower_eval_daytime = df_eval_daytime[lower_cols].to_numpy()

        # If none exists -> return None (as you wanted earlier)
        if num_step_ahead is None:
            return None

        y_eval_daytime = pd.concat([df_eval_daytime["I"]] * num_step_ahead, axis=1).to_numpy()

        return y_eval_daytime, yhat_eval_daytime, upper_eval_daytime, lower_eval_daytime


class EvaluationMetrics:
    """Evaluation metrics for regression and prediction intervals."""

    @staticmethod
    def to_tensor(x, dtype=torch.float32, device=None, requires_grad=False):
        """
        Convert input data to a PyTorch tensor.

        Parameters
        ----------
        x : array-like, scalar, or torch.Tensor
            Input data to convert.
        dtype : torch.dtype
            Desired data type (default: torch.float32)
        device : str or torch.device
            Device to place the tensor (CPU or GPU)
        requires_grad : bool
            Whether to track gradients for autograd

        Returns
        -------
        torch.Tensor
            Tensor with specified dtype, device, and gradient tracking.
        """
        if isinstance(x, torch.Tensor):
            tensor = x.clone().detach().to(dtype=dtype, device=device)
            tensor.requires_grad_(requires_grad)
        else:
            tensor = torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
        return tensor

    @staticmethod
    def to_numpy(x):
        """
        Convert input data to a NumPy array.

        Parameters
        ----------
        x : torch.Tensor or array-like
            Input data to convert.

        Returns
        -------
        np.ndarray
            Numpy array representation of input.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def PICP(y, upper, lower):
        """
        Prediction Interval Coverage Probability (PICP).

        Parameters
        ----------
        y : array-like or tensor
            True target values.
        upper : array-like or tensor
            Upper bound of prediction interval.
        lower : array-like or tensor
            Lower bound of prediction interval.

        Returns
        -------
        float or np.ndarray
            Coverage probability. If y is 1D, returns float; if 2D, returns 1D array.
        """
        y, upper, lower = map(EvaluationMetrics.to_numpy, (y, upper, lower))
        mask = (y > lower) & (y < upper)
        picp = np.sum(mask, axis=0) / y.shape[0]
        return float(picp) if y.ndim == 1 else picp

    @staticmethod
    def PINAW(upper, lower, ytarget=None, alongaxis=0):
        """
        Prediction Interval Normalized Average Width (PINAW).

        Parameters
        ----------
        upper : array-like or tensor
            Upper bound of prediction interval.
        lower : array-like or tensor
            Lower bound of prediction interval.
        ytarget : array-like or tensor, optional
            Target values for normalization (95th-5th percentile)
        alongaxis : int
            Axis along which to compute the mean width.

        Returns
        -------
        float or np.ndarray
            Average width (normalized if ytarget provided).
        """
        upper, lower = map(EvaluationMetrics.to_numpy, (upper, lower))
        piaw = np.mean(upper - lower, axis=alongaxis)
        if ytarget is None:
            return piaw
        ytarget = EvaluationMetrics.to_numpy(ytarget)
        y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
        return piaw / y_range

    @staticmethod
    def PINALW(upper, lower, ytarget=None, quantile=0.5, alongaxis=0):
        """
        Prediction Interval Normalized Average Largest Width (PINALW).

        Parameters
        ----------
        upper : array-like or tensor
            Upper bound of prediction interval.
        lower : array-like or tensor
            Lower bound of prediction interval.
        ytarget : array-like or tensor, optional
            Target values for normalization (95th-5th percentile)
        quantile : float
            Quantile of largest widths to average (default 0.5)
        alongaxis : int
            Axis along which to compute largest widths.

        Returns
        -------
        float or np.ndarray
            Average largest width (normalized if ytarget provided).
        """
        upper, lower = map(EvaluationMetrics.to_tensor, (upper, lower))
        widtharray = upper - lower
        k = int(np.floor((1 - quantile) * widtharray.shape[0]))
        pialw = (
            torch.nanmean(torch.topk(widtharray, k, dim=alongaxis, largest=True)[0], dim=alongaxis)
            .detach()
            .cpu()
            .numpy()
        )
        if ytarget is None:
            return pialw
        ytarget = EvaluationMetrics.to_numpy(ytarget)
        y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
        return pialw / y_range

    @staticmethod
    def Winklerscore(y, upper, lower, ytarget=None, delta=0.1, alongaxis=0):
        """
        Winkler Score for prediction intervals.

        Parameters
        ----------
        y : array-like or tensor
            True target values.
        upper : array-like or tensor
            Upper bound of prediction interval.
        lower : array-like or tensor
            Lower bound of prediction interval.
        ytarget : array-like or tensor, optional
            Target values for normalization
        delta : float
            Confidence level parameter for penalization
        alongaxis : int
            Axis along which to compute mean score.

        Returns
        -------
        float or np.ndarray
            Winkler score (normalized if ytarget provided).
        """
        y, upper, lower = map(EvaluationMetrics.to_numpy, (y, upper, lower))
        Winkler_i = np.abs(upper - lower) + (2 / delta) * (
            (lower - y) * (y < lower) + (y - upper) * (y > upper)
        )
        Winkler = np.mean(Winkler_i, axis=alongaxis)
        if ytarget is None:
            return Winkler
        ytarget = EvaluationMetrics.to_numpy(ytarget)
        y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
        return Winkler / y_range

    @staticmethod
    def MSE(y, yhat, ytarget=None, alongaxis=0):
        """
        Mean Squared Error (MSE), optionally normalized by ytarget range.

        Parameters
        ----------
        y : array-like or tensor
            True target values.
        yhat : array-like or tensor
            Predicted values.
        ytarget : array-like or tensor, optional
            Target values for normalization
        alongaxis : int
            Axis along which to compute MSE.

        Returns
        -------
        float or np.ndarray
            MSE or normalized MSE.
        """
        y = EvaluationMetrics.to_numpy(y)
        yhat = EvaluationMetrics.to_numpy(yhat)
        mse = np.mean((y - yhat) ** 2, axis=alongaxis)
        if ytarget is not None:
            ytarget = EvaluationMetrics.to_numpy(ytarget)
            y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            mse /= y_range**2
        return mse

    @staticmethod
    def RMSE(y, yhat, ytarget=None, alongaxis=0):
        """
        Root Mean Squared Error (RMSE), optionally normalized by ytarget range.
        """
        return np.sqrt(EvaluationMetrics.MSE(y, yhat, ytarget, alongaxis))

    @staticmethod
    def MAE(y, yhat, ytarget=None, alongaxis=0):
        """
        Mean Absolute Error (MAE), optionally normalized by ytarget range.
        """
        y = EvaluationMetrics.to_numpy(y)
        yhat = EvaluationMetrics.to_numpy(yhat)
        mae = np.mean(np.abs(y - yhat), axis=alongaxis)
        if ytarget is not None:
            ytarget = EvaluationMetrics.to_numpy(ytarget)
            y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            mae /= y_range
        return mae

    @staticmethod
    def MBE(y, yhat, ytarget=None, alongaxis=0):
        """
        Mean Bias Error (MBE)
        """
        y = EvaluationMetrics.to_numpy(y)
        yhat = EvaluationMetrics.to_numpy(yhat)
        mbe = np.mean(yhat - y, axis=alongaxis)
        if ytarget is not None:
            ytarget = EvaluationMetrics.to_numpy(ytarget)
            y_range = np.quantile(ytarget, 0.95) - np.quantile(ytarget, 0.05)
            mbe /= y_range
        return mbe

    @staticmethod
    def MAPE(y, yhat, alongaxis=0):
        """
        Mean Absolute Percentage Error (MAPE)
        """
        y = EvaluationMetrics.to_numpy(y)
        yhat = EvaluationMetrics.to_numpy(yhat)
        epsilon = 1e-8  # to avoid division by zero
        return np.mean(np.abs((y - yhat) / (y + epsilon)), axis=alongaxis) * 100

    @staticmethod
    def evaluate_all(
        y,
        upper,
        lower,
        yhat=None,
        ytarget=None,
        normalize=False,
        delta=0.1,
        quantile=0.5,
    ):
        """
        Compute all evaluation metrics at once.

        Parameters
        ----------
        y : array-like or tensor
            True target values.
        upper : array-like or tensor
            Upper prediction interval.
        lower : array-like or tensor
            Lower prediction interval.
        yhat : array-like or tensor, optional
            Predicted values for point metrics
        ytarget : array-like or tensor, optional
            Target values for normalization
        normalize : bool
            If True, include nMSE, nRMSE, nMAE normalized by ytarget range.
        delta : float
            Confidence level for Winkler score.
        quantile : float
            Quantile for PINALW computation.

        Returns
        -------
        dict
            Dictionary of computed metrics.
        """
        metrics = dict()

        # --- PI-based metrics (evaluate only if upper and lower are provided) ---
        if upper is not None and lower is not None:
            metrics["PICP"] = EvaluationMetrics.PICP(y, upper, lower)
            metrics["PINAW"] = EvaluationMetrics.PINAW(upper, lower, ytarget)
            metrics["PINALW"] = EvaluationMetrics.PINALW(upper, lower, ytarget, quantile)
            metrics["Winkler"] = EvaluationMetrics.Winklerscore(y, upper, lower, ytarget, delta)
        else:
            metrics["PICP"] = np.nan
            metrics["PINAW"] = np.nan
            metrics["PINALW"] = np.nan
            metrics["Winkler"] = np.nan

        # --- Point-based metrics (evaluate only if yhat provided) ---
        if yhat is not None:
            #             metrics['MSE'] = EvaluationMetrics.MSE(y, yhat)
            metrics["MAE"] = EvaluationMetrics.MAE(y, yhat)
            metrics["RMSE"] = EvaluationMetrics.RMSE(y, yhat)
            metrics["MBE"] = EvaluationMetrics.MBE(y, yhat)
            #             metrics['MAPE'] = EvaluationMetrics.MAPE(y, yhat)
            if normalize:
                #                 metrics['nMSE'] = EvaluationMetrics.MSE(y, yhat, ytarget)
                metrics["nMAE"] = EvaluationMetrics.MAE(y, yhat, ytarget)
                metrics["nRMSE"] = EvaluationMetrics.RMSE(y, yhat, ytarget)
                metrics["nMBE"] = EvaluationMetrics.MBE(y, yhat, ytarget)
        else:
            metrics["MAE"] = np.nan
            metrics["RMSE"] = np.nan
            metrics["MBE"] = np.nan

        return metrics

    @staticmethod
    def report_performance(eval_all, decimal_place=3, axis_mode=0, step_max=None):
        """
        Print evaluation metrics table with aligned columns.

        Parameters
        ----------
        eval_all : dict[str, list or float]
            Dictionary mapping metric names to their step-wise values.
        decimal_place : int, optional
            Number of decimal places to display.
        axis_mode : int, optional
            0 = steps as columns (metrics as rows)
            1 = metrics as columns (steps as rows)
        step_max : int or None, optional
            Limit the number of steps to display (if provided).
        """

        # Detect number of steps
        steps = None
        for v in eval_all.values():
            if hasattr(v, "__iter__") and not isinstance(v, str):
                steps = len(v)
                break
        if steps is None:
            steps = 1

        # --- Apply manual limit if specified ---
        if step_max is not None:
            steps = min(steps, step_max)

        # =========================
        # axis_mode = 1 → steps as rows (metrics as columns)
        # =========================
        if axis_mode == 1:
            metrics = list(eval_all.keys())
            headers = ["Step"] + metrics

            # --- Determine max width per column ---
            col_widths = []
            # Step column width
            step_width = max(len("Step"), len(str(steps)))
            col_widths.append(step_width)

            # Metric column widths (based on all steps)
            for metric in metrics:
                v = eval_all[metric]
                # Convert to list for consistent indexing
                if not (hasattr(v, "__iter__") and not isinstance(v, str)):
                    v = [v] * steps
                formatted_vals = [
                    f"{x:.{decimal_place}f}"
                    if isinstance(x, (int, float)) and not math.isnan(x)
                    else "nan"
                    for x in v[:steps]
                ]
                max_val_len = max(len(s) for s in formatted_vals)
                col_widths.append(max(len(metric), max_val_len))

            # --- Print header ---
            header_row = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
            print("=" * len(header_row))
            print(header_row)
            print("-" * len(header_row))

            # --- Print each step as row ---
            for step in range(steps):
                row = [f"{step + 1}".ljust(col_widths[0])]
                for j, metric in enumerate(metrics):
                    v = eval_all[metric]
                    if hasattr(v, "__iter__") and not isinstance(v, str):
                        val = v[step] if step < len(v) else float("nan")
                    else:
                        val = v
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        val_str = f"{val:.{decimal_place}f}"
                    else:
                        val_str = "nan"
                    row.append(val_str.rjust(col_widths[j + 1]))
                print(" | ".join(row))

            print("=" * len(header_row))

        # =========================
        # axis_mode = 0 → metrics as rows (default)
        # =========================
        else:
            headers = ["Metric"] + [f"{i + 1} step" for i in range(steps)]

            # Compute column widths
            col_widths = [max(len("Metric"), max(len(k) for k in eval_all.keys()))]
            for i in range(steps):
                max_val_len = max(
                    len(f"{v[i]:.{decimal_place}f}")
                    if hasattr(v, "__iter__") and len(v) > i and not math.isnan(v[i])
                    else len("nan")
                    for v in eval_all.values()
                )
                col_widths.append(max(len(headers[i + 1]), max_val_len))

            # Print header
            header_row = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
            print("=" * len(header_row))
            print(header_row)
            print("-" * len(header_row))

            # Print rows
            for metric_name, performance in eval_all.items():
                row = [metric_name.ljust(col_widths[0])]
                if hasattr(performance, "__iter__") and not isinstance(performance, str):
                    for i in range(steps):
                        if i < len(performance) and not math.isnan(performance[i]):
                            val_str = f"{performance[i]:.{decimal_place}f}"
                        else:
                            val_str = "nan"
                        row.append(val_str.rjust(col_widths[i + 1]))
                else:
                    row += [
                        f"{performance:.{decimal_place}f}".rjust(col_widths[i + 1])
                        for i in range(steps)
                    ]
                print(" | ".join(row))

            print("=" * len(header_row))


# Example of usage:
# metrics = EvaluationMetrics()
# picp = metrics.PICP(y, upper, lower)
# pinaw = metrics.PINAW(upper, lower, ytarget = y)
# pinalw = metrics.PINALW(upper, lower, ytarget = y)
# winkler = metrics.Winklerscore(y, upper, lower, ytarget = y)
# mae = metrics.MAE(y, yhat)
# eval_all = metrics.evaluate_all(y, upper, lower, yhat=yhat
#                                  , ytarget=y, normalize=True, delta=0.1, quantile=0.5)
# metrics.report_performance(eval_all)

# Point forecast evaluation
# metrics = EvaluationMetrics()
# mae = metrics.MAE(y, yhat)
# pointmetrics = metrics.evaluate_all(y, None, None, yhat=yhat, ytarget=None, normalize=False):
