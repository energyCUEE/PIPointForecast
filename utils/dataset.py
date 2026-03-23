import os
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, IterableDataset

from utils.helper import get_prefix_list

FeatureType = Literal["lag", "ahead"]


class CachedPIRawBatchDataset(Dataset):
    """Load pre-cached .pt batches produced by cache_piraw.py.

    Each cached file may contain a large chunk (e.g. 32 768 rows).
    ``batch_size`` controls the actual training mini-batch returned by
    ``__getitem__``.  The dataset transparently sub-divides every cached
    chunk so the DataLoader can iterate over training-sized pieces.

    Returns per index:
        ((lag_regressor, future_regressor), target)
    shapes:
        lag:    (batch_size, C_lag, T_lag)
        future: (batch_size, C_fut, T_fut)
        target: (batch_size, T_ahead)
    """

    def __init__(
        self,
        cache_dir: str,
        batch_size: int = 4096,
        num_lag: int | None = None,
        shuffle: bool = False,
        chronos_future: bool = False,
    ):
        super().__init__()
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.num_lag = num_lag  # None = use all cached lag steps
        self.shuffle = shuffle
        self.chronos_future = chronos_future

        all_files = sorted(os.listdir(cache_dir))
        self.lag_files = [f for f in all_files if f.startswith("lag_")]
        self.future_files = [f for f in all_files if f.startswith("future_")]
        self.target_files = [f for f in all_files if f.startswith("target_")]

        assert len(self.lag_files) == len(self.future_files) == len(self.target_files), (
            f"Mismatch: {len(self.lag_files)} lag, {len(self.future_files)} future, "
            f"{len(self.target_files)} target files in {cache_dir}"
        )

        # Infer num_ahead from first target
        sample = torch.load(os.path.join(cache_dir, self.target_files[0]), weights_only=True)
        self.num_ahead = sample.shape[-1]

        # Build index: list of (file_idx, row_start, row_end) for every mini-batch
        self._index: list[tuple[int, int, int]] = []
        for file_idx in range(len(self.lag_files)):
            target_path = os.path.join(cache_dir, self.target_files[file_idx])
            chunk_len = torch.load(target_path, weights_only=True).shape[0]
            for start in range(0, chunk_len, self.batch_size):
                end = min(start + self.batch_size, chunk_len)
                self._index.append((file_idx, start, end))

        # Cache for the last loaded file to avoid redundant torch.load
        self._cached_file_idx: int | None = None
        self._cached_data: tuple | None = None

    def __len__(self):
        return len(self._index)

    def _load_file(self, file_idx: int):
        """Load a cached chunk from disk (with simple 1-file LRU cache)."""
        if self._cached_file_idx == file_idx:
            return self._cached_data
        lag = torch.load(os.path.join(self.cache_dir, self.lag_files[file_idx]), weights_only=True)
        future = torch.load(
            os.path.join(self.cache_dir, self.future_files[file_idx]), weights_only=True
        )
        target = torch.load(
            os.path.join(self.cache_dir, self.target_files[file_idx]), weights_only=True
        )
        if self.chronos_future:
            chronos_future = torch.load(
                os.path.join(self.cache_dir, f"chronos_future_quantiles_{file_idx:06d}.pt"),
                weights_only=True,
            )
            chronos_future = rearrange(
                chronos_future, "b t q -> b q t"
            )  # q = C_Chronos = 3 for (0.05, 0.5, 0.95) quantiles
            future = torch.cat([future, chronos_future], dim=1)  # (B, C_fut + C_chronos, T_fut)
        self._cached_file_idx = file_idx
        self._cached_data = (lag, future, target)
        return self._cached_data

    def __getitem__(self, index: int):
        file_idx, row_start, row_end = self._index[index]
        lag, future, target = self._load_file(file_idx)
        lag_batch = lag[row_start:row_end]
        if self.num_lag is not None:
            # Keep the most recent num_lag timesteps (columns are ordered furthest → closest)
            lag_batch = lag_batch[:, :, -self.num_lag :]
        return (lag_batch, future[row_start:row_end]), target[row_start:row_end]


class PIRawBatchDataset(IterableDataset):
    def __init__(
        self,
        csv_file_path: str | os.PathLike,
        feature_list: List[str],
        target_scaler: StandardScaler,
        batch_size: int,
        num_lag: int = 192,
        num_ahead: int = 16,
    ):

        self.target_scaler = target_scaler

        self.chunksize = batch_size
        self.csv_file_path = csv_file_path
        self.feature_list = feature_list
        self.target_col_ahead = get_prefix_list(self.feature_list, "I", "ahead")
        self.target_col_lag = get_prefix_list(self.feature_list, "I", "lag")

        self.num_ahead = num_ahead
        self.num_lag = num_lag

        # Pre-compute column lists for extract_input
        past_feature_cols: List[str] = ["CI_R", "CI_CM"]
        future_feature_cols: List[str] = ["Iclr", "Icams", "HI"]

        self.past_covariates_cols = {
            col: get_prefix_list(self.feature_list, col, "lag") for col in past_feature_cols
        }
        self.future_covariates_cols = {
            col: get_prefix_list(self.feature_list, col, "ahead") for col in future_feature_cols
        }

        # Use cached line count if provided to avoid rereading file
        self.len = (self._count_file_lines() + self.chunksize - 1) // self.chunksize

    def _count_file_lines(self) -> int:
        with open(self.csv_file_path, "r") as f:
            row_count = sum(1 for line in f) - 1  # Subtract 1 for header
        return row_count

    def extract_input(self, df):
        # Use pre-computed column lists from __init__
        past_covariates_arrays = [df[cols].values for cols in self.past_covariates_cols.values()]
        future_covariates_arrays = [
            df[cols].values for cols in self.future_covariates_cols.values()
        ]

        target_ahead_arrays = df[self.target_col_ahead].values
        target_lag_arrays = df[self.target_col_lag].values

        # Stack and transpose using numpy only (no torch conversion overhead)
        lag_regressor = np.stack([target_lag_arrays] + past_covariates_arrays)
        future_regressor = np.stack(future_covariates_arrays)

        # Transpose from (c, b, t) to (b, c, t) using numpy
        lag_regressor = np.transpose(lag_regressor, (1, 0, 2))
        future_regressor = np.transpose(future_regressor, (1, 0, 2))

        lag_regressor = lag_regressor[:, :, -self.num_lag :]

        return lag_regressor, future_regressor, self.target_scaler.transform(target_ahead_arrays)

    def __iter__(self):
        with pd.read_csv(self.csv_file_path, chunksize=self.chunksize) as reader:
            for chunk in reader:
                lag_regressor, future_regressor, target_arrays = self.extract_input(chunk)
                yield (lag_regressor, future_regressor), target_arrays

    def __len__(self):
        return self.len
