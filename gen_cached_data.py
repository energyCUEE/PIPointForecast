"""
Cache PIRawBatchDataset to disk as .pt tensor files for fast loading.

Usage:
    uv run gen_cached_data.py [--data-path DATA_PKL] [--output-dir DIR] [--batch-size N]

This iterates through the CSV via PIRawBatchDataset and saves each batch as
    lag_{idx:06d}.pt      shape (B, C_lag, T_lag)
    future_{idx:06d}.pt   shape (B, C_fut, T_fut)
    target_{idx:06d}.pt   shape (B, T_ahead)
"""

import argparse
import os
import pickle

import torch
from chronos import (
    BaseChronosPipeline,
    Chronos2Pipeline,
)
from tqdm import tqdm

from utils.dataset import PIRawBatchDataset
from utils.helper import get_scaler, setup

setup()

DEFAULT_DATA_PATH = "data/sample_data_paths_16_numlags_192_df_file.pkl"
DEFAULT_OUTPUT_DIR = "data/raw"
DEFAULT_BATCH_SIZE = 16


def load_data(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def cache_split(
    dataset: PIRawBatchDataset,
    output_dir: str | os.PathLike,
):
    """Iterate through dataset and save each batch as .pt files."""
    os.makedirs(output_dir, exist_ok=True)

    model_name = "amazon/chronos-2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map=device)

    def chronos_predict(inputs, batch_size=256, cross_learning=False):
        quantiles, point = pipeline.predict_quantiles(
            inputs=inputs,
            quantile_levels=[0.05, 0.5, 0.95],
            prediction_length=16,
            batch_size=batch_size,
            cross_learning=cross_learning,
        )
        return quantiles, point

    for idx, ((lag, future), target) in enumerate(
        tqdm(dataset, total=len(dataset), desc=f"Caching to {output_dir}")
    ):
        lag_t = torch.from_numpy(lag).float()  # (B, C_lag, T_lag)
        future_t = torch.from_numpy(future).float()  # (B, C_fut, T_fut)
        target_t = torch.from_numpy(target).float()  # (B, T_fut)

        q, _ = chronos_predict(lag_t, batch_size=512, cross_learning=False)
        # C_lag = ["I", "CI_R", "CI_CM"]
        # q list (len == batch) of tensors (C_lag, T_lag, 3)  # 3 quantiles
        # p list (len == batch) of tensors (C_lag, T_lag)     # point prediction

        q = torch.stack(q)  # (B, C_lag, T_lag, 3)
        q = q[:, 0, ...].clone()  # (B, T_lag, 3)  # take only the "I" lower, point, upper quantiles

        torch.save(lag_t, os.path.join(output_dir, f"lag_{idx:06d}.pt"))
        torch.save(future_t, os.path.join(output_dir, f"future_{idx:06d}.pt"))
        torch.save(q, os.path.join(output_dir, f"chronos_future_quantiles_{idx:06d}.pt"))
        torch.save(target_t, os.path.join(output_dir, f"target_{idx:06d}.pt"))

    print(f"  Saved {idx + 1} batches to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Cache PIRawBatchDataset to .pt files")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    data = load_data(args.data_path)

    num_lag = data.get("num_lag", 192)
    num_step_ahead = data.get("num_step_ahead", 16)
    target_col_name = data.get("target_col", "I")

    y_scaler = get_scaler(
        data["df_train_nonan"],
        target_col_name,
        save_path="./save_model",
        save_name="target",
    )

    batch_dir = f"b{args.batch_size}"

    for split in ["train", "val", "test"]:
        csv_path = data[f"df_{split}_nonan"]
        print(f"\n=== Caching {split} split  (csv: {csv_path}) ===")

        ds = PIRawBatchDataset(
            csv_file_path=csv_path,
            feature_list=data["features_list"],
            target_scaler=y_scaler,
            batch_size=args.batch_size,
            num_lag=num_lag,
            num_ahead=num_step_ahead,
        )

        out_dir = os.path.join(args.output_dir, split, batch_dir)
        cache_split(ds, out_dir)

    print("\nDone! Cached tensors are in:", args.output_dir)


if __name__ == "__main__":
    main()
