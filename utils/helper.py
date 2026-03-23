import os
import pickle
import random
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from sklearn.preprocessing import StandardScaler

FeatureType = Literal["lag", "ahead"]


def get_prefix_list(col_list: List[str], prefix: str, type: FeatureType) -> List[str]:
    prefix = prefix.strip()
    prefix_list = [col for col in col_list if col.startswith(f"{prefix}_{type}")]

    if type == "lag":
        prefix_list = sorted(
            prefix_list,
            key=lambda x: int(x.replace(prefix + "_lag", "")),
            reverse=True,  # time_step => [lag45, lag30, lag15, ...]
        )
        prefix_list.append(prefix)  # include the base feature in past covariate
    else:
        prefix_list = sorted(
            prefix_list,
            key=lambda x: int(x.replace(prefix + "_ahead", "")),
            # time_step => [ahead15, ahead30, ahead45, ...]
        )
    return prefix_list


@logger.catch
def get_scaler(
    train_file_name: os.PathLike, columns: List[str], save_path: os.PathLike, save_name: str
) -> StandardScaler:
    scaler = StandardScaler()
    os.makedirs(save_dir := os.path.join(save_path, "scaler"), exist_ok=True)

    if os.path.exists(os.path.join(save_dir, f"{save_name}_scaler.pkl")):
        logger.info("Existing Cached scaler found, loading...")
        with open(os.path.join(save_dir, f"{save_name}_scaler.pkl"), "rb") as f_scaler:
            scaler = pickle.load(f_scaler)
        return scaler

    logger.info(f"Fitting scaler on CSV file: {train_file_name}...")
    reader = pd.read_csv(train_file_name, chunksize=100_000)
    for df in reader:
        scaler.partial_fit(df[columns].values)

    # Save
    with open(os.path.join(save_dir, f"{save_name}_scaler.pkl"), "wb") as f_scaler:
        pickle.dump(scaler, f_scaler)

    return scaler


def setup():
    # SETUP
    load_dotenv()
    ## Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
