#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import load_all_configs, resolve_path


def load_and_clean_csv(csv_path: str):
    df = pd.read_csv(
        csv_path,
        sep=",",
        header=None,
        names=["source", "target"],
    )
    df = df[["source", "target"]].dropna()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"] != "") & (df["target"] != "")]
    return df


def get_fixed_test_size(base_path: Path) -> int:
    name = base_path.name
    if "nng" in name:
        return 1000
    if "nmf" in name:
        return 800
    raise ValueError(f"Cannot determine fixed test size for file: {base_path}")


def split_and_save(csv_path: str, seed: int):
    base_path = Path(csv_path)
    df = load_and_clean_csv(csv_path)
    n = len(df)
    if n == 0:
        raise ValueError(f"No data found in {csv_path}")

    test_size = get_fixed_test_size(base_path)
    if test_size >= n:
        raise ValueError(
            f"Test size {test_size} >= total rows {n} for {csv_path}. "
            f"Please reduce the fixed test size."
        )

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_df = df.iloc[:test_size].copy()
    remaining_df = df.iloc[test_size:].copy()

    val_fraction = 0.1
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
    )

    split_dir = base_path.with_suffix("")  # e.g. data/engLatn_nngLatn
    os.makedirs(split_dir, exist_ok=True)

    train_path = split_dir / "train.csv"
    val_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"

    train_df.to_csv(train_path, sep=",", index=False)
    val_df.to_csv(val_path, sep=",", index=False)
    test_df.to_csv(test_path, sep=",", index=False)

    print(f"\nSplit {csv_path}:")
    print(f"  Total: {n}")
    print(f"  Test:  {len(test_df)} -> {test_path}")
    print(f"  Val:   {len(val_df)} -> {val_path}")
    print(f"  Train: {len(train_df)} -> {train_path}")


def main():
    _, directions_cfg, _, runtime_cfg = load_all_configs()

    training_cfg = runtime_cfg.get("training", {})
    seed = int(training_cfg.get("seed", 42))

    corpus_files = set()
    for _, direction_info in directions_cfg["directions"].items():
        corpus_file = direction_info.get("corpus_file")
        if not corpus_file:
            continue
        corpus_files.add(resolve_path(corpus_file))

    print("Base corpus files to split:")
    for path in corpus_files:
        print(f"  - {path}")

    for csv_path in corpus_files:
        split_and_save(csv_path, seed)


if __name__ == "__main__":
    main()
