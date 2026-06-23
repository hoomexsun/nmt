import os
import pandas as pd

# -------------------------------------------------------- FILE CHECKS


def model_exists(exp_dir: str) -> bool:
    config_path = os.path.join(exp_dir, "config.json")
    tokenizer_path = os.path.join(exp_dir, "tokenizer.json")
    model_bin = os.path.join(exp_dir, "pytorch_model.bin")
    model_safe = os.path.join(exp_dir, "model.safetensors")

    return (
        os.path.exists(config_path)
        and os.path.exists(tokenizer_path)
        and (os.path.exists(model_bin) or os.path.exists(model_safe))
    )


def predictions_exist(exp_dir: str) -> bool:
    path = os.path.join(exp_dir, "validation.predictions.csv")
    return os.path.exists(path) and os.path.getsize(path) > 0


def evaluation_exists(exp_dir: str) -> bool:
    path = os.path.join(exp_dir, "scores", "evaluation_results.csv")
    return os.path.exists(path) and os.path.getsize(path) > 0


# -------------------------------------------------------- EXPERIMENT PATHS


def prepare_default_exp_dir(
    base_exp_dir: str, model_folder_name: str, src_lang: str, tgt_lang: str
) -> str:
    dir_name = os.path.join(base_exp_dir, f"{model_folder_name}_{src_lang}_{tgt_lang}")
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


# -------------------------------------------------------- DATA LOADING


def load_and_prepare_df(csv_file: str, reverse: bool = False):
    df = pd.read_csv(
        csv_file,
        sep=",",
        header=0,
        names=["source", "target"],
    )
    df = df[["source", "target"]].dropna()
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df = df[(df["source"] != "") & (df["target"] != "")]

    if reverse:
        df = df.rename(columns={"source": "target", "target": "source"})

    print(
        f"Loaded {len(df)} samples from {csv_file}" + (" (reversed)" if reverse else "")
    )
    print(f"Sample source: {df['source'].iloc[0][:100]}")
    print(f"Sample target: {df['target'].iloc[0][:100]}")
    return df


def load_validation_df(exp_dir: str):
    val_csv = os.path.join(exp_dir, "validation.csv")
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Validation file not found: {val_csv}")

    val_df = pd.read_csv(val_csv, sep=",")
    val_df = val_df[["source", "target"]].dropna()
    val_df["source"] = val_df["source"].astype(str).str.strip()
    val_df["target"] = val_df["target"].astype(str).str.strip()
    val_df = val_df[(val_df["source"] != "") & (val_df["target"] != "")]
    return val_df
