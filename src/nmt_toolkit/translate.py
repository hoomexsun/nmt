import csv
import os

import pandas as pd
import torch
from tqdm.auto import tqdm

from .loader import load_for_inference
from .utils import (
    load_validation_df,
    prepare_default_exp_dir,
    load_and_prepare_df,
)
from pathlib import Path

# -------------------------------------------------------- TRANSLATION UTILS


def get_resume_count(pred_file: str) -> int:
    if not os.path.exists(pred_file):
        return 0
    try:
        existing_df = pd.read_csv(pred_file, sep="\t")
        return len(existing_df)
    except Exception as e:
        print(f"[WARNING] Could not read existing prediction file {pred_file}: {e}")
        return 0


def append_prediction_rows(pred_file: str, rows: list, write_header: bool):
    mode = "w" if write_header else "a"
    with open(pred_file, mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        if write_header:
            writer.writerow(["src", "pred", "ref"])
        writer.writerows(rows)


def save_flat_files(translations_dir: str, prefix: str, srcs, preds, refs):
    """
    Save plain-text files for inspection.
    prefix: 'val' or 'test'
    """
    src_path = os.path.join(translations_dir, f"{prefix}.src.txt")
    ref_path = os.path.join(translations_dir, f"{prefix}.ref.txt")
    pred_path = os.path.join(translations_dir, f"{prefix}.pred.txt")

    with open(src_path, "w", encoding="utf-8") as f_src, open(
        ref_path, "w", encoding="utf-8"
    ) as f_ref, open(pred_path, "w", encoding="utf-8") as f_pred:
        for s, r, p in zip(srcs, refs, preds):
            f_src.write(str(s).strip() + "\n")
            f_ref.write(str(r).strip() + "\n")
            f_pred.write(str(p).strip() + "\n")

    print(f"Saved: {src_path}")
    print(f"Saved: {ref_path}")
    print(f"Saved: {pred_path}")


def load_test_df(cfg: dict):
    """
    Load test split from precomputed data directory, apply reverse at direction level.
    We reuse load_and_prepare_df and cfg['csv_file'].
    """
    base_csv = Path(cfg["corpus_file"])
    split_dir = base_csv.with_suffix("")  # e.g. data/engLatn_nngLatn
    test_path = split_dir / "test.csv"

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split not found: {test_path}. "
            "Run 'python -m src.nmt_toolkit.split_corpus' first."
        )

    reverse = cfg.get("reverse", False)
    return load_and_prepare_df(str(test_path), reverse=reverse)


# -------------------------------------------------------- TRANSLATION


def translate_split(cfg: dict, mode: str = "validation"):
    """
    Translate either 'validation' or 'test' split.
    - mode='validation': uses exp_dir/validation.csv (copy of pre-split)
    - mode='test': uses data/.../test.csv via load_test_df
    """
    if mode not in {"validation", "test"}:
        raise ValueError(f"Unsupported translate mode: {mode}")

    folder_prefix = cfg["folder_prefix"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    infer_cfg = cfg["inference"]

    exp_dir = prepare_default_exp_dir(
        cfg["base_exp_dir"], folder_prefix, src_lang, tgt_lang
    )
    print(f"exp_dir={exp_dir}")

    translations_dir = os.path.join(exp_dir, "translations")
    os.makedirs(translations_dir, exist_ok=True)

    if mode == "validation":
        pred_file = os.path.join(translations_dir, "validation.predictions.csv")
    else:
        pred_file = os.path.join(translations_dir, "test.predictions.csv")

    if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
        print(f"[SKIP] Predictions already exist: {pred_file}")
        return exp_dir

    model, tokenizer = load_for_inference(exp_dir, cfg)

    if mode == "validation":
        val_df = load_validation_df(exp_dir)
        sources = val_df["source"].tolist()
        references = val_df["target"].tolist()
        prefix = "val"
    else:
        test_df = load_test_df(cfg)
        sources = test_df["source"].tolist()
        references = test_df["target"].tolist()
        prefix = "test"

    total = len(sources)

    start_idx = get_resume_count(pred_file)
    if start_idx > total:
        start_idx = 0
        if os.path.exists(pred_file):
            os.remove(pred_file)

    all_sources, all_predictions, all_references = [], [], []

    device = next(model.parameters()).device
    pbar = tqdm(
        range(start_idx, total, infer_cfg["batch_size"]),
        desc=f"Translating ({mode})",
        unit="batch",
    )
    write_header = start_idx == 0

    for i in pbar:
        batch_sources = sources[i : i + infer_cfg["batch_size"]]
        batch_refs = references[i : i + infer_cfg["batch_size"]]

        inputs = tokenizer(
            batch_sources,
            max_length=infer_cfg["max_length"],
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=infer_cfg["max_length"],
                num_beams=infer_cfg["num_beams"],
                early_stopping=True,
                forced_bos_token_id=model.config.forced_bos_token_id,
            )

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        batch_preds = [p.strip() for p in batch_preds]

        rows = list(zip(batch_sources, batch_preds, batch_refs))
        append_prediction_rows(pred_file, rows, write_header=write_header)
        write_header = False

        all_sources.extend(batch_sources)
        all_predictions.extend(batch_preds)
        all_references.extend(batch_refs)

    save_flat_files(
        translations_dir, prefix, all_sources, all_predictions, all_references
    )

    return exp_dir


def translate_direction(cfg: dict):
    """
    High-level entry: translate both validation and test splits.
    """
    # First validation (for continuity with existing metrics)
    translate_split(cfg, mode="validation")
    # Then test (final evaluation)
    translate_split(cfg, mode="test")
    return True
