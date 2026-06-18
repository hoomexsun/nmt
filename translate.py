import os
import csv

import torch
import pandas as pd
from tqdm.auto import tqdm

from .loader import load_for_inference
from .utils import predictions_exist, prepare_default_exp_dir, load_validation_df

infer_defaults = {
    "max_length": 128,
    "batch_size": 16,
    "num_beams": 4,
}


# ---------------------------------------------------- TRANSLATION UTILS
def get_resume_count(pred_file: str) -> int:
    if not os.path.exists(pred_file):
        return 0
    try:
        existing_df = pd.read_csv(pred_file, sep="\t")
        return len(existing_df)
    except Exception as e:
        print(f"[WARNING] Could not read existing prediction file {pred_file}: {e}")
        return 0


# -------------------------------------------------------- TRANSLATION
def append_prediction_rows(pred_file: str, rows: list, write_header: bool):
    mode = "w" if write_header else "a"
    with open(pred_file, mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["src", "pred", "ref"])
        writer.writerows(rows)


def translate_direction(cfg: dict):
    folder_prefix = cfg["folder_prefix"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]

    exp_dir = prepare_default_exp_dir(folder_prefix, src_lang, tgt_lang)

    print(f"exp_dir={exp_dir}")

    if predictions_exist(exp_dir):
        print(f"[SKIP] Predictions already exist: {exp_dir}")
        return exp_dir

    model, tokenizer = load_for_inference(exp_dir, cfg)
    val_df = load_validation_df(exp_dir)

    pred_file = os.path.join(exp_dir, "validation.predictions.tsv")
    sources = val_df["source"].tolist()
    references = val_df["target"].tolist()
    total = len(sources)

    start_idx = get_resume_count(pred_file)
    if start_idx > total:
        start_idx = 0
        if os.path.exists(pred_file):
            os.remove(pred_file)

    if start_idx > 0:
        existing_df = pd.read_csv(pred_file, sep="\t")
        all_sources = existing_df["src"].tolist()
        all_predictions = existing_df["pred"].tolist()
        all_references = existing_df["ref"].tolist()
    else:
        all_sources, all_predictions, all_references = [], [], []

    device = next(model.parameters()).device
    pbar = tqdm(
        range(start_idx, total, infer_defaults["batch_size"]),
        desc="Translating",
        unit="batch",
    )
    write_header = start_idx == 0

    for i in pbar:
        batch_sources = sources[i : i + infer_defaults["batch_size"]]
        batch_refs = references[i : i + infer_defaults["batch_size"]]

        inputs = tokenizer(
            batch_sources,
            max_length=infer_defaults["max_length"],
            truncation=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=infer_defaults["max_length"],
                num_beams=infer_defaults["num_beams"],
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

    return exp_dir
