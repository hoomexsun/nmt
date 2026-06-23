import csv
import os
import pandas as pd
import torch
from tqdm.auto import tqdm

from .loader import load_for_inference
from .utils import load_validation_df, prepare_default_exp_dir

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
        writer = csv.writer(f, delimiter="\t")
        if write_header:
            writer.writerow(["src", "pred", "ref"])
        writer.writerows(rows)


def save_flat_files(translations_dir: str, srcs, preds, refs):
    src_path = os.path.join(translations_dir, "val.src.txt")
    ref_path = os.path.join(translations_dir, "val.ref.txt")
    pred_path = os.path.join(translations_dir, "val.pred.txt")

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


# -------------------------------------------------------- TRANSLATION


def translate_direction(cfg: dict):
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

    pred_file = os.path.join(translations_dir, "validation.predictions.tsv")
    if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
        print(f"[SKIP] Predictions already exist: {pred_file}")
        return exp_dir

    model, tokenizer = load_for_inference(exp_dir, cfg)
    val_df = load_validation_df(exp_dir)

    sources = val_df["source"].tolist()
    references = val_df["target"].tolist()
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
        desc="Translating",
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

    save_flat_files(translations_dir, all_sources, all_predictions, all_references)

    return exp_dir
