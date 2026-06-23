import csv
import json
import os

import sacrebleu
import torch

try:
    from comet import download_model, load_from_checkpoint
except Exception:
    download_model = None
    load_from_checkpoint = None


# -------------------------------------------------------- EVALUATION UTILS


def load_comet_model(comet_model_tag):
    if download_model is None or load_from_checkpoint is None:
        print("[WARNING] COMET is not installed. Skipping COMET.")
        return None
    try:
        model_path = download_model(comet_model_tag)
        return load_from_checkpoint(model_path)
    except Exception as e:
        print(f"[WARNING] COMET model could not be loaded: {e}")
        return None


def compute_comet(model, srcs, preds, refs):
    data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
    try:
        output = model.predict(
            data,
            batch_size=8,
            gpus=1 if torch.cuda.is_available() else 0,
        )
    except TypeError:
        output = model.predict(data, batch_size=8)

    if hasattr(output, "system_score"):
        return float(output.system_score)
    if isinstance(output, dict) and "system_score" in output:
        return float(output["system_score"])
    if isinstance(output, tuple) and len(output) >= 2:
        return float(output[1])
    if hasattr(output, "scores"):
        scores = output.scores
        return float(sum(scores) / len(scores)) if scores else None
    if isinstance(output, dict) and "scores" in output:
        scores = output["scores"]
        return float(sum(scores) / len(scores)) if scores else None
    return None


# -------------------------------------------------------- EVALUATION


def evaluate_predictions(
    exp_dir: str, comet_model_tag: str = None, strict: bool = False
):
    translations_dir = os.path.join(exp_dir, "translations")
    scores_dir = os.path.join(exp_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    pred_tsv_path = os.path.join(translations_dir, "validation.predictions.tsv")
    txt_path = os.path.join(scores_dir, "evaluation_results.txt")
    csv_path = os.path.join(scores_dir, "evaluation_results.csv")
    json_path = os.path.join(scores_dir, "evaluation_results.json")

    if not os.path.exists(pred_tsv_path):
        raise FileNotFoundError(f"Prediction TSV not found: {pred_tsv_path}")

    srcs, preds, refs = [], [], []
    with open(pred_tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            srcs.append(row.get("src", ""))
            preds.append(row.get("pred", ""))
            refs.append(row.get("ref", ""))

    if strict and not (len(srcs) == len(preds) == len(refs)):
        raise ValueError("Length mismatch in prediction file")

    n = min(len(srcs), len(preds), len(refs))
    srcs, preds, refs = srcs[:n], preds[:n], refs[:n]

    bleu = sacrebleu.corpus_bleu(preds, [refs])
    chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=0)
    chrfpp = sacrebleu.corpus_chrf(preds, [refs], word_order=2)
    ter = sacrebleu.corpus_ter(preds, [refs])

    comet_score = None
    if comet_model_tag:
        comet_model = load_comet_model(comet_model_tag)
        if comet_model is not None:
            try:
                comet_score = compute_comet(comet_model, srcs, preds, refs)
            except Exception as e:
                print(f"[WARNING] COMET failed: {e}")

    metrics = {
        "exp_dir": exp_dir,
        "prediction_tsv": pred_tsv_path,
        "n_examples": n,
        "bleu": round(float(bleu.score), 4),
        "chrf": round(float(chrf.score), 4),
        "chrfpp": round(float(chrfpp.score), 4),
        "ter": round(float(ter.score), 4),
        "comet": None if comet_score is None else round(float(comet_score), 4),
        "signature": bleu.format(),
    }

    # Human-readable text summary
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    # CSV metrics
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    # JSON metrics (nice for tools / tracking)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved: {txt_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    return metrics
