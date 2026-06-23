import json
import os
from pathlib import Path

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .loader import load_for_training
from .utils import load_and_prepare_df, model_exists, prepare_default_exp_dir

# -------------------------------------------------------- TRAINING UTILS


def preprocess_function_factory(
    src_lang, tgt_lang, tokenizer, max_source_length=128, max_target_length=128
):
    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_source_length,
            truncation=True,
        )

        labels = tokenizer(
            text_target=examples["target"],
            max_length=max_target_length,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [x.strip() for x in decoded_preds]
    decoded_labels = [x.strip() for x in decoded_labels]

    exact_match = np.mean([p == l for p, l in zip(decoded_preds, decoded_labels)])

    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])
    chrf = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=0)
    chrfpp = sacrebleu.corpus_chrf(decoded_preds, [decoded_labels], word_order=2)
    ter = sacrebleu.corpus_ter(decoded_preds, [decoded_labels])

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_f1_scores = [
        rouge.score(ref, pred)["rougeL"].fmeasure
        for pred, ref in zip(decoded_preds, decoded_labels)
    ]
    rouge_l_f1 = float(np.mean(rouge_l_f1_scores)) if rouge_l_f1_scores else 0.0

    meteor_scores = [
        meteor_score([ref.split()], pred.split())
        for pred, ref in zip(decoded_preds, decoded_labels)
    ]
    meteor = float(np.mean(meteor_scores)) if meteor_scores else 0.0

    return {
        "exact_match": float(exact_match),
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "chrfpp": float(chrfpp.score),
        "ter": float(ter.score),
        "rougeL": rouge_l_f1,
        "meteor": meteor,
    }


def _save_train_config(output_dir: str, cfg: dict):
    path = os.path.join(output_dir, "train_config.json")

    safe_cfg = dict(cfg)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_cfg, f, indent=2, ensure_ascii=False)

    print(f"Saved train config to: {path}")


# -------------------------------------------------------- TRAINING


def train_direction(cfg: dict):
    folder_prefix = cfg["folder_prefix"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    reverse = cfg.get("reverse", False)
    train_cfg = cfg["training"]

    seed = int(train_cfg["seed"])
    max_src_len = int(train_cfg["max_src_len"])
    max_tgt_len = int(train_cfg["max_tgt_len"])
    batch_size = int(train_cfg["batch_size"])
    num_epochs = int(train_cfg["num_epochs"])
    lr = float(train_cfg["lr"])
    weight_decay = float(train_cfg["weight_decay"])
    report_to = train_cfg["report_to"]

    output_dir = prepare_default_exp_dir(
        cfg["base_exp_dir"], folder_prefix, src_lang, tgt_lang
    )
    os.makedirs(output_dir, exist_ok=True)

    _save_train_config(output_dir, cfg)

    if model_exists(output_dir):
        print(f"[SKIP] Model already exists: {output_dir}")
        return output_dir

    print(f"\n{'='*60}")
    print(f"Training: {src_lang} -> {tgt_lang}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*60}\n")

    # --------------------------------------------------------
    # Load precomputed train/validation splits (CSV)
    # --------------------------------------------------------
    corpus_file = cfg["corpus_file"]  # from directions.yaml
    base_path = Path(corpus_file)
    split_dir = base_path.with_suffix("")  # e.g. data/engLatn_nngLatn

    train_path = split_dir / "train.csv"
    val_path = split_dir / "validation.csv"
    test_path = split_dir / "test.csv"  # reserved for final evaluation

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Pre-split files not found.\n"
            f"Expected train: {train_path}\n"
            f"Expected validation: {val_path}\n"
            f"Run 'python -m src.nmt_toolkit.split_corpus' first."
        )

    # Load pre-split train/validation; apply reverse at direction level
    train_df = load_and_prepare_df(str(train_path), reverse=reverse)
    val_df = load_and_prepare_df(str(val_path), reverse=reverse)

    print(val_df.head(3))

    # Keep copies under exp_dir (for translation script)
    train_out = os.path.join(output_dir, "train.csv")
    val_out = os.path.join(output_dir, "validation.csv")
    train_df.to_csv(train_out, sep=",", index=False)
    val_df.to_csv(val_out, sep=",", index=False)

    hf_dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(
                train_df.reset_index(drop=True), preserve_index=False
            ),
            "validation": Dataset.from_pandas(
                val_df.reset_index(drop=True), preserve_index=False
            ),
        }
    )

    model, tokenizer = load_for_training(cfg)

    preprocess_fn = preprocess_function_factory(
        src_lang,
        tgt_lang,
        tokenizer,
        max_source_length=max_src_len,
        max_target_length=max_tgt_len,
    )

    tokenized = hf_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=hf_dataset["train"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        num_train_epochs=num_epochs,
        predict_with_generate=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=report_to,
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, tokenizer),
    )

    # Resume from last checkpoint only if it exists
    last_checkpoint = None
    if os.path.isdir(output_dir):
        # look for subdirs named 'checkpoint-XXXX'
        checkpoints = [
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(output_dir, d))
        ]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)

    if last_checkpoint is not None:
        print(f"[INFO] Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("[INFO] No checkpoint found, starting training from scratch.")
        trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model to: {output_dir}")
    return output_dir
