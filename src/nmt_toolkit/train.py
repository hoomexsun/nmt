import json
import os

import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
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
        # For mBART/NLLB, src_lang/tgt_lang handled via loader.
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
    tsv_file = cfg["tsv_file"]
    reverse = cfg.get("reverse", False)
    train_cfg = cfg["training"]

    test_size = float(train_cfg["test_size"])
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

    df = load_and_prepare_df(tsv_file, reverse=reverse)

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    train_df.to_csv(os.path.join(output_dir, "train.tsv"), sep="\t", index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.tsv"), sep="\t", index=False)

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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
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

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model to: {output_dir}")
    return output_dir
