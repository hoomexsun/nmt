import os


import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from .loader import load_for_training
from .utils import model_exists, prepare_default_exp_dir, load_and_prepare_df

# Requires 16GB VRAM for this batch size
train_defaults = {
    "test_size": 0.1,
    "seed": 42,
    "max_src_len": 128,
    "max_tgt_len": 128,
    "batch_size": 8,
    "num_epochs": 30,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "report_to": "tensorboard",
}


# ---------------------------------------------------- TRAINING UTILS
def preprocess_function_factory(
    src_lang: str,
    tgt_lang: str,
    tokenizer,
    max_source_length=128,
    max_target_length=128,
):
    def preprocess_function(examples):
        tokenizer.src_lang = src_lang
        model_inputs = tokenizer(
            examples["source"],
            max_length=max_source_length,
            truncation=True,
        )

        tokenizer.tgt_lang = tgt_lang
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
    return {"exact_match": float(exact_match)}


# -------------------------------------------------------- TRAINING
def train_direction(cfg: dict):
    folder_prefix = cfg["folder_prefix"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    tsv_file = cfg["tsv_file"]
    reverse = cfg.get("reverse", False)

    output_dir = prepare_default_exp_dir(folder_prefix, src_lang, tgt_lang)
    os.makedirs(output_dir, exist_ok=True)

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
        test_size=train_defaults["test_size"],
        random_state=train_defaults["seed"],
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
        max_source_length=train_defaults["max_src_len"],
        max_target_length=train_defaults["max_tgt_len"],
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
        learning_rate=train_defaults["lr"],
        per_device_train_batch_size=train_defaults["batch_size"],
        per_device_eval_batch_size=train_defaults["batch_size"],
        weight_decay=train_defaults["weight_decay"],
        num_train_epochs=train_defaults["num_epochs"],
        predict_with_generate=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=train_defaults["report_to"],
        seed=train_defaults["seed"],
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
