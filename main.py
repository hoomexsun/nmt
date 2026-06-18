#!/usr/bin/env python3
"""
Unified task-by-task pipeline for:
  - Maring <-> English
  - Tangkhul <-> English

Supports:
  - MBART50
  - NLLB-200

Workflow per task:
  1. Train if model checkpoint does not exist
  2. Translate if predictions do not exist
  3. Evaluate if evaluation file does not exist
  4. Move to next task
"""

import os
import argparse


from evaluate import evaluate_predictions
from train import train_direction
from translate import translate_direction
from utils import (
    evaluation_exists,
    model_exists,
    predictions_exist,
    prepare_default_exp_dir,
)

# -------------------------------------------------------- Configuration
TASKS = [
    {
        "model_family": "mbart50",
        "hf_model_path": "facebook/mbart-large-50-many-to-many-mmt",
        "folder_prefix": "mbart-large-50-many-to-many-mmt",
        "src_lang": "nng_XX",
        "tgt_lang": "en_XX",
        "tsv_file": "data/nngLatn_engLatn.tsv",
        "reverse": False,
        "direction": "nng_to_en",
    },
    {
        "model_family": "mbart50",
        "hf_model_path": "facebook/mbart-large-50-many-to-many-mmt",
        "folder_prefix": "mbart-large-50-many-to-many-mmt",
        "src_lang": "en_XX",
        "tgt_lang": "nng_XX",
        "tsv_file": "data/nngLatn_engLatn.tsv",
        "reverse": True,
        "direction": "en_to_nng",
    },
    {
        "model_family": "mbart50",
        "hf_model_path": "facebook/mbart-large-50-many-to-many-mmt",
        "folder_prefix": "mbart-large-50-many-to-many-mmt",
        "src_lang": "nmf_XX",
        "tgt_lang": "en_XX",
        "tsv_file": "data/engLatn_nmfLatn.tsv",
        "reverse": True,
        "direction": "nmf_to_en",
    },
    {
        "model_family": "mbart50",
        "hf_model_path": "facebook/mbart-large-50-many-to-many-mmt",
        "folder_prefix": "mbart-large-50-many-to-many-mmt",
        "src_lang": "en_XX",
        "tgt_lang": "nmf_XX",
        "tsv_file": "data/engLatn_nmfLatn.tsv",
        "reverse": False,
        "direction": "en_to_nmf",
    },
    {
        "model_family": "nllb200-dist",
        "hf_model_path": "facebook/nllb-200-distilled-600M",
        "folder_prefix": "nllb-200-distilled-600M",
        "src_lang": "nng_Latn",
        "tgt_lang": "eng_Latn",
        "tsv_file": "data/nngLatn_engLatn.tsv",
        "reverse": False,
        "direction": "nng_to_en",
    },
    {
        "model_family": "nllb200-dist",
        "hf_model_path": "facebook/nllb-200-distilled-600M",
        "folder_prefix": "nllb-200-distilled-600M",
        "src_lang": "eng_Latn",
        "tgt_lang": "nng_Latn",
        "tsv_file": "data/nngLatn_engLatn.tsv",
        "reverse": True,
        "direction": "en_to_nng",
    },
    {
        "model_family": "nllb200-dist",
        "hf_model_path": "facebook/nllb-200-distilled-600M",
        "folder_prefix": "nllb-200-distilled-600M",
        "src_lang": "nmf_Latn",
        "tgt_lang": "eng_Latn",
        "tsv_file": "data/engLatn_nmfLatn.tsv",
        "reverse": True,
        "direction": "nmf_to_en",
    },
    {
        "model_family": "nllb200-dist",
        "hf_model_path": "facebook/nllb-200-distilled-600M",
        "folder_prefix": "nllb-200-distilled-600M",
        "src_lang": "eng_Latn",
        "tgt_lang": "nmf_Latn",
        "tsv_file": "data/engLatn_nmfLatn.tsv",
        "reverse": False,
        "direction": "en_to_nmf",
    },
]


# -------------------------------------------------------- PIPELINE
def process_task(cfg: dict, args):
    model_family = cfg["model_family"]
    folder_prefix = cfg["folder_prefix"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    direction = cfg["direction"]

    exp_dir = prepare_default_exp_dir(folder_prefix, src_lang, tgt_lang)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Task: {direction} | {model_family}")
    print(f"Exp:  {exp_dir}")
    print(f"{'='*60}")

    if args.do_train and not model_exists(exp_dir):
        train_direction(cfg)

    if args.do_translate and not predictions_exist(exp_dir):
        translate_direction(cfg)

    if args.do_evaluate and not evaluation_exists(exp_dir):
        evaluate_predictions(exp_dir, comet_model_tag=args.comet_model_tag)


# -------------------------------------------------------- CLI
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["all", "mbart50", "nllb200-dist"], default="all"
    )
    parser.add_argument(
        "--task",
        choices=["all", "nng_to_en", "en_to_nng", "nmf_to_en", "en_to_nmf"],
        default="all",
    )
    parser.add_argument("--do-train", action="store_true")
    parser.add_argument("--do-translate", action="store_true")
    parser.add_argument("--do-evaluate", action="store_true")
    parser.add_argument("--comet_model_tag", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.do_train and not args.do_translate and not args.do_evaluate:
        args.do_train = True
        args.do_translate = True
        args.do_evaluate = True

    tasks = TASKS
    if args.model != "all":
        tasks = [t for t in tasks if t["model_family"] == args.model]
    if args.task != "all":
        tasks = [t for t in tasks if t["direction"] == args.task]

    if not tasks:
        print("[ERROR] No tasks selected.")
        return

    for cfg in tasks:
        process_task(cfg, args)


if __name__ == "__main__":
    main()
