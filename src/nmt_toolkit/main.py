import argparse

from .config import build_job_configs, maybe_enable_offline_mode
from .evaluate import evaluate_predictions
from .gui import launch_gui
from .train import train_direction
from .translate import translate_direction
from .utils import (
    evaluation_exists,
    model_exists,
    predictions_exist,
    prepare_default_exp_dir,
)
from .split_corpus import main as split_corpus_main

# -------------------------------------------------------- PIPELINE


def process_task(cfg: dict, args):
    maybe_enable_offline_mode(cfg)

    model_family = cfg["model_family"]
    folder_prefix = cfg["folder_prefix"]
    src_lang = cfg["src_lang"]
    tgt_lang = cfg["tgt_lang"]
    direction = cfg["direction"]

    exp_dir = prepare_default_exp_dir(
        cfg["base_exp_dir"], folder_prefix, src_lang, tgt_lang
    )

    print(f"\n{'='*60}")
    print(f"Task: {direction} | {model_family}")
    print(f"Exp:  {exp_dir}")
    print(f"{'='*60}")

    if args.do_train and not model_exists(exp_dir):
        train_direction(cfg)

    if args.do_translate and not predictions_exist(exp_dir):
        translate_direction(cfg)

    if args.do_evaluate and not evaluation_exists(exp_dir):
        evaluate_predictions(
            exp_dir,
            comet_model_tag=args.comet_model_tag,
            strict=cfg["evaluation"].get("strict", False),
        )


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
    parser.add_argument("--allow-online", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--pre-split",
        action="store_true",
        help="Run data splitting (train/val/test) before training",
    )
    return parser.parse_args()


# -------------------------------------------------------- MAIN


def main():
    args = parse_args()

    if args.gui:
        launch_gui()
        return

    # Optional: run splitting once before any tasks
    if args.pre_split:
        print("[INFO] Running pre-split of corpora (train/val/test)...")
        split_corpus_main()

    tasks = build_job_configs()

    if not args.do_train and not args.do_translate and not args.do_evaluate:
        args.do_train = True
        args.do_translate = True
        args.do_evaluate = True

    if args.model != "all":
        tasks = [t for t in tasks if t["model_family"] == args.model]
    if args.task != "all":
        tasks = [t for t in tasks if t["direction"] == args.task]

    for task in tasks:
        task["allow_online"] = args.allow_online

    if not tasks:
        print("[ERROR] No tasks selected.")
        return

    for cfg in tasks:
        process_task(cfg, args)


if __name__ == "__main__":
    main()
