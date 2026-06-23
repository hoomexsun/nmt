#!/usr/bin/env python3
from pathlib import Path

import nltk
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


nltk.download("wordnet")
nltk.download("omw-1.4")


# -------------------------------------------------------- HELPERS


def prepare_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_and_save_mbart():
    local_dir = MODELS_DIR / "mbart-large-50-many-to-many-mmt"
    prepare_dir(local_dir)
    print(f"[INFO] Downloading mBART-50 to {local_dir} ...")

    tok = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    mdl = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )

    tok.save_pretrained(local_dir)
    mdl.save_pretrained(local_dir)
    print(f"[OK] Saved mBART-50 to {local_dir}")


def download_and_save_nllb():
    local_dir = MODELS_DIR / "nllb-200-distilled-600M"
    prepare_dir(local_dir)
    print(f"[INFO] Downloading NLLB-200 distilled 600M to {local_dir} ...")

    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

    tok.save_pretrained(local_dir)
    mdl.save_pretrained(local_dir)
    print(f"[OK] Saved NLLB-200 distilled 600M to {local_dir}")


def download_and_save_mt5_small():
    local_dir = MODELS_DIR / "mt5-small"
    prepare_dir(local_dir)
    print(f"[INFO] Downloading mT5-small to {local_dir} ...")

    tok = AutoTokenizer.from_pretrained("google/mt5-small")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")

    tok.save_pretrained(local_dir)
    mdl.save_pretrained(local_dir)
    print(f"[OK] Saved mT5-small to {local_dir}")


def download_and_save_byt5_small():
    local_dir = MODELS_DIR / "byt5-small"
    prepare_dir(local_dir)
    print(f"[INFO] Downloading ByT5-small to {local_dir} ...")

    tok = AutoTokenizer.from_pretrained("google/byt5-small")
    mdl = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

    tok.save_pretrained(local_dir)
    mdl.save_pretrained(local_dir)
    print(f"[OK] Saved ByT5-small to {local_dir}")


# -------------------------------------------------------- MAIN


def main():
    print("[INFO] Preparing local models directory:", MODELS_DIR)
    # download_and_save_mbart()
    # download_and_save_nllb()
    download_and_save_mt5_small()
    download_and_save_byt5_small()
    print("[DONE] All base models downloaded and saved locally.")


if __name__ == "__main__":
    main()
