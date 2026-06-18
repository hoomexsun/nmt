#!/usr/bin/env python3
import os
import csv
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import torch
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

BASE_EXP_DIR = "exp"
MAX_LENGTH = 128
NUM_BEAMS = 4

MODEL_CONFIGS = {
    "mBART-50": {
        "base_name": "facebook/mbart-large-50-many-to-many-mmt",
        "folder_name": "mbart-large-50-many-to-many-mmt",
        "type": "mbart",
    },
    "NLLB-200": {
        "base_name": "facebook/nllb-200-distilled-600M",
        "folder_name": "nllb-200-distilled-600M",
        "type": "nllb",
    },
}

LANG_NAMES = {
    "nng_XX": "Maring",
    "nmf_XX": "Tangkhul",
    "en_XX": "English",
    "nng_Latn": "Maring",
    "nmf_Latn": "Tangkhul",
    "eng_Latn": "English",
}

NLLB_LANG_MAP = {
    "nng_Latn": "nng_Latn",
    "nmf_Latn": "nmf_Latn",
    "eng_Latn": "eng_Latn",
}

DIRECTIONS = {
    "Maring -> English": ("nng_XX", "en_XX"),
    "English -> Maring": ("en_XX", "nng_XX"),
    "Tangkhul -> English": ("nmf_XX", "en_XX"),
    "English -> Tangkhul": ("en_XX", "nmf_XX"),
}

CORPUS_FILES = {
    "Maring -> English": ("data/nngLatn_engLatn.tsv", False),
    "English -> Maring": ("data/nngLatn_engLatn.tsv", True),
    "Tangkhul -> English": ("data/engLatn_nmfLatn.tsv", True),
    "English -> Tangkhul": ("data/engLatn_nmfLatn.tsv", False),
}


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Low-Resource Translator")
        self.root.geometry("1000x640")
        self.root.minsize(860, 580)
        self.root.configure(bg="#f6f7fb")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.current_model_dir = None
        self.current_direction = tk.StringVar(value="Maring -> English")
        self.current_model_name = tk.StringVar(value="mBART-50")
        self.status_var = tk.StringVar(value=f"Ready on {self.device}")
        self.loaded_model_var = tk.StringVar(value="Not loaded")
        self.is_loading = False
        self.is_translating = False
        self.samples = {}

        self._build_ui()
        self.load_samples()
        self.refresh_samples_ui()
        self.load_model_async()

    def make_exp_dir(self, model_folder_name: str, src_lang: str, tgt_lang: str) -> str:
        return os.path.join(BASE_EXP_DIR, f"{model_folder_name}_{src_lang}_{tgt_lang}")

    def get_direction(self):
        return DIRECTIONS[self.current_direction.get()]

    def get_model_config(self):
        return MODEL_CONFIGS[self.current_model_name.get()]

    def _get_codes(self, src_lang, tgt_lang, model_type):
        if model_type == "mbart":
            return src_lang, tgt_lang
        mbart_to_nllb = {
            "nng_XX": "nng_Latn",
            "nmf_XX": "nmf_Latn",
            "en_XX": "eng_Latn",
        }
        return mbart_to_nllb[src_lang], mbart_to_nllb[tgt_lang]

    def _get_model_dir(self, cfg, src_lang, tgt_lang):
        src_code, tgt_code = self._get_codes(src_lang, tgt_lang, cfg["type"])
        return (
            self.make_exp_dir(cfg["folder_name"], src_code, tgt_code),
            src_code,
            tgt_code,
        )

    def _set_nllb_tokenizer(self, tokenizer, model, src_code, tgt_code):
        for lang in [src_code, tgt_code]:
            if lang not in tokenizer.get_vocab():
                tokenizer.add_special_tokens(
                    {"additional_special_tokens": [lang]},
                    replace_additional_special_tokens=False,
                )
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
        tokenizer.src_lang = src_code
        tokenizer.tgt_lang = tgt_code
        return self._get_nllb_token_id(tokenizer, tgt_code)

    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        bg = "#f6f7fb"
        card = "#ffffff"
        line = "#dde3ee"
        text = "#1f2937"
        subtle = "#5f6b7a"
        accent = "#2563eb"
        accent_hover = "#1d4ed8"

        self.root.option_add("*Font", ("Segoe UI", 10))

        style.configure("App.TFrame", background=bg)
        style.configure("Card.TFrame", background=card, relief="flat")
        style.configure("TLabel", background=bg, foreground=text, font=("Segoe UI", 10))
        style.configure(
            "Card.TLabel", background=card, foreground=text, font=("Segoe UI", 10)
        )
        style.configure(
            "Title.TLabel",
            background=bg,
            foreground=text,
            font=("Segoe UI Semibold", 16),
        )
        style.configure(
            "Header.TLabel",
            background=card,
            foreground=text,
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Subtle.TLabel", background=bg, foreground=subtle, font=("Segoe UI", 9)
        )
        style.configure(
            "Status.TLabel", background=card, foreground=subtle, font=("Segoe UI", 9)
        )
        style.configure(
            "TButton", font=("Segoe UI", 10), padding=(10, 6), relief="flat"
        )
        style.map("TButton", background=[("active", "#eef3ff")])
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(12, 7),
            foreground="#ffffff",
            background=accent,
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", accent_hover), ("pressed", accent_hover)],
        )
        style.configure("TCombobox", padding=4)

        outer = ttk.Frame(self.root, style="App.TFrame", padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(5, weight=1)

        title_row = ttk.Frame(outer, style="App.TFrame")
        title_row.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            title_row, text="Low-Resource Translator Demo", style="Title.TLabel"
        ).pack(anchor="w")
        ttk.Label(
            title_row,
            text="Compact laptop-friendly layout • mBART-50 + NLLB-200",
            style="Subtle.TLabel",
        ).pack(anchor="w", pady=(1, 0))

        controls_card = ttk.Frame(outer, style="Card.TFrame", padding=10)
        controls_card.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        for i in range(6):
            controls_card.columnconfigure(i, weight=0)
        controls_card.columnconfigure(1, weight=1)
        controls_card.columnconfigure(3, weight=1)

        ttk.Label(controls_card, text="Model", style="Card.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.model_box = ttk.Combobox(
            controls_card,
            textvariable=self.current_model_name,
            values=list(MODEL_CONFIGS.keys()),
            state="readonly",
            width=16,
        )
        self.model_box.grid(row=0, column=1, sticky="ew", padx=(6, 12))
        self.model_box.bind(
            "<<ComboboxSelected>>", lambda e: self.on_model_or_direction_changed()
        )

        ttk.Label(controls_card, text="Direction", style="Card.TLabel").grid(
            row=0, column=2, sticky="w"
        )
        self.direction_box = ttk.Combobox(
            controls_card,
            textvariable=self.current_direction,
            values=list(DIRECTIONS.keys()),
            state="readonly",
            width=24,
        )
        self.direction_box.grid(row=0, column=3, sticky="ew", padx=(6, 12))
        self.direction_box.bind(
            "<<ComboboxSelected>>", lambda e: self.on_model_or_direction_changed()
        )

        ttk.Button(controls_card, text="Swap", command=self.swap_direction).grid(
            row=0, column=4, sticky="ew", padx=(0, 6)
        )
        ttk.Button(controls_card, text="Reload", command=self.load_model_async).grid(
            row=0, column=5, sticky="ew"
        )

        info_card = ttk.Frame(outer, style="Card.TFrame", padding=10)
        info_card.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        info_card.columnconfigure(1, weight=1)
        info_card.columnconfigure(3, weight=1)

        ttk.Label(info_card, text="Loaded model:", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            info_card, textvariable=self.loaded_model_var, style="Status.TLabel"
        ).grid(row=0, column=1, sticky="ew", padx=(8, 18))
        ttk.Label(info_card, text="Status:", style="Header.TLabel").grid(
            row=0, column=2, sticky="w"
        )
        ttk.Label(info_card, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=3, sticky="ew", padx=(8, 0)
        )

        action_card = ttk.Frame(outer, style="Card.TFrame", padding=10)
        action_card.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(
            action_card,
            text="Translate",
            style="Accent.TButton",
            command=self.translate_async,
        ).pack(side="left")
        ttk.Button(
            action_card,
            text="Clear",
            command=self.clear_text,
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            action_card,
            text="Load Sample",
            command=self.load_selected_sample,
        ).pack(side="left", padx=(8, 0))

        samples_card = ttk.Frame(outer, style="Card.TFrame", padding=10)
        samples_card.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        samples_card.columnconfigure(0, weight=1)

        ttk.Label(samples_card, text="Corpus Sample", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.sample_var = tk.StringVar()
        self.sample_box = ttk.Combobox(
            samples_card,
            textvariable=self.sample_var,
            state="readonly",
        )
        self.sample_box.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.sample_box.bind(
            "<<ComboboxSelected>>", lambda e: self.load_selected_sample()
        )

        main = ttk.Frame(outer, style="App.TFrame")
        main.grid(row=5, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        left = ttk.Frame(main, style="Card.TFrame", padding=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        right = ttk.Frame(main, style="Card.TFrame", padding=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        ttk.Label(left, text="Input Sentence / Text", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.input_text = tk.Text(
            left,
            height=8,
            wrap="word",
            font=("Segoe UI", 11),
            relief="flat",
            bd=1,
            bg="#fbfcfe",
            fg="#111827",
            insertbackground="#111827",
            highlightthickness=1,
            highlightbackground=line,
            highlightcolor=accent,
            padx=10,
            pady=10,
        )
        self.input_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        ttk.Label(right, text="Translated Sentence / Text", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.output_text = tk.Text(
            right,
            height=8,
            wrap="word",
            font=("Segoe UI", 11),
            relief="flat",
            bd=1,
            bg="#fbfcfe",
            fg="#111827",
            insertbackground="#111827",
            highlightthickness=1,
            highlightbackground=line,
            highlightcolor=accent,
            padx=10,
            pady=10,
        )
        self.output_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.output_text.configure(state="disabled")

    def load_samples(self):
        self.samples = {}
        for direction, (path, reverse) in CORPUS_FILES.items():
            items = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
                    first_row = next(reader, None)
                    if first_row and len(first_row) >= 2:
                        try:
                            int(first_row[0])
                            rows_iter = [first_row] + list(reader)
                        except Exception:
                            rows_iter = reader
                    else:
                        rows_iter = reader

                    for row in rows_iter:
                        if len(row) < 2:
                            continue
                        src, tgt = row[0].strip(), row[1].strip()
                        if not src or not tgt:
                            continue
                        if reverse:
                            src, tgt = tgt, src
                        items.append((src, tgt))
                        if len(items) >= 12:
                            break
            self.samples[direction] = items

    def refresh_samples_ui(self):
        direction = self.current_direction.get()
        items = self.samples.get(direction, [])
        formatted = [f"{i+1}. {src}" for i, (src, _) in enumerate(items)]
        self.sample_box["values"] = formatted
        self.sample_var.set(formatted[0] if formatted else "")

    def on_model_or_direction_changed(self):
        self.refresh_samples_ui()
        self.load_model_async()

    def load_selected_sample(self):
        direction = self.current_direction.get()
        items = self.samples.get(direction, [])
        current = self.sample_box.current()
        if current is None or current < 0 or current >= len(items):
            return
        src, _ = items[current]
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", src)

    def swap_direction(self):
        order = list(DIRECTIONS.keys())
        cur = self.current_direction.get()
        idx = order.index(cur)
        self.current_direction.set(order[(idx + 1) % len(order)])
        self.on_model_or_direction_changed()

    def clear_text(self):
        self.input_text.delete("1.0", tk.END)
        self.set_output("")
        self.status_var.set("Cleared input and output.")

    def set_output(self, text: str):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", text)
        self.output_text.configure(state="disabled")

    def load_model_async(self):
        if self.is_loading:
            return
        self.is_loading = True
        threading.Thread(target=self.load_model, daemon=True).start()

    def _register_custom_langs_mbart(self, tokenizer, langs):
        for lang in langs:
            if lang in tokenizer.lang_code_to_id:
                continue
            lang_id = tokenizer.convert_tokens_to_ids(lang)
            if lang_id == tokenizer.unk_token_id:
                raise ValueError(f"Language token {lang} not found in tokenizer vocab")
            tokenizer.lang_code_to_id[lang] = lang_id

    def _get_nllb_token_id(self, tokenizer, lang_code):
        if (
            hasattr(tokenizer, "lang_code_to_id")
            and lang_code in tokenizer.lang_code_to_id
        ):
            return tokenizer.lang_code_to_id[lang_code]
        token_id = tokenizer.convert_tokens_to_ids(lang_code)
        if token_id == tokenizer.unk_token_id:
            raise ValueError(f"NLLB token not found in tokenizer: {lang_code}")
        return token_id

    def load_model(self):
        src_lang, tgt_lang = self.get_direction()
        cfg = self.get_model_config()
        model_dir, src_code, tgt_code = self._get_model_dir(cfg, src_lang, tgt_lang)

        self.root.after(
            0,
            lambda path=model_dir, name=self.current_model_name.get(): self.status_var.set(
                f"Loading {name} from {path} ..."
            ),
        )

        if not os.path.isdir(model_dir):
            self.root.after(
                0,
                lambda: self.status_var.set(f"Model directory not found: {model_dir}"),
            )
            self.root.after(0, lambda: self.loaded_model_var.set("Not loaded"))
            self.is_loading = False
            return

        try:
            if cfg["type"] == "mbart":
                tokenizer = MBart50TokenizerFast.from_pretrained(
                    model_dir,
                    src_lang="en_XX",
                    tgt_lang="en_XX",
                )
                model = MBartForConditionalGeneration.from_pretrained(model_dir)
                self._register_custom_langs_mbart(tokenizer, [src_lang, tgt_lang])
                tokenizer.src_lang = src_lang
                tokenizer.tgt_lang = tgt_lang
                model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_dir,
                        fix_mistral_regex=True,
                    )
                except TypeError:
                    tokenizer = AutoTokenizer.from_pretrained(model_dir)

                model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
                for lang in [src_code, tgt_code]:
                    if lang not in tokenizer.get_vocab():
                        tokenizer.add_special_tokens(
                            {"additional_special_tokens": [lang]},
                            replace_additional_special_tokens=False,
                        )
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception:
                    pass
                tokenizer.src_lang = src_code
                tokenizer.tgt_lang = tgt_code
                model.config.forced_bos_token_id = self._get_nllb_token_id(
                    tokenizer, tgt_code
                )

            model.to(self.device)
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            self.current_model_dir = model_dir

            loaded_text = (
                f"{self.current_model_name.get()} • {os.path.basename(model_dir)}"
            )
            status_text = (
                f"Loaded {self.current_model_name.get()} | "
                f"{LANG_NAMES[src_lang]} -> {LANG_NAMES[tgt_lang]} | Device: {self.device}"
            )
            self.root.after(0, lambda text=loaded_text: self.loaded_model_var.set(text))
            self.root.after(0, lambda text=status_text: self.status_var.set(text))

        except Exception as e:
            err = str(e)
            self.model = None
            self.tokenizer = None
            self.current_model_dir = None
            self.root.after(0, lambda: self.loaded_model_var.set("Not loaded"))
            self.root.after(
                0, lambda err=err: self.status_var.set(f"Failed to load model: {err}")
            )
        finally:
            self.is_loading = False

    def translate_async(self):
        if self.is_translating:
            return
        self.is_translating = True
        threading.Thread(target=self.translate, daemon=True).start()

    def translate(self):
        try:
            if self.model is None or self.tokenizer is None:
                self.root.after(
                    0, lambda: messagebox.showerror("Error", "Model is not loaded yet.")
                )
                return

            text = self.input_text.get("1.0", tk.END).strip()
            if not text:
                self.root.after(
                    0,
                    lambda: messagebox.showwarning(
                        "Warning", "Please enter some text."
                    ),
                )
                return

            src_lang, tgt_lang = self.get_direction()
            cfg = self.get_model_config()
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Translating with {self.current_model_name.get()}..."
                ),
            )

            if cfg["type"] == "mbart":
                self._register_custom_langs_mbart(self.tokenizer, [src_lang, tgt_lang])
                self.tokenizer.src_lang = src_lang
                self.tokenizer.tgt_lang = tgt_lang
                forced_bos = self.tokenizer.lang_code_to_id[tgt_lang]
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                )
            else:
                src_code, tgt_code = self._get_codes(src_lang, tgt_lang, cfg["type"])
                self.tokenizer.src_lang = src_code
                self.tokenizer.tgt_lang = tgt_code
                forced_bos = self._get_nllb_token_id(self.tokenizer, tgt_code)
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=MAX_LENGTH,
                    num_beams=NUM_BEAMS,
                    early_stopping=True,
                )

            translation = self.tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()
            self.root.after(0, lambda: self.set_output(translation))
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Done | {self.current_model_name.get()} | {LANG_NAMES[src_lang]} -> {LANG_NAMES[tgt_lang]} | Device: {self.device}"
                ),
            )
        except Exception as e:
            err = str(e)
            self.root.after(
                0, lambda err=err: messagebox.showerror("Translation Error", err)
            )
            self.root.after(0, lambda: self.status_var.set("Translation failed."))
        finally:
            self.is_translating = False


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
