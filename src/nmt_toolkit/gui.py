#!/usr/bin/env python3
import csv
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)

from .config import build_direction_registry, load_all_configs
from .utils import prepare_default_exp_dir

LANG_NAMES = {
    "nng_XX": "Maring",
    "nmf_XX": "Tangkhul",
    "en_XX": "English",
    "nng_Latn": "Maring",
    "nmf_Latn": "Tangkhul",
    "eng_Latn": "English",
}


# -------------------------------------------------------- GUI HELPERS


def build_gui_catalog():
    directions = build_direction_registry()
    models_cfg, _, _, runtime_cfg = load_all_configs()
    model_registry = models_cfg["models"]
    gui_cfg = runtime_cfg["gui"]

    model_names = {
        "mbart50": "mBART-50",
        "nllb200-dist": "NLLB-200",
    }

    model_choices = [(model_names[k], k) for k in model_registry.keys()]
    direction_choices = [(item["label"], item["direction"]) for item in directions]

    direction_map = {item["direction"]: item for item in directions}
    label_to_direction = {item["label"]: item["direction"] for item in directions}
    direction_to_label = {item["direction"]: item["label"] for item in directions}
    model_label_to_key = {label: key for label, key in model_choices}
    model_key_to_label = {key: label for label, key in model_choices}

    return {
        "gui_cfg": gui_cfg,
        "direction_map": direction_map,
        "label_to_direction": label_to_direction,
        "direction_to_label": direction_to_label,
        "model_choices": model_choices,
        "model_label_to_key": model_label_to_key,
        "model_key_to_label": model_key_to_label,
    }


def load_corpus_samples(corpus_file: str, reverse: bool, limit: int = 12):
    items = []
    if os.path.exists(corpus_file):
        with open(corpus_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",")
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
                if len(items) >= limit:
                    break
    return items


# -------------------------------------------------------- GUI APP


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Low-Resource Translator")
        self.root.geometry("1000x640")
        self.root.minsize(860, 580)
        self.root.configure(bg="#f6f7fb")

        self.catalog = build_gui_catalog()
        self.gui_cfg = self.catalog["gui_cfg"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.current_model_dir = None
        self.is_loading = False
        self.is_translating = False
        self.samples = {}

        first_direction_label = list(self.catalog["label_to_direction"].keys())[0]
        first_model_label = self.catalog["model_choices"][0][0]

        self.current_direction_label = tk.StringVar(value=first_direction_label)
        self.current_model_label = tk.StringVar(value=first_model_label)
        self.status_var = tk.StringVar(value=f"Ready on {self.device}")
        self.loaded_model_var = tk.StringVar(value="Not loaded")
        self.sample_var = tk.StringVar()

        self._build_ui()
        self.load_samples()
        self.refresh_samples_ui()
        self.load_model_async()

    def _selected_direction_key(self):
        return self.catalog["label_to_direction"][self.current_direction_label.get()]

    def _selected_model_key(self):
        return self.catalog["model_label_to_key"][self.current_model_label.get()]

    def _current_pair_cfg(self):
        direction_key = self._selected_direction_key()
        model_key = self._selected_model_key()
        direction_info = self.catalog["direction_map"][direction_key]
        pair = direction_info["pairs"][model_key]
        return {
            "direction": direction_key,
            "direction_label": direction_info["label"],
            "corpus_file": direction_info["corpus_file"],
            **pair,
        }

    def _exp_dir_for_pair(self, pair_cfg):
        return prepare_default_exp_dir(
            os.path.abspath("exp"),
            pair_cfg["folder_prefix"],
            pair_cfg["src_lang"],
            pair_cfg["tgt_lang"],
        )

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
            text="YAML-driven local-model toolkit • mBART-50 + NLLB-200",
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
            textvariable=self.current_model_label,
            values=[label for label, _ in self.catalog["model_choices"]],
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
            textvariable=self.current_direction_label,
            values=list(self.catalog["label_to_direction"].keys()),
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
        for direction_key, direction_info in self.catalog["direction_map"].items():
            # prefer mbart50 pair for sample reversal, or fall back
            pair = direction_info["pairs"].get("mbart50") or next(
                iter(direction_info["pairs"].values())
            )
            self.samples[direction_key] = load_corpus_samples(
                direction_info["corpus_file"],
                pair.get("reverse", False),
            )

    def refresh_samples_ui(self):
        direction_key = self._selected_direction_key()
        items = self.samples.get(direction_key, [])
        formatted = [f"{i+1}. {src}" for i, (src, _) in enumerate(items)]
        self.sample_box["values"] = formatted
        self.sample_var.set(formatted[0] if formatted else "")

    def on_model_or_direction_changed(self):
        self.refresh_samples_ui()
        self.load_model_async()

    def load_selected_sample(self):
        direction_key = self._selected_direction_key()
        items = self.samples.get(direction_key, [])
        current = self.sample_box.current()
        if current is None or current < 0 or current >= len(items):
            return
        src, _ = items[current]
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", src)

    def swap_direction(self):
        labels = list(self.catalog["label_to_direction"].keys())
        cur = self.current_direction_label.get()
        idx = labels.index(cur)
        self.current_direction_label.set(labels[(idx + 1) % len(labels)])
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
        pair_cfg = self._current_pair_cfg()
        model_dir = self._exp_dir_for_pair(pair_cfg)
        model_family = pair_cfg["model_family"]
        src_lang = pair_cfg["src_lang"]
        tgt_lang = pair_cfg["tgt_lang"]

        self.root.after(
            0,
            lambda path=model_dir, name=self.current_model_label.get(): self.status_var.set(
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
            if model_family == "mbart50":
                tokenizer = MBart50TokenizerFast.from_pretrained(
                    model_dir,
                    src_lang="en_XX",
                    tgt_lang="en_XX",
                    local_files_only=True,
                )
                model = MBartForConditionalGeneration.from_pretrained(
                    model_dir,
                    local_files_only=True,
                )
                self._register_custom_langs_mbart(tokenizer, [src_lang, tgt_lang])
                tokenizer.src_lang = src_lang
                tokenizer.tgt_lang = tgt_lang
                model.config.forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    local_files_only=True,
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_dir,
                    local_files_only=True,
                )
                tokenizer.src_lang = src_lang
                tokenizer.tgt_lang = tgt_lang
                model.config.forced_bos_token_id = self._get_nllb_token_id(
                    tokenizer,
                    tgt_lang,
                )

            model.to(self.device)
            model.eval()

            self.model = model
            self.tokenizer = tokenizer
            self.current_model_dir = model_dir

            loaded_text = (
                f"{self.current_model_label.get()} • {os.path.basename(model_dir)}"
            )
            status_text = (
                f"Loaded {self.current_model_label.get()} | "
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

            pair_cfg = self._current_pair_cfg()
            src_lang = pair_cfg["src_lang"]
            tgt_lang = pair_cfg["tgt_lang"]
            model_family = pair_cfg["model_family"]

            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Translating with {self.current_model_label.get()}..."
                ),
            )

            if model_family == "mbart50":
                self._register_custom_langs_mbart(self.tokenizer, [src_lang, tgt_lang])
                self.tokenizer.src_lang = src_lang
                self.tokenizer.tgt_lang = tgt_lang
                forced_bos = self.tokenizer.lang_code_to_id[tgt_lang]
            else:
                self.tokenizer.src_lang = src_lang
                self.tokenizer.tgt_lang = tgt_lang
                forced_bos = self._get_nllb_token_id(self.tokenizer, tgt_lang)

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.gui_cfg["max_length"],
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=self.gui_cfg["max_length"],
                    num_beams=self.gui_cfg["num_beams"],
                    early_stopping=True,
                )

            translation = self.tokenizer.decode(
                generated[0], skip_special_tokens=True
            ).strip()
            self.root.after(0, lambda: self.set_output(translation))
            self.root.after(
                0,
                lambda: self.status_var.set(
                    f"Done | {self.current_model_label.get()} | "
                    f"{LANG_NAMES[src_lang]} -> {LANG_NAMES[tgt_lang]} | Device: {self.device}"
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


def launch_gui():
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
