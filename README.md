# Low-Resource Neural Machine Translation (NMT) System

A modular, production-ready NMT system supporting **2 multilingual models** for translating between **Maring**, **Tangkhul**, and **English**. Built for low-resource Tibeto-Burman languages with end-to-end **training**, **inference**, **evaluation** and **inspection** pipelines.

---

## 🎯 Features

- **2-model support**: mBART-50, NLLB-200 (distilled)
- **YAML-driven registry**:
  - `conf/models.yaml` - model registry (per-model overrides like batch size)
  - `conf/directions.yaml` - language directions and corpora
  - `conf/jobs.yaml` - model × direction jobs
  - `conf/runtime.yaml` - global runtime, training, inference, GUI settings
- **Direction support**: Maring ↔ English, Tangkhul ↔ English (4 directions total)
- **Evaluation metrics**: BLEU, chrF, chrF++, TER, COMET
- **GUI + CLI**: Tkinter desktop GUI and CLI share the same config registry
- **Local-first models**:
  - One-time download via `setup_models.py` or `--setup-models`
  - Training/inference runs with `local_files_only=True`
- **Experiment tracking**:
  - `train_config.json` per experiment
  - `translations/` (TSV + `val.src.txt`, `val.ref.txt`, `val.pred.txt`)
  - `scores/` (TXT + CSV + JSON metrics)
  - `list_experiments.py` for quick summaries
- **Tensorboard logging**: training/validation loss curves + epoch metrics

---

## 📦 Model Support

| Model              | Family  | Size | Custom Tokens | Prompt-Based | HF Path                                    |
| ------------------ | ------- | ---- | ------------- | ------------ | ------------------------------------------ |
| mBART-50           | mbart50 | 472M | ✅            | ❌           | `facebook/mbart-large-50-many-to-many-mmt` |
| NLLB-200 Distilled | nllb200 | 600M | ✅            | ❌           | `facebook/nllb-200-distilled-600M`         |

---

## 🗂️ Project Structure

```bash
.
├── conf/
│   ├── models.yaml       # Model registry (per-model overrides)
│   ├── directions.yaml   # Directions & corpora (Maring/Tangkhul/English)
│   ├── jobs.yaml         # Model × direction jobs to run
│   └── runtime.yaml      # Global runtime, training, inference, GUI, eval
├── data/                 # Input TSVs [gitignored]
├── exp/                  # Experiments: models, translations, scores [gitignored]
├── models/               # Local Hugging Face models (mbart, nllb, ...)
├── src/
│   └── nmt_toolkit/
│       ├── __init__.py
│       ├── config.py     # YAML config loading & registry builders
│       ├── loader.py     # Model/tokenizer loading & resizing
│       ├── train.py      # Training pipeline (Seq2SeqTrainer)
│       ├── translate.py  # Translation / prediction generation
│       ├── evaluate.py   # BLEU/chrF/TER/COMET evaluation
│       ├── gui.py        # Tkinter GUI using same YAML registry
│       ├── main.py       # CLI entrypoint / pipeline orchestration
│       └── utils.py      # File checks, data loading, paths
├── run.py                # Entrypoint: from src.nmt_toolkit.main import main
├── setup_models.py       # One-time base model downloader (mbart + nllb)
├── list_experiments.py   # Experiment dashboard (summaries & CSV export)
├── requirements.txt
└── README.md
```

Each experiment in `exp/` is self-contained:

```bash
exp/
└── mbart-large-50-many-to-many-mmt_nng_XX_en_XX/
    ├── train_config.json             # Snapshot of job config
    ├── train.tsv                     # Training data
    ├── validation.tsv                # Validation data
    ├── translations/
    │   ├── validation.predictions.tsv  # src/pred/ref TSV
    │   ├── val.src.txt                # sources
    │   ├── val.ref.txt                # references
    │   └── val.pred.txt               # predictions
    └── scores/
        ├── evaluation_results.txt     # Human-readable metrics
        ├── evaluation_results.csv     # Tabular metrics
        └── evaluation_results.json    # JSON metrics (for tools)
```

---

## ⚙️ Setup: Environment & Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements include:

- Python 3.10+,
- PyTorch with GPU support if available,
- `transformers`, `datasets`, `sacrebleu`, etc.

---

## 📥 One-Time Model Download (Local-First)

Prepare local copies of the base models using setup_models.py:

```bash
python setup_models.py
```

This will:

- download `facebook/mbart-large-50-many-to-many-mmt`,
- download `facebook/nllb-200-distilled-600M`,
- save both into `models/...` with `save_pretrained(...)` so they are ready for offline `from_pretrained`.

(Optionally, if wired into the CLI, you can also do `python run.py --setup-models.`)

After this step, all training/translation/evaluation runs will use `local_files_only=True` against `models/mbart-large-50-many-to-many-mmt` and `models/nllb-200-distilled-600M`.

---

## 🚀 Running Training / Translation / Evaluation (CLI)

Default: run all jobs (both models × all directions) with train+translate+evaluate:

```bash
python run.py
```

Common variants:

Only mBART jobs:

```bash
python run.py --model mbart50
```

Only NLLB jobs:

```bash
python run.py --model nllb200-dist
```

Only a specific direction (e.g., Maring → English):

```bash
python run.py --task nng_to_en
```

Control phases:

```bash
python run.py --do-train --do-translate --do-evaluate
python run.py --do-train          # train only
python run.py --do-translate      # translate only (requires trained models)
python run.py --do-evaluate       # evaluate only (requires translations)
```

If no `--do-*` flags are given, all three phases run.

---

## 🧪 Evaluation Outputs

During training (`Seq2SeqTrainer`), per-epoch metrics on validation include:

- `exact_match`,
- `bleu`, `chrf`, `chrfpp`, `ter`,
- `rougeL`, `meteor`.

These are logged to TensorBoard and `trainer_state.json`. After training, `evaluate.py` recomputes metrics on the validation set and writes:

- `exp/.../scores/evaluation_results.txt`
- `exp/.../scores/evaluation_results.csv`
- `exp/.../scores/evaluation_results.json`

---

## 🧭 Experiment Dashboard (list_experiments.py)

Summarize all experiments:

```bash
python list_experiments.py
```

This prints a table of:

- experiment name and path,
- model family,
- direction,
- source/target languages,
- number of examples,
- BLEU / chrF / chrF++ / TER / COMET.

If you use the extended version, you can also:

```bash
python list_experiments.py --model-family mbart50
python list_experiments.py --direction nng_to_en
python list_experiments.py --to-csv exp_summaries.csv
```

---

## 🖥️ GUI (Tkinter)

Launch the GUI:

```bash
python run.py --gui
```

The GUI:

- reads conf/models.yaml and conf/directions.yaml,
- lists available models (mBART-50, NLLB-200),
- lists directions (Maring ↔ English, Tangkhul ↔ English),
- loads fine-tuned checkpoints from exp/...,
- shows corpus samples for quick testing,
- provides input/output text areas for interactive translation.

CLI and GUI share the same YAML registry.

---

## 📊 Tensorboard

Monitor training for a given experiment, e.g. Maring → English with mBART:

```sh
tensorboard --logdir exp/mbart-large-50-many-to-many-mmt_nng_XX_en_XX
```

You’ll see training loss, validation metrics, and best-checkpoint snapshots.

---

## 🧩 Extending the System

- **Add new models**: register them in `conf/models.yaml` and give them appropriate training overrides (batch size, max lengths).
- **Add new directions**: define them in `conf/directions.yaml` and wire jobs in `conf/jobs.yaml`.
- **Adjust defaults**: tweak `conf/runtime.yaml` for training, inference, GUI, and evaluation behavior.
- **Custom metrics**: extend `evaluate.py` or `compute_metrics` in `train.py` to add domain-specific metrics as needed.

Right now the code and configs are focused on the two robust, multilingual MT models (mBART-50, NLLB-200). If you later decide to add another model family, we can integrate it in a separate, well-isolated path so it doesn’t affect your main pipeline.

---
