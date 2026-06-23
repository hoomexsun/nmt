# Low-Resource Neural Machine Translation (NMT) System

A modular, production-ready NMT system supporting **2 multilingual models** for translating between **Maring**, **Tangkhul**, and **English**. Built for low-resource Tibeto-Burman languages with end-to-end **training**, **inference**, **evaluation**, and **inspection** pipelines.

---

## 🎯 Features

- **2-model support**: mBART-50, NLLB-200 (distilled)
- **YAML-driven registry**:
  - `conf/models.yaml` – model registry (per-model overrides like batch size)
  - `conf/directions.yaml` – language directions and corpora
  - `conf/jobs.yaml` – model × direction jobs
  - `conf/runtime.yaml` – global runtime, training, inference, GUI settings
- **Direction support**: Maring ↔ English, Tangkhul ↔ English (4 directions total)
- **Pre-split corpora**:
  - Fixed test sizes per corpus (Maring: 1000; Tangkhul: 800),
  - From the remaining data: 90% train / 10% validation,
  - Shared across all models and directions for reproducible experiments.
- **Evaluation metrics**:
  - Training-time: `exact_match`, BLEU, chrF, chrF++, TER, ROUGE-L, METEOR
  - Final evaluation: BLEU, chrF/chrF++, TER, optional COMET
  - **Best checkpoint selected by chrF** (not exact match) on the validation set.
- **GUI + CLI**: Tkinter desktop GUI and CLI share the same config registry
- **Local-first models**:
  - One-time download via `setup_models.py` or `--setup-models`
  - Training/inference runs with `local_files_only=True`
- **Experiment tracking**:
  - `train_config.json` per experiment
  - `translations/` (val + test TSVs and flat text files)
  - `scores/` (val + test metrics as TXT + CSV + JSON)
  - `list_experiments.py` for quick summaries
- **Tensorboard logging**: training/validation loss curves + per-epoch metrics

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
├── data/                 # Original TSVs + pre-split train/val/test [gitignored]
├── exp/                  # Experiments: models, translations, scores [gitignored]
├── models/               # Local Hugging Face models (mbart, nllb, ...)
├── src/
│   └── nmt_toolkit/
│       ├── __init__.py
│       ├── config.py        # YAML config loading & registry builders
│       ├── loader.py        # Model/tokenizer loading & resizing
│       ├── train.py         # Training pipeline (Seq2SeqTrainer, chrF-based best)
│       ├── translate.py     # Translation / prediction generation (val + test)
│       ├── evaluate.py      # chrF/BLEU/TER/COMET evaluation (val + test)
│       ├── split_corpus.py  # One-time splitter: train/val/test per corpus
│       ├── gui.py           # Tkinter GUI using same YAML registry
│       ├── main.py          # CLI entrypoint / pipeline orchestration
│       └── utils.py         # File checks, data loading, paths
├── run.py                # Entrypoint: calls src.nmt_toolkit.main.main()
├── setup_models.py       # One-time base model downloader (mbart + nllb)
├── list_experiments.py   # Experiment dashboard (summaries & CSV export)
├── requirements.txt
└── README.md
```

### Data layout

- Original parallel corpora:
  - `data/engLatn_nngLatn.tsv` (English–Maring)
  - `data/engLatn_nmfLatn.tsv` (English–Tangkhul)
- Pre-split files (created by `split_corpus.py`):
  - `data/engLatn_nngLatn/train.tsv`
  - `data/engLatn_nngLatn/validation.tsv`
  - `data/engLatn_nngLatn/test.tsv`
  - `data/engLatn_nmfLatn/train.tsv`
  - `data/engLatn_nmfLatn/validation.tsv`
  - `data/engLatn_nmfLatn/test.tsv`

These splits are shared by all directions and models, ensuring consistent train/validation/test sets across experiments.

---

## 📁 Experiment Layout

Each experiment in `exp/` is self-contained:

```bash
exp/
└── mbart-large-50-many-to-many-mmt_nng_XX_en_XX/
    ├── train_config.json             # Snapshot of job config
    ├── train.tsv                     # Training data (copy of pre-split)
    ├── validation.tsv                # Validation data (copy of pre-split)
    ├── test.tsv                      # Optional: test data copy (if desired)
    ├── translations/
    │   ├── validation.predictions.tsv  # validation src/pred/ref TSV
    │   ├── test.predictions.tsv        # test src/pred/ref TSV
    │   ├── val.src.txt                 # validation sources
    │   ├── val.ref.txt                 # validation references
    │   ├── val.pred.txt                # validation predictions
    │   ├── test.src.txt                # test sources
    │   ├── test.ref.txt                # test references
    │   └── test.pred.txt               # test predictions
    └── scores/
        ├── validation_results.txt      # Validation metrics (chrF/BLEU/etc.)
        ├── validation_results.csv
        ├── validation_results.json
        ├── test_results.txt            # Test metrics (chrF/BLEU/etc.)
        ├── test_results.csv
        └── test_results.json
```

---

## ⚙️ Setup: Environment & Dependencies

Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements include:

- Python 3.10+
- PyTorch with GPU support if available
- `transformers`, `datasets`, `sacrebleu`, `comet-ml` (optional), etc.

---

## 📥 One-Time Model Download (Local-First)

Prepare local copies of the base models using `setup_models.py`:

```bash
python setup_models.py
```

This will:

- download `facebook/mbart-large-50-many-to-many-mmt`,
- download `facebook/nllb-200-distilled-600M`,
- save both into `models/...` with `save_pretrained(...)`.

After this step, all training/translation/evaluation runs use `local_files_only=True` against:

- `models/mbart-large-50-many-to-many-mmt`
- `models/nllb-200-distilled-600M`

---

## 🔪 Data Splitting (Train / Validation / Test)

Before training, each corpus is pre-split into train, validation, and test sets:

- **Fixed test sizes**:
  - Maring (`engLatn_nngLatn`): 1000 examples.
  - Tangkhul (`engLatn_nmfLatn`): 800 examples.
- **From the remaining data**:
  - 10% is used as **validation**.
  - 90% is used as **training**.

Splitting is handled by `src/nmt_toolkit/split_corpus.py`, which reads `conf/directions.yaml` and `conf/runtime.yaml` (for the random seed) and writes:

- `data/engLatn_nngLatn/{train,validation,test}.tsv`
- `data/engLatn_nmfLatn/{train,validation,test}.tsv`

You can run the splitter explicitly:

```bash
python -m src.nmt_toolkit.split_corpus
```

Or, if your `run.py` calls `split_corpus` first, simply run `python run.py` to perform splitting and then training/translation/evaluation in one go.

---

## 🚀 Running Training / Translation / Evaluation (CLI)

Default: run all jobs (both models × all directions) with train + translate + evaluate:

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

Only a specific direction (e.g. Maring → English):

```bash
python run.py --task nng_to_en
```

Control phases explicitly:

```bash
python run.py --do-train --do-translate --do-evaluate
python run.py --do-train          # train only
python run.py --do-translate      # translate only (requires trained models)
python run.py --do-evaluate       # evaluate only (requires translations)
```

If no `--do-*` flags are given, all three phases run by default.

The high-level pipeline per job is:

1. **Train**:
   - Loads pre-split `train.tsv` and `validation.tsv`.
   - Trains using `Seq2SeqTrainer`.
   - Selects the best checkpoint by **validation chrF** (`metric_for_best_model="chrf"`).
2. **Translate**:
   - Generates predictions for both validation and test:
     - `translations/validation.predictions.tsv`
     - `translations/test.predictions.tsv`
   - Saves flat text files for quick inspection (`val.*.txt`, `test.*.txt`).
3. **Evaluate**:
   - Computes metrics on both validation and test and writes:
     - `scores/validation_results.{txt,csv,json}`
     - `scores/test_results.{txt,csv,json}`

---

## 🧪 Evaluation Outputs

During training (`Seq2SeqTrainer`), per-epoch metrics on the **validation** split include:

- `exact_match`
- `bleu`, `chrf`, `chrfpp`, `ter`
- `rougeL`, `meteor`

The best checkpoint is selected by **validation chrF** (`metric_for_best_model="chrf"`), which tends to correlate better with human judgments in low-resource, morphologically rich languages than exact match or BLEU alone.

After training and translation, `evaluate.py` computes metrics for:

- **Validation**:
  - Predictions: `translations/validation.predictions.tsv`
  - Metrics: `scores/validation_results.{txt,csv,json}`
- **Test**:
  - Predictions: `translations/test.predictions.tsv`
  - Metrics: `scores/test_results.{txt,csv,json}`

Metrics include BLEU, chrF/chrF++, TER, and optional COMET when available.

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
- BLEU / chrF / chrF++ / TER / COMET (where available).

You can filter or export:

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

- reads `conf/models.yaml` and `conf/directions.yaml`,
- lists available models (mBART-50, NLLB-200),
- lists directions (Maring ↔ English, Tangkhul ↔ English),
- loads fine-tuned checkpoints from `exp/...`,
- shows corpus samples for quick testing,
- provides input/output text areas for interactive translation.

CLI and GUI share the same YAML registry.

---

## 📊 Tensorboard

Monitor training for a given experiment, e.g. Maring → English with mBART:

```bash
tensorboard --logdir exp/mbart-large-50-many-to-many-mmt_nng_XX_en_XX
```

You’ll see training loss, validation metrics (including chrF, BLEU, etc.), and best-checkpoint snapshots.

---

## 🧩 Extending the System

- **Add new models**:
  - Register them in `conf/models.yaml`.
  - Provide training overrides (batch size, max lengths).
- **Add new directions**:
  - Define them in `conf/directions.yaml`.
  - Wire jobs in `conf/jobs.yaml`.
- **Adjust defaults**:
  - Tweak `conf/runtime.yaml` for training, inference, GUI, and evaluation behaviour.
- **Custom metrics**:
  - Extend `evaluate.py` or `compute_metrics` in `train.py` to add domain-specific metrics as needed.

The code and configs are focused on two robust, multilingual MT models (mBART-50, NLLB-200). Additional model families can be integrated via the same YAML-driven registry without breaking the main pipeline.
