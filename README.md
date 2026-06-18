# Low-Resource Neural Machine Translation (NMT) System

A modular, production-ready NMT system supporting **2 multilingual models** for translating between **Maring**, **Tangkhul**, and **English**. Built for low-resource Tibeto-Burman languages with end-to-end training, inference, and evaluation pipelines.

---

## 🎯 Features

- **2-model support**: mBART-50, NLLB-200 (distilled)
- **One training paradigm**:
  - Custom-token models (mBART, NLLB): language-code + forced BOS token
- **Direction support**: Maring ↔ English, Tangkhul ↔ English (4 directions total)
- **Evaluation metrics**: BLEU, chrF, chrF++, TER, COMET
- **GUI + CLI**: both desktop GUI and command-line interface
- **Tensorboard logging**: training/validation loss curves + epoch metrics

---

## 📦 Model Support

| Model              | Family  | Size | Custom Tokens | Prompt-Based | HF Path                                    |
| ------------------ | ------- | ---- | ------------- | ------------ | ------------------------------------------ |
| mBART-50           | mbart50 | 472M | ✅            | ❌           | `facebook/mbart-large-50-many-to-many-mmt` |
| NLLB-200 Distilled | nllb200 | 600M | ✅            | ❌           | `facebook/nllb-200-distilled-600M`         |

---

## 🗂️ Project Structure

```sh
.
├── data # Contains input data [gitignore]
├── evaluate.py # Evaluation script
├── exp # Contains experiment models and outputs [gitignore]
├── gui.py # Desktop GUI for demo using tkinter
├── loader.py # Model and Tokenization loading and logics
├── main.py # Main runner file
├── README.md
├── requirements.txt
├── train.py # Training script
├── translate.py # Translation script
└── utils.py # Utility functions
```

A modular NMT system supporting **2 multilingual models** for translating between **Maring**, **Tangkhul**, and **English**. Built for low-resource Tibeto-Burman languages with end-to-end training, inference, and evaluation pipelines.

---
