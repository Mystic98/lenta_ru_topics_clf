# Lenta.ru news topic classifier

A small NLP project for **multi-class topic classification** of Russian news texts (Lenta.ru).
The project compares several text vectorization approaches and trains a **Logistic Regression** classifier inside an sklearn `Pipeline`.

## Goal
Compare different vectorizers and their performance on the Lenta.ru dataset.

## Pipeline overview
1) Build a processed dataset from raw Lenta.ru data
2) (Optionally) build a token corpus for custom vectorizers
3) Train `Pipeline(vectorizer → LogisticRegression)`
4) Evaluate and save artifacts (model + label map + reports)
5) Load the saved `.joblib` pipeline for inference

## Supported vectorizers
- `count` — `CountVectorizer`
- `tfidf` — `TfidfVectorizer`
- `w2v` / `w2v_weighted` — Word2Vec-based vectorizer trained on the project corpus, with optional TF‑IDF weighting
- `navec` / `navec_weighted` — pretrained Navec vectorizer, with optional TF‑IDF weighting

### Notebooks:
- data overview and preprocessing: [data_overview.ipynb](notebooks/data_overview.ipynb)
- custom vectorizer implementation: [custom_vectorizer.ipynb](notebooks/custom_vectorizer.ipynb)
- pipeline training and evaluation: [training_and_evaluation.ipynb](notebooks/training_and_evaluation.ipynb)

## Results
Best performing vectorizer: `count`

Metrics:
- `n_samples`: 20000
- `accuracy`: 0.81835
- `f1_macro`: 0.77752
- `f1_micro`: 0.81835
- `f1_weighted`: 0.81799

---

## Project structure

```text
lenta_ru_topic_clf/
├── src/
│   └── lenta_topic/
│       ├── __init__.py
│       ├── config.py          # constants: paths, random_state, vectorizer params, etc.
│       ├── data.py            # loading, filtering, sampling, label mapping
│       ├── vectorizers.py     # custom vectorizers
│       ├── train.py           # pipeline training utilities
│       └── evaluate.py        # evaluation utilities (metrics, reports, confusion matrix)
├── scripts/
│   ├── build_dataset.py       # build processed dataset + label_map.json
│   ├── train_model.py         # train pipeline and save .joblib model
│   └── evaluate_model.py      # run evaluation and save reports
├── data/
│   ├── raw/
│   │   ├── lenta-ru-news.csv.gz
│   │   └── navec_news_v1_1B_250K_300d_100q.tar
│   └── processed/
│       ├── dataset.csv
│       ├── train_dataset.csv
│       ├── test_dataset.csv
│       └── tokens.csv
├── models/
│   ├── label_map.json
│   └── pipeline_<vec_name>.joblib
├── reports/
│   ├── figures/
│   └── metrics/
├── notebooks/
│   ├── data_overview.ipynb
│   ├── custom_vectorizer.ipynb
│   └── training_and_evaluation.ipynb
├── pyproject.toml
└── README.md
```

Folder roles:
- `src/lenta_topic/` — importable modules for topic classification
- `scripts/` — preprocessing, training, evaluation
- `data/raw/` — downloaded datasets (Lenta.ru + navec embedding)
- `data/processed/` — processed datasets produced by scripts
- `models/` — trained pipelines and label mapping
- `reports/` — evaluation outputs (metrics tables, figures)
- `notebooks/` — analysis and explanations

---

## Generated files
- After running `python scripts/build_dataset.py`:
  - `data/processed/dataset.csv`
  - `models/label_map.json`
- After running `python scripts/build_tokens.py`:
  - `data/processed/tokens.csv`
  - `data/processed/dataset.csv` will be updated
- After running `python scripts/train_model.py`:
  - `models/pipeline_<vec_name>.joblib`
  - `data/processed/train_dataset.csv`
  - `data/processed/test_dataset.csv`
- After running `python scripts/evaluate_model.py`:
  - `reports/figures/confusion_matrix_<vec_name>.png`
  - `reports/metrics/metrics_<vec_name>.csv`

---

## Setup

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

### 2) Install dependencies (editable)
```bash
pip install -e .
```

### 3) Download spaCy model
```bash
python -m spacy download ru_core_news_sm
```

### 4) Download Navec pretrained embedding to `data/raw/`
```bash
mkdir -p data/raw
curl -L -o data/raw/navec_news_v1_1B_250K_300d_100q.tar \
  https://storage.yandexcloud.net/natasha-navec/packs/navec_news_v1_1B_250K_300d_100q.tar
```

### 5) Download Lenta.ru dataset to `data/raw/`
```bash
mkdir -p data/raw
curl -L -o data/raw/lenta-ru-news.csv.gz \
  https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz
```

---

## Running the project

Build dataset:
```bash
python scripts/build_dataset.py
```

Train model (you will be prompted for a vectorizer name):
```bash
python scripts/train_model.py
```

Evaluate model (you will be prompted for the same vectorizer name that was used for training):
```bash
python scripts/evaluate_model.py
```

Vectorizer names:
- `count`, `tfidf`, `w2v`, `w2v_weighted`, `navec`, `navec_weighted`
  
