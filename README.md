# PCXP Pneumonia Detection – MLOps Pipeline

A complete MLOps pipeline for pneumonia detection from chest X-ray images with a **stacked ensemble** (3 base models + logistic regression meta-learner with clinical features), from training to production deployment via REST API and Docker.

---

## Project Overview

This project implements a binary lung-opacity classifier using a **stacked ensemble** trained on the **RSNA Pneumonia Detection Challenge** dataset (`stage2_train_metadata.csv`). The ensemble combines three diverse base models and a meta-learner that also consumes clinical metadata (age, sex, imaging position).

| Stage | Phase | Description |
|-------|-------|-------------|
| 1 | EDA | Explore raw data, understand structure, identify issues |
| 2 | Preprocessing | Clean & transform images (resize, normalize, augment) |
| 3 | Modelling | Train & optimise ResNet18 + TinyCNN ensemble with stacking |
| 4 | Code structure | Modular Python package (`python/pcxp_mlops/`) |
| 5 | Tests | Unit & functional tests for pipeline components |
| 6 | DVC | Data versioning with DVC pipeline |
| 7 | MLflow | Experiment tracking – hyperparams, metrics, artefacts |
| 8 | API | REST service exposing single-model & ensemble endpoints |
| 9 | Docker | Containerised deployment |
| 10 | GitHub | Version control, structured repo, documentation |

---

## Showcase

![Web Interface](./images/showcaseInterface.png)

The API serves an interactive web interface at `/` where you can upload a chest X-ray, adjust the confidence threshold, and view the prediction alongside model version and request metadata.

---

## Project Structure

```
.
├── python/
│   ├── api.py                      # ASGI entrypoint (single & ensemble)
│   ├── train.py                    # Single-model training CLI
│   ├── evaluate.py                 # Single-model evaluation CLI
│   ├── train_ensemble.py           # Ensemble training CLI
│   ├── evaluate_ensemble.py        # Ensemble evaluation CLI
│   ├── ml_pipeline.py              # Legacy compatibility helpers
│   └── pcxp_mlops/
│       ├── api.py                  # FastAPI app factories & endpoints
│       ├── config.py               # Centralised paths & constants
│       ├── data_loader.py          # RSNA CSV dataset + transforms
│       ├── evaluation.py           # Evaluation & MLflow logging
│       ├── metadata.py             # Model metadata persistence
│       ├── metrics.py              # Metric helpers
│       ├── model.py                # ResNet18, TinyCNN, wrappers, ensemble
│       ├── predict.py              # Inference services
│       ├── training.py             # Training loops (single + ensemble)
│       └── static/                 # Web UI (HTML, CSS, JS)
├── models/
│   ├── metadata.json               # Single-model version & metrics
│   ├── classes.json                # Class label order
│   ├── model.pth                   # Single ResNet18 weights
│   └── ensemble/                   # Ensemble artefacts
│       ├── metadata.json
│       ├── meta_model.pkl
│       ├── resnet18.pth
│       ├── tiny_cnn.pth
│       └── resnet18_low_lr.pth
├── tests/
│   ├── test_api.py
│   └── test_ml_pipeline.py
├── data/
│   ├── stage2_train_metadata.csv   # RSNA metadata (DVC-tracked)
│   ├── stage2_test_metadata.csv    # Test metadata (DVC-tracked)
│   └── Training/Images/            # PNG images (DVC-tracked)
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                        # DVC pipeline definition
├── requirements.txt
└── roadmap.txt
```

---

## Dataset

The pipeline uses the RSNA Pneumonia Detection Challenge dataset in CSV-driven format:

- **`data/stage2_train_metadata.csv`** – 30,227 rows, 26,684 unique patients; contains `patientId`, `Target` (0/1), `age`, `sex`, `position` columns.
- **`data/Training/Images/{patientId}.png`** – 1024×1024 grayscale chest X-rays.

Images are referred to by `patientId` from the CSV; the old folder-based layout (`NORMAL/`, `PNEUMONIA/` subdirectories) is no longer used.

Training uses an 80/20 random split from the training CSV (the test CSV lacks a `Target` column).

---

## Models

### Single Model
- **ResNet18** (pretrained on ImageNet) – entrypoints `python.train`, `python.evaluate`.

### Stacked Ensemble
Three diverse base models retrained on a held-out 80% split and combined via logistic regression:

| Model | Architecture | LR | Epochs | Purpose |
|-------|-------------|----|--------|---------|
| `resnet18` | ResNet18 (pretrained) | 1e-3 | 5 | High-capacity baseline |
| `tiny_cnn` | TinyCNN (scratch) | 1e-3 | 8 | Lightweight diversity |
| `resnet18_low_lr` | ResNet18 (pretrained) | 1e-4 | 5 | Regularised variant |

**Meta-learner input** (6 features): `[prob_resnet18, prob_tiny_cnn, prob_resnet18_low_lr, age_norm, sex_enc, pos_enc]`

- `age_norm` = age / 100
- `sex_enc` = 0 (male), 1 (female)
- `pos_enc` = 0 (AP), 1 (PA)

---

## Local Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the single model

```bash
python -m python.train
```

Press **Ctrl+C** at any point during training to save the current weights and stop early.

### Evaluate the single model

```bash
python -m python.evaluate
```

### Train the ensemble (base models + meta-learner)

```bash
python -m python.train_ensemble
```

This runs two phases:
1. **Phase 1** – trains all 3 base models on the full dataset (produces `models/ensemble/{name}.pth`)
2. **Phase 2** – retrains each base for `max(epochs//2, 2)` epochs on an 80% inner fold, generates validation predictions, fits `LogisticRegression` on `[probs, age, sex, position]` (produces `models/ensemble/meta_model.pkl`)

Press **Ctrl+C** during any training loop to save partial progress and stop.

### Evaluate the ensemble

```bash
python -m python.evaluate_ensemble
```

### Run the single-model API

```bash
uvicorn python.api:app --host 0.0.0.0 --port 8000
```

### Run the ensemble API

```bash
uvicorn python.api:ensemble_app --host 0.0.0.0 --port 8000
```

### Run tests

```bash
python -m unittest discover -s tests
```

---

## API Endpoints

### Single-Model API (`create_app`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web interface |
| GET | `/health` | Status & model version |
| GET | `/model-info` | Full metadata & metrics |
| POST | `/predict` | Classify a chest X-ray |

**`POST /predict`** request:
```json
{
  "image_path": "data/PCXP/test/PNEUMONIA/person100_bacteria_475.jpeg"
}
```

Alternative: `image_base64` (base64 string), optional `threshold` (0.0–1.0).

**Response:**
```json
{
  "request_id": "7f0d5d1c-8c8d-44c4-8687-7e4f0d7d6f13",
  "predicted_class": "PNEUMONIA",
  "predicted_index": 1,
  "probability": 0.97,
  "threshold": 0.87,
  "model_version": "pcxp-resnet18-v1"
}
```

### Ensemble API (`create_ensemble_app`)

Same endpoints plus optional clinical metadata on `POST /predict`:

```json
{
  "image_path": "...",
  "age": 55,
  "sex": "M",
  "position": "AP"
}
```

`age`, `sex`, and `position` are forwarded to the meta-learner for a prediction that incorporates patient context.

---

## Checkpoint / Early Stop

All training functions wrap the epoch loop in `try/except KeyboardInterrupt`. Press **Ctrl+C** at any time:

```
^C
  resnet18 | Interrupted — saving partial progress at epoch 3/5.
  resnet18 | Total: 142.3s (47.4s/epoch)
```

The model weights from the last completed epoch are saved to disk before the process exits normally.

---

## DVC Pipeline

The dataset is versioned with DVC. The pipeline (`dvc.yaml`) defines four stages:

| Stage | Produces | Description |
|-------|----------|-------------|
| `train` | `model.pth`, `classes.json`, `metadata.json` | Single ResNet18 |
| `evaluate` | `metadata.json` (updated) | Metrics on held-out split |
| `train_ensemble` | `ensemble/*.pth`, `ensemble/meta_model.pkl`, `ensemble/metadata.json` | Full ensemble |
| `evaluate_ensemble` | `ensemble/metadata.json` (updated) | Ensemble metrics |

Reproduce the full pipeline:

```bash
dvc repro
```

---

## MLflow

Experiments are logged during evaluation. Launch the UI to compare runs:

```bash
mlflow ui
```

Both single-model and ensemble evaluations create separate experiments (`Pneumonia_Model_Evaluation` and `Pneumonia_Ensemble_Evaluation`).

---

## Docker

### Build the image

```bash
docker build -t pcxp-api .
```

### Run with Docker

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models" pcxp-api
```

### Run with Docker Compose

```bash
docker compose up --build
```

The Compose setup mounts `./models` as a volume, exposes port 8000, and sets the required environment variables.

---

## Tech Stack

- **PyTorch** & **Torchvision** – deep learning framework
- **FastAPI** & **Uvicorn** – REST API
- **DVC** – data versioning
- **MLflow** – experiment tracking
- **Docker** & **Docker Compose** – containerisation
- **scikit-learn** – logistic regression meta-learner, metrics
- **joblib** – meta-model serialisation
- **pandas** – CSV metadata parsing
- **unittest** – automated testing
