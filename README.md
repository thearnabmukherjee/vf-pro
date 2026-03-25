# Indo Fashion Classifier

Image classification for **15** Indian clothing categories using **EfficientNet-B0** (ImageNet pretrained), **CUDA** training with mixed precision, **class-weighted loss**, a **FastAPI** service, and a **Streamlit** UI.

## Features

- **Transfer learning**: EfficientNet-B0 via `efficientnet_pytorch`
- **Two-stage training**: frozen backbone warmup (5 epochs), then full fine-tuning at lower LR
- **CUDA-only training**: `train.py` asserts a GPU is available
- **Mixed precision (AMP)**: `autocast` + `GradScaler`
- **Fast dataloading**: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`
- **Stratified subset**: `SAMPLES_PER_CLASS` in `train.py` (or `None` for full training set)
- **Class-weighted loss**: inverse frequency from **full** `train_data.json` when `USE_CLASS_WEIGHTS = True` (helps when the training subset is balanced per class)
- **Inference**: `/predict` returns top-3, `low_confidence` flag, and optional guidance when below a threshold
- **Streamlit**: warnings + top-3 table when the API reports low confidence

## Dataset

Expected layout under `Fashion/` (paths are relative to the project root):

```text
Fashion/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── train_data.json
├── val_data.json
└── test_data.json
```

Each line in `*_data.json` is one JSON object with at least:

- `image_path` — path relative to `Fashion/`
- `class_label` — one of 15 category names

**Git:** `Fashion/` is listed in `.gitignore` (large images). Clone the repo, then place or download the dataset locally before training.

Approximate full split sizes (reference): ~91k train / 7.5k val / 7.5k test. Class counts in the full train set are **imbalanced**; subset training uses stratified sampling per class.

## Project structure

```text
vf-pro/
├── Fashion/                 # local only (gitignored)
├── results/
│   ├── best_model.pth
│   ├── label_map.json
│   ├── training_history.json
│   ├── classification_report.txt   # after evaluate.py
│   ├── confusion_matrix.png
│   └── plots/
├── src/
│   ├── __init__.py
│   ├── dataset.py          # loaders, subset, class_weights_from_full_train
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── api.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup

### 1) Virtual environment (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Dependencies

```powershell
pip install -r requirements.txt
```

### 3) CUDA PyTorch (required for `train.py`)

Plain `pip install torch` often installs a **CPU-only** wheel on Windows. Use the CUDA index:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Expect `+cu128` in the version string and `True` for CUDA when a GPU is present.

## Training (`src/train.py`)

From the **project root**:

```powershell
python src/train.py
```

### Hyperparameters (edit in `train.py`)

| Setting | Typical value | Notes |
|--------|----------------|--------|
| `BATCH_SIZE` | `32` | |
| `EPOCHS` | `25` | Early stopping may stop sooner |
| `WARMUP_EPOCHS` | `5` | Backbone frozen; only head trains |
| `PATIENCE` | `5` | Early stopping on val loss |
| `LR` | `0.001` | Warmup phase |
| `FINETUNE_LR` | `1e-4` | After unfreezing |
| `NUM_WORKERS` | `8` | Reduce to `4`/`2` on Windows if unstable |
| `SAMPLES_PER_CLASS` | `1800` in repo | Set to `None` to use **all** training images |
| `USE_CLASS_WEIGHTS` | `True` | `CrossEntropyLoss` weights from full train counts |

Outputs:

- `results/best_model.pth` — best val accuracy checkpoint
- `results/label_map.json` — class name → index
- `results/training_history.json` — per-epoch loss/accuracy

Training uses `multiprocessing.freeze_support()` for Windows spawn compatibility.

## Preprocessing

**Train** (`dataset.py`): resize 224×224, random flip, rotation (15°), color jitter, ImageNet normalize.

**Val / test / API**: resize shorter side 256, center crop 224, normalize.

## Evaluation (`src/evaluate.py`)

```powershell
python src/evaluate.py
```

Uses **CPU or CUDA** automatically, full **test** loader, writes:

- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/plots/sample_predictions.png`
- `results/plots/loss_curve.png`, `accuracy_curve.png` if `training_history.json` exists

## API (`src/api.py`)

### Start server

From **project root**:

```powershell
uvicorn src.api:app --port 8000
```

From **`src/`**:

```powershell
uvicorn api:app --port 8000
```

Or from `src/`:

```powershell
uvicorn --app-dir .. src.api:app --port 8000
```

### Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `PREDICT_CONFIDENCE_THRESHOLD` | `0.45` | If top softmax probability is below this, `low_confidence` is `true` |

### Endpoints

- `GET /health` — `{ "status": "ok", "model_loaded": bool }`
- `GET /classes` — class list and count
- `POST /predict` — multipart field `file` (image)

### `POST /predict` response shape

```json
{
  "predicted_class": "blouse",
  "confidence": 0.77,
  "low_confidence": false,
  "confidence_threshold": 0.45,
  "top_3": [
    { "class": "blouse", "probability": 0.77 },
    { "class": "gowns", "probability": 0.12 },
    { "class": "dupattas", "probability": 0.05 }
  ],
  "message": null,
  "all_probabilities": { "...": 0.01 }
}
```

When `low_confidence` is `true`, `message` contains short guidance (clearer full-garment photo). `all_probabilities` maps every class name to a probability.

Example curl (adjust path):

```powershell
curl -X POST "http://localhost:8000/predict" -F "file=@Fashion\images\test\0.jpeg"
```

## Streamlit (`src/app.py`)

```powershell
streamlit run src/app.py
```

Open `http://localhost:8501`. Set **API URL** in the sidebar (default `http://localhost:8000`).

The UI shows prediction, confidence, a **Top 3** table from the API, top-5 bar chart, and full probabilities. If the API sets `low_confidence`, a **warning** is shown with the API `message`.

## Common issues

### `AssertionError: CUDA GPU not found`

Install CUDA PyTorch in the same venv you use for `python src/train.py` (see setup step 3).

### `ModuleNotFoundError: No module named 'src'` (Uvicorn)

Do not run `uvicorn src.api:app` from inside `src/` without fixing the path. Use `uvicorn api:app` from `src/`, or run Uvicorn from the project root.

### Training slow or workers flaky on Windows

Lower `NUM_WORKERS` in `train.py`. Ensure the dataset lives on a fast disk (SSD).

## Requirements

See `requirements.txt`: `torch`, `torchvision`, `efficientnet_pytorch`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `Pillow`, `tqdm`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `streamlit`, `pandas`, `requests`.

## Tech stack

| Piece | Role |
|-------|------|
| PyTorch / torchvision | Training & inference |
| efficientnet_pytorch | EfficientNet-B0 backbone |
| scikit-learn | Metrics |
| matplotlib / seaborn | Plots |
| FastAPI / uvicorn | REST API |
| Streamlit / pandas / requests | Demo UI |
