# Indo Fashion Classifier

Image classification for 15 Indian clothing categories using transfer learning with EfficientNet-B0, CUDA training, FastAPI inference, and a Streamlit UI.

## Features

- EfficientNet-B0 pretrained on ImageNet
- Two-stage training (warmup + full fine-tuning)
- CUDA-only training (fails fast if GPU is unavailable)
- Mixed precision (AMP) for faster GPU training
- Parallel dataloading with worker/prefetch support
- Optional stratified training subset (`samples_per_class`)
- FastAPI prediction service
- Streamlit frontend for interactive testing

## Dataset

Dataset root is expected at `Fashion/` with this structure:

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

Each line in `*_data.json` is JSON with at least:

- `image_path` (relative path under `Fashion/`)
- `class_label`

The project uses 15 classes total.

## Project Structure

```text
vf-pro/
├── Fashion/
├── results/
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── api.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

### 1) Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

### 3) Install CUDA PyTorch in the venv (required for training)

This project's training script enforces CUDA.

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:
- torch version ending with `+cu128`
- `True` for CUDA availability
- your NVIDIA GPU name

## Training

Run from project root:

```powershell
python src/train.py
```

### Current training configuration (`src/train.py`)

- `BATCH_SIZE = 32`
- `EPOCHS = 25`
- `WARMUP_EPOCHS = 5`
- `PATIENCE = 5`
- `LR = 1e-3` (warmup)
- `FINETUNE_LR = 1e-4` (after unfreeze)
- `NUM_WORKERS = 8`
- `SAMPLES_PER_CLASS = 500` (set `None` for full dataset)

### What makes training faster now

- CUDA-only device selection
- AMP (`autocast` + `GradScaler`)
- `cudnn.benchmark = True`
- non-blocking host-to-device transfers
- parallel dataloader workers + prefetch + persistent workers
- optional stratified subset sampling

### Notes

- If Windows dataloader stability/performance is inconsistent, reduce `NUM_WORKERS` to `4` or `2`.
- With `SAMPLES_PER_CLASS = 500`, training uses ~7,500 training images (500 x 15 classes).

## Evaluation

Run:

```powershell
python src/evaluate.py
```

This script:
- loads `results/best_model.pth`
- evaluates on full test split
- prints overall accuracy and classification report
- saves report and plots

Outputs include:
- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/plots/sample_predictions.png`
- `results/plots/loss_curve.png` and `results/plots/accuracy_curve.png` (if training history exists)

## Inference API (FastAPI)

File: `src/api.py`

### Start API

From project root:

```powershell
uvicorn src.api:app --port 8000
```

From `src/` folder:

```powershell
uvicorn api:app --port 8000
```

Alternative from `src/` folder:

```powershell
uvicorn --app-dir .. src.api:app --port 8000
```

### Endpoints

- `GET /health` -> API/model status
- `GET /classes` -> class names
- `POST /predict` -> image classification

Example:

```powershell
curl -X POST "http://localhost:8000/predict" -F "file=@Fashion\images\test\0.jpeg"
```

## Streamlit App

Run:

```powershell
streamlit run src/app.py
```

Open `http://localhost:8501` and upload an image.

The app calls the API (default `http://localhost:8000`) and shows:
- predicted class
- confidence
- top-5 probabilities
- full probability table

## Common Issues and Fixes

### `AssertionError: CUDA GPU not found`

Cause: CPU-only PyTorch installed in venv.

Fix:

```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### `ModuleNotFoundError: No module named 'src'` when starting Uvicorn

Cause: running `uvicorn src.api:app` from inside `src/`.

Fix: use one of:
- `uvicorn api:app --port 8000` (from `src/`)
- `uvicorn --app-dir .. src.api:app --port 8000` (from `src/`)
- run from project root and use `uvicorn src.api:app --port 8000`

### Training is still slow on GPU

- Lower `SAMPLES_PER_CLASS` further for quick experiments
- Reduce image augmentation intensity
- Tune `NUM_WORKERS` (`2`, `4`, `8`) for your machine
- Ensure no other heavy GPU workloads are running

## Requirements

See `requirements.txt`:
- torch, torchvision
- efficientnet_pytorch
- scikit-learn
- matplotlib, seaborn
- Pillow
- tqdm
- fastapi, uvicorn[standard], python-multipart
- streamlit
- requests
# Indo Fashion Classifier

Image classification of 15 categories of Indian fashion using transfer learning with **EfficientNet-B0**.

## Approach

- **Transfer Learning**: EfficientNet-B0 pretrained on ImageNet
- **Warmup + Fine-tune**: Classifier head trained for 5 epochs with frozen backbone, then all layers unfrozen for full fine-tuning at a lower learning rate
- **Regularization**: Dropout (0.3) before the final classifier layer
- **Early Stopping**: Training halts if validation loss does not improve for 5 consecutive epochs

## Dataset

The **Indo Fashion** dataset with 15 garment categories:

| Category | Train Count |
|---|---|
| women_kurta | 11,694 |
| saree | 10,791 |
| blouse | 9,174 |
| leggings_and_salwars | 7,787 |
| kurta_men | 6,951 |
| dupattas | 6,587 |
| nehru_jackets | 6,491 |
| lehenga | 5,753 |
| gowns | 5,211 |
| petticoats | 4,441 |
| dhoti_pants | 4,145 |
| palazzos | 3,375 |
| mojaris_women | 3,228 |
| sherwanis | 2,992 |
| mojaris_men | 2,546 |

**Total**: 91,166 train / 7,500 val / 7,500 test images.

Data files are JSONL format with `image_path` and `class_label` fields.

## Preprocessing

**Training augmentations:**
- Resize to 224x224
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Normalize with ImageNet statistics

**Validation / Test:**
- Resize to 256, center crop to 224
- Normalize with ImageNet statistics

## Training

| Parameter | Value |
|---|---|
| Model | EfficientNet-B0 |
| Optimizer | Adam |
| Learning Rate | 0.001 (warmup), 0.0001 (fine-tune) |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=2) |
| Batch Size | 32 |
| Epochs | 25 (max) |
| Warmup Epochs | 5 |
| Early Stopping | Patience = 5 |

## Results

After training, results are saved in the `results/` directory:
- `best_model.pth` — Best model checkpoint
- `training_history.json` — Per-epoch metrics
- `classification_report.txt` — Per-class precision, recall, F1
- `confusion_matrix.png` — 15x15 heatmap
- `plots/loss_curve.png` — Train vs val loss
- `plots/accuracy_curve.png` — Train vs val accuracy
- `plots/sample_predictions.png` — Grid of example predictions

## API

A **FastAPI** prediction endpoint is provided in `src/api.py`.

**Endpoints:**
- `GET /health` — Check if model is loaded
- `GET /classes` — List all 15 class names
- `POST /predict` — Upload an image, returns predicted class and confidence scores

## Streamlit App

An interactive **Streamlit** frontend is provided in `src/app.py`. Upload an image and get instant predictions with confidence bar charts.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

This saves the best checkpoint to `results/best_model.pth` and training history to `results/training_history.json`.

### 3. Evaluate on test set

```bash
python src/evaluate.py
```

Prints accuracy, classification report, and saves confusion matrix and plots.

### 4. Start the API server

```bash
uvicorn src.api:app --port 8000
```

Test with:

```bash
curl -X POST http://localhost:8000/predict -F "file=@path/to/image.jpg"
```

### 5. Launch the Streamlit app

```bash
streamlit run src/app.py
```

Open the browser at `http://localhost:8501`, upload an image, and see the prediction.

## Project Structure

```
indo-fashion-classifier/
├── Fashion/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── train_data.json
│   ├── val_data.json
│   └── test_data.json
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   ├── api.py
│   └── app.py
├── results/
│   ├── plots/
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch + torchvision | Training framework |
| EfficientNet-B0 | Pretrained backbone |
| scikit-learn | Metrics and confusion matrix |
| matplotlib + seaborn | Plots and visualizations |
| Pillow | Image loading |
| tqdm | Progress bars |
| FastAPI + uvicorn | REST API for predictions |
| Streamlit | Interactive web frontend |
