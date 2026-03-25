import os
import json
import io
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

try:
    # Works when started from project root: uvicorn src.api:app
    from src.model import build_model
except ModuleNotFoundError:
    # Works when started from src/: uvicorn api:app
    from model import build_model

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

app = FastAPI(title="Indo Fashion Classifier API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
label_map = None
idx_to_label = None
device = None


@app.on_event("startup")
def load_model():
    global model, label_map, idx_to_label, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map_path = os.path.join(RESULTS_DIR, "label_map.json")
    if not os.path.exists(label_map_path):
        raise RuntimeError(f"Label map not found at {label_map_path}. Run training first.")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    num_classes = len(label_map)
    model = build_model(num_classes=num_classes, dropout=0.3)

    ckpt_path = os.path.join(RESULTS_DIR, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f"Model checkpoint not found at {ckpt_path}. Run training first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"Model loaded on {device} with {num_classes} classes.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/classes")
def get_classes():
    return {"classes": list(label_map.keys()), "count": len(label_map)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image.")

    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    pred_idx = probs.argmax().item()
    predicted_class = idx_to_label[pred_idx]
    confidence = probs[pred_idx].item()

    all_probabilities = {
        idx_to_label[i]: round(probs[i].item(), 6) for i in range(len(idx_to_label))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "all_probabilities": all_probabilities,
    }
