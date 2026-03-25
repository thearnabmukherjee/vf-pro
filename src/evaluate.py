import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import get_dataloaders
from src.model import build_model
from src.utils import plot_confusion_matrix, plot_sample_predictions, plot_training_curves

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Fashion")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
NUM_WORKERS = 4
BATCH_SIZE = 32


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_images = []
    collected_samples = 0

    for images, labels in tqdm(loader, desc="Test inference"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if collected_samples < 16:
            need = 16 - collected_samples
            all_images.extend(images[:need].cpu())
            collected_samples += min(need, images.size(0))

    return np.array(all_preds), np.array(all_labels), all_images[:16]


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load label map ───────────────────────────────────────────────────────
    label_map_path = os.path.join(RESULTS_DIR, "label_map.json")
    with open(label_map_path, "r") as f:
        label_map = json.load(f)

    idx_to_label = {v: k for k, v in label_map.items()}
    label_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    num_classes = len(label_map)

    # ── Load model ───────────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes, dropout=0.3)
    ckpt_path = os.path.join(RESULTS_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    print(f"Loaded checkpoint from {ckpt_path}")

    # ── Data ─────────────────────────────────────────────────────────────────
    _, _, test_loader, _ = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
    )

    # ── Inference ────────────────────────────────────────────────────────────
    preds, labels, sample_images = run_inference(model, test_loader, device)

    # ── Metrics ──────────────────────────────────────────────────────────────
    overall_acc = accuracy_score(labels, preds)
    print(f"\nOverall Test Accuracy: {overall_acc:.4f} ({overall_acc * 100:.2f}%)\n")

    report = classification_report(labels, preds, target_names=label_names, digits=4)
    print("Classification Report:\n")
    print(report)

    report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Overall Test Accuracy: {overall_acc:.4f}\n\n")
        f.write(report)
    print(f"Report saved to {report_path}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(labels, preds, label_names, cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # ── Sample predictions ───────────────────────────────────────────────────
    sample_preds = preds[:16]
    sample_labels = labels[:16]
    sample_path = os.path.join(PLOTS_DIR, "sample_predictions.png")
    plot_sample_predictions(sample_images, sample_preds, sample_labels, label_names, sample_path)
    print(f"Sample predictions saved to {sample_path}")

    # ── Training curves ──────────────────────────────────────────────────────
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    if os.path.exists(history_path):
        plot_training_curves(history_path, PLOTS_DIR)
        print(f"Training curves saved to {PLOTS_DIR}")
    else:
        print(f"Warning: {history_path} not found, skipping training curves.")


if __name__ == "__main__":
    main()
