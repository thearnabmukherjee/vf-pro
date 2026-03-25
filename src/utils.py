import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def plot_training_curves(history_path, save_dir):
    """Plot train vs val loss and accuracy curves from a saved JSON history."""
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", markersize=4)
    ax.plot(epochs, history["val_acc"], "r-o", label="Val Accuracy", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """Compute and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _denormalize(tensor):
    """Convert a normalized image tensor back to displayable [0, 1] range."""
    img = tensor.permute(1, 2, 0).numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def plot_sample_predictions(images, preds, trues, label_names, save_path):
    """Display a 4x4 grid of images with predicted vs actual labels.

    Args:
        images: list of image tensors (C, H, W), at most 16.
        preds: array of predicted label indices.
        trues: array of true label indices.
        label_names: list mapping index -> class name.
        save_path: where to save the figure.
    """
    n = min(len(images), 16)
    rows, cols = 4, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 16))

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols]
        if i < n:
            img = _denormalize(images[i])
            ax.imshow(img)
            pred_name = label_names[preds[i]]
            true_name = label_names[trues[i]]
            correct = preds[i] == trues[i]
            color = "green" if correct else "red"
            ax.set_title(f"P: {pred_name}\nT: {true_name}", fontsize=9, color=color)
        ax.axis("off")

    fig.suptitle("Sample Predictions (Green = Correct, Red = Wrong)", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
