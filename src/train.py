import os
import sys
import json
import time
import copy
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import get_dataloaders, class_weights_from_full_train
from src.model import build_model, unfreeze_model

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 32
LR = 0.001
FINETUNE_LR = 1e-4
EPOCHS = 25
WARMUP_EPOCHS = 5
PATIENCE = 5
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Fashion")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
NUM_WORKERS = 8
SAMPLES_PER_CLASS = 1800  # set to None to use full dataset
# Weight loss by inverse frequency using full train_data.json (helps when subset is balanced).
USE_CLASS_WEIGHTS = True


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    assert torch.cuda.is_available(), "CUDA GPU not found. Install the CUDA build of PyTorch."
    device = torch.device("cuda")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    torch.backends.cudnn.benchmark = True

    # ── Data ─────────────────────────────────────────────────────────────────
    print("Loading data...")
    train_loader, val_loader, _, label_map = get_dataloaders(
        DATA_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        samples_per_class=SAMPLES_PER_CLASS,
        persistent_workers=True,
        prefetch_factor=2,
    )

    label_map_path = os.path.join(RESULTS_DIR, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to {label_map_path}")

    num_classes = len(label_map)

    if USE_CLASS_WEIGHTS:
        cw = class_weights_from_full_train(DATA_DIR, label_map).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
        print("Using class-weighted CrossEntropyLoss (weights from full train counts).")
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR,
    )
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2,
    )

    # ── Training loop ────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_val_loss = float("inf")
    unfrozen = False

    print(f"\nStarting training for {EPOCHS} epochs (warmup: {WARMUP_EPOCHS})...\n")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Unfreeze after warmup
        if epoch == WARMUP_EPOCHS + 1 and not unfrozen:
            print(">> Unfreezing all layers for fine-tuning")
            unfreeze_model(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
            scaler = torch.cuda.amp.GradScaler()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2,
            )
            patience_counter = 0
            best_val_loss = float("inf")
            unfrozen = True

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"  -> LR reduced: {old_lr:.6f} -> {new_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            ckpt_path = os.path.join(RESULTS_DIR, "best_model.pth")
            torch.save(best_model_wts, ckpt_path)
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed / 60:.1f} min. Best val acc: {best_val_acc:.4f}")

    # ── Save history ─────────────────────────────────────────────────────────
    history_path = os.path.join(RESULTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
