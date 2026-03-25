import os
import json
import random
import logging
from collections import defaultdict, Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_data(json_path):
    """Read a JSONL file and return list of (image_path, class_label) tuples."""
    data = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            data.append((record["image_path"], record["class_label"]))
    return data


def stratified_subset(data, samples_per_class, seed=42):
    """Randomly sample up to `samples_per_class` items from each category."""
    by_class = defaultdict(list)
    for item in data:
        by_class[item[1]].append(item)

    rng = random.Random(seed)
    subset = []
    for label in sorted(by_class):
        items = by_class[label]
        n = min(samples_per_class, len(items))
        subset.extend(rng.sample(items, n))

    rng.shuffle(subset)
    return subset


def build_label_map(data):
    """Build a sorted label-to-index mapping from a list of (image_path, label) tuples."""
    labels = sorted(set(label for _, label in data))
    return {label: idx for idx, label in enumerate(labels)}


class IndoFashionDataset(Dataset):
    def __init__(self, data, label_map, base_dir, transform=None):
        self.data = data
        self.label_map = label_map
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, class_label = self.data[idx]
        full_path = os.path.join(self.base_dir, image_path)
        label_idx = self.label_map[class_label]

        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {full_path}: {e}. Returning black tensor.")
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label_idx


def class_weights_from_full_train(base_dir, label_map):
    """Inverse-frequency weights (mean 1.0) from full train JSONL counts.

    Use this when training on a balanced subset so rare classes in the full
    dataset still get higher loss weight.
    """
    path = os.path.join(base_dir, "train_data.json")
    data = load_data(path)
    cnt = Counter(lbl for _, lbl in data)
    num_classes = len(label_map)
    counts = torch.ones(num_classes, dtype=torch.float32)
    for name, idx in label_map.items():
        counts[idx] = float(max(cnt.get(name, 0), 1))
    w = counts.sum() / (num_classes * counts)
    w = w / w.mean()
    return w


def get_transforms():
    """Return train and val/test transform pipelines.

    Train uses RandomResizedCrop (not a fixed squash to 224×224) so geometry
    matches val/test and API inference: resize shorter side, then square crop.
    Squash-train + crop-infer shifts the distribution and hurts classes like
    women_kurta vs gowns/petticoats on full-body catalog photos.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.7, 1.0), ratio=(0.75, 1.333333),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_test_transform


def get_dataloaders(
    base_dir,
    batch_size=32,
    num_workers=4,
    pin_memory=None,
    samples_per_class=None,
    persistent_workers=True,
    prefetch_factor=2,
):
    """Build datasets and dataloaders for train, val, and test splits.

    Args:
        base_dir: Path to the Fashion/ directory containing the JSON files and images/.
        batch_size: Batch size for all loaders.
        num_workers: Number of dataloader worker processes.
        samples_per_class: If set, randomly sample this many images per category
                           for training (val/test are kept intact).

    Returns:
        (train_loader, val_loader, test_loader, label_map)
    """
    train_data = load_data(os.path.join(base_dir, "train_data.json"))
    val_data = load_data(os.path.join(base_dir, "val_data.json"))
    test_data = load_data(os.path.join(base_dir, "test_data.json"))

    if samples_per_class is not None:
        full = len(train_data)
        train_data = stratified_subset(train_data, samples_per_class)
        print(f"Subset: {full} -> {len(train_data)} training samples "
              f"({samples_per_class}/class)")

    label_map = build_label_map(train_data)

    train_transform, val_test_transform = get_transforms()

    train_dataset = IndoFashionDataset(train_data, label_map, base_dir, train_transform)
    val_dataset = IndoFashionDataset(val_data, label_map, base_dir, val_test_transform)
    test_dataset = IndoFashionDataset(test_data, label_map, base_dir, val_test_transform)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers) and num_workers > 0,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, label_map
