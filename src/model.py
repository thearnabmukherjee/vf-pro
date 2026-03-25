import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def build_model(num_classes=15, dropout=0.3):
    """Load a pretrained EfficientNet-B0 and replace the classifier head.

    The backbone is frozen initially so only the new head trains during warmup.
    """
    model = EfficientNet.from_pretrained("efficientnet-b0")

    for param in model.parameters():
        param.requires_grad = False

    in_features = model._fc.in_features
    model._fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )

    return model


def unfreeze_model(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
