"""Model definitions, shared by training and benchmarking.

Why this module exists
---------------------
`train_dermamnist.py` builds the model and `benchmark.py` has to rebuild the
*identical* architecture in order to load a `state_dict` out of a checkpoint.
`load_state_dict` matches on parameter names and shapes, so the two definitions
must agree exactly. They used to be copy-pasted, which meant that editing one and
forgetting the other produced a confusing `RuntimeError` about missing/unexpected
keys. That duplication is now removed: both scripts import from here.

If you change the architecture, you change it in exactly one place.
"""

from __future__ import annotations

import torch.nn as nn

ARCH_NAME = "resnet18-28x28"


def build_model(num_classes: int = 7) -> nn.Module:
    """ResNet-18 adapted for 28x28 input.

    Stock ResNet-18 is designed for 224x224: a 7x7 stride-2 convolution followed
    by a 3x3 stride-2 max-pool downsamples by 4x *before the first residual
    stage*. On a 28x28 image that leaves a 7x7 feature map and discards most of
    the spatial detail.

    The standard CIFAR-style adaptation replaces the stem with a 3x3 stride-1
    convolution and drops the max-pool, so 28x28 reaches the first stage intact.

    Two knobs change; the residual body is untouched:

      conv1    7x7 stride 2  ->  3x3 stride 1
      maxpool  3x3 stride 2  ->  Identity
      fc       1000 classes  ->  num_classes
    """
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model_from_checkpoint(checkpoint: dict) -> nn.Module:
    """Rebuild the architecture recorded in a checkpoint and load its weights.

    Reads `num_classes` from the checkpoint rather than assuming 7, so a model
    trained on a different dataset subset still loads.
    """
    import torch

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("checkpoint must be a dict containing a 'state_dict' key")

    recorded_arch = checkpoint.get("arch")
    if recorded_arch is not None and recorded_arch != ARCH_NAME:
        # Not fatal, but worth surfacing: a mismatch here usually means the
        # architecture was changed after the checkpoint was trained, and
        # load_state_dict is about to fail with confusing key names.
        print(
            f"note: checkpoint was trained as '{recorded_arch}', "
            f"this build is '{ARCH_NAME}'"
        )

    model = build_model(int(checkpoint.get("num_classes", 7)))
    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """torch.load wrapper with the flags this project needs."""
    import torch

    return torch.load(path, map_location=map_location, weights_only=False)
