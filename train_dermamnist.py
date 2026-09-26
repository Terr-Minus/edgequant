#!/usr/bin/env python
"""Fine-tune ResNet-18 on DermaMNIST (MedMNIST) and report test accuracy.

Day-2 deliverable (part 1 of 2): produces the FP32 reference checkpoint whose
test accuracy becomes the baseline number every later quantised run is compared
against.

Pipeline position
-----------------
    this script  ->  fp32 checkpoint + test accuracy   (baseline)
    benchmark.py ->  latency / p50 / p95 / peak memory (baseline)
    (week 1-2)   ->  ONNX export -> FP16/INT8 quantisation -> re-measure both

Data-split discipline (the rule this script exists to keep)
-----------------------------------------------------------
    train split  -> fine-tuning
    train split  -> quantisation calibration (week 1-2)
    val split    -> optional early stopping
    TEST split   -> accuracy reporting ONLY, never calibration

Using the test split for calibration would leak test data into the quantisation
process and inflate the reported accuracy. See the data-split discipline section
of README.md.

Usage
-----
    python train_dermamnist.py --epochs 10
    python train_dermamnist.py --epochs 20 --batch-size 128 --out checkpoints/resnet18-fp32.pt
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# MedMNIST v3 renamed the per-dataset classes; v2 exposes only the generic
# DataClass. Support both rather than pinning a single version.
try:
    from medmnist import DermaMNIST as _DermaMNIST
except ImportError:  # pragma: no cover - depends on installed medmnist version
    from medmnist import DataClass as _DermaMNIST


def build_model(num_classes: int) -> nn.Module:
    """The architecture now lives in models.py; this is a thin re-export.

    Rationale: benchmark.py has to rebuild the *identical* architecture to load a
    checkpoint's state_dict, and two copy-pasted definitions drift. See models.py.
    """
    from models import build_model as _build

    return _build(num_classes)


def loaders(batch_size: int, num_workers: int, root: str = "data"):
    """Build the three loaders.

    `root` must be created before medmnist sees it. From medmnist/dataset.py:

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory.")

    medmnist never creates the directory, so a fresh clone fails with that
    RuntimeError *before* any download starts. We create it here, which also
    keeps the dataset inside the repository so a run is reproducible from a
    clean checkout.
    """
    import torchvision.transforms as T
    from pathlib import Path

    Path(root).mkdir(parents=True, exist_ok=True)

    # Normalisation constants for DermaMNIST as published by MedMNIST.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    eval_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_ds = _DermaMNIST(split="train", transform=train_tf, download=True, size=28, root=root)
    val_ds = _DermaMNIST(split="val", transform=eval_tf, download=True, size=28, root=root)
    test_ds = _DermaMNIST(split="test", transform=eval_tf, download=True, size=28, root=root)

    pin = torch.cuda.is_available()
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin),
        train_ds,
    )


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        targets = targets.squeeze().long()
        correct += int((preds == targets).sum())
        total += targets.numel()
    return correct / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows")
    parser.add_argument(
        "--data-root", default="data",
        help="where medmnist stores the .npz files (default: ./data, gitignored)",
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU (slow; for smoke-testing only)")
    parser.add_argument("--out", default="checkpoints/resnet18-dermamnist-fp32.pt")
    parser.add_argument("--report", default="results/fp32-accuracy.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.cpu and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available to torch. You are probably in an environment "
            "with the CPU-only build. Run `python -c \"import torch; print(torch.__version__)\"` "
            "and check for a +cpu suffix. See ENVIRONMENT.md."
        )

    torch.manual_seed(args.seed)
    device = torch.device("cpu" if args.cpu else "cuda")

    train_loader, val_loader, test_loader, train_ds = loaders(
        args.batch_size, args.num_workers, root=args.data_root
    )
    num_classes = len(train_ds.info["label"])
    model = build_model(num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    print(f"device={device}  classes={num_classes}  epochs={args.epochs}  train={len(train_ds)}")
    started = time.perf_counter()
    best_val = 0.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.squeeze().long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * images.size(0)

        scheduler.step()
        val_acc = evaluate(model, val_loader, device)
        flag = ""
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best so far"
        print(f"  epoch {epoch:>2}/{args.epochs}  loss={running / len(train_ds):.4f}  val_acc={val_acc:.4f}{flag}")

    # Restore the best-validation weights before reporting test accuracy. This is
    # the whole point of keeping a val split: test must be touched exactly once,
    # by the final model, not used to pick the epoch.
    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    train_acc = evaluate(model, train_loader, device)
    elapsed = time.perf_counter() - started

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "arch": "resnet18-28x28",
            "dataset": "DermaMNIST",
            "split_protocol": "trained on train, selected on val, reported on test",
            "test_accuracy": test_acc,
        },
        out_path,
    )

    report = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "model": "resnet18-28x28",
        "dataset": "DermaMNIST",
        "precision": "fp32",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "best_val_accuracy": round(best_val, 4),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "train_seconds": round(elapsed, 1),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": str(out_path).replace("\\", "/"),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nbest val_acc = {best_val:.4f}")
    print(f"TEST  acc   = {test_acc:.4f}   <- this is the FP32 baseline accuracy")
    print(f"train acc   = {train_acc:.4f}")
    if train_acc - test_acc > 0.15:
        print(
            "\nNOTE: train accuracy exceeds test accuracy by more than 15 points, "
            "which suggests overfitting. That is acceptable for this project "
            "(the goal is a stable FP32 reference for quantisation comparison, "
            "not a leaderboard score), but state it honestly in the report."
        )
    print(f"\nsaved checkpoint: {out_path}")
    print(f"wrote report:     {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
