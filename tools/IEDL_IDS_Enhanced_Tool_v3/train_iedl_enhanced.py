"""Train an attention/focal-loss IEDL-IDS extension with controlled ablations.

Variants:
  cbam       : CBAM encoder + ordinary cross-entropy
  focal      : original CNN encoder + class-balanced focal loss
  cbam_focal : CBAM encoder + class-balanced focal loss (proposed hybrid)

The data, preprocessing, split files, class IDs, and test set are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from model_enhanced import AttentionIEDLIDS, count_parameters

LABEL_NAMES = [
    "backdoor", "ddos", "dos", "injection", "normal",
    "password", "ransomware", "scanning", "xss",
]
NUM_CLASSES = len(LABEL_NAMES)


class NPZImageDataset(Dataset):
    def __init__(self, npz_path: Path) -> None:
        with np.load(npz_path, allow_pickle=False) as data:
            if not {"X", "y"}.issubset(data.files):
                raise ValueError(f"{npz_path} must contain X and y; found {data.files}")
            x, y = data["X"], data["y"]
        if x.ndim != 4 or x.shape[1:] != (21, 21, 3):
            raise ValueError(f"Bad image shape in {npz_path}: {x.shape}")
        if len(x) != len(y):
            raise ValueError(f"Image/label length mismatch in {npz_path}")
        if not np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64)
        if np.any(y < 0) or np.any(y >= NUM_CLASSES):
            raise ValueError(f"Labels in {npz_path} must be IDs 0..{NUM_CLASSES-1}")
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weights."""
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            if alpha.ndim != 1 or len(alpha) != NUM_CLASSES:
                raise ValueError("alpha must have one weight per class")
            self.register_buffer("alpha", alpha.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1.0 - pt).clamp(min=0.0) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_alpha(train_labels: torch.Tensor) -> torch.Tensor:
    """Normalized inverse-square-root class weights, derived from train only."""
    counts = torch.bincount(train_labels, minlength=NUM_CLASSES).float()
    if torch.any(counts == 0):
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"Training split has no samples for classes: {missing}")
    alpha = counts.rsqrt()
    alpha = alpha / alpha.mean()
    return alpha


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_obj, total_ce, n_seen = 0.0, 0.0, 0
    all_y, all_pred = [], []
    ce_criterion = nn.CrossEntropyLoss(reduction="sum")

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                loss.backward()
                optimizer.step()
        bs = yb.size(0)
        total_obj += float(loss.item()) * bs
        total_ce += float(ce_criterion(logits.detach(), yb).item())
        n_seen += bs
        all_y.append(yb.detach().cpu().numpy())
        all_pred.append(logits.detach().argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    return {
        "loss": total_obj / max(n_seen, 1),
        "cross_entropy": total_ce / max(n_seen, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", required=True, choices=["cbam", "focal", "cbam_focal"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--latent-dim", type=int, default=20)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        ap.error("--epochs and --batch-size must be positive")

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {
        split: data_dir / f"images_{split}.npz"
        for split in ["train", "validation", "test"]
    }
    for split, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {split} file: {path}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = {k: NPZImageDataset(v) for k, v in files.items()}
    for name, ds in datasets.items():
        labels, counts = np.unique(ds.y.numpy(), return_counts=True)
        print(f"{name}: X={tuple(ds.x.shape)}, y={tuple(ds.y.shape)}, "
              f"labels={dict(zip(labels.tolist(), counts.tolist()))}")

    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"), generator=generator),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers,
                                 pin_memory=(device.type == "cuda")),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=(device.type == "cuda")),
    }

    use_cbam = args.variant in ("cbam", "cbam_focal")
    use_focal = args.variant in ("focal", "cbam_focal")
    model = AttentionIEDLIDS(
        num_classes=NUM_CLASSES, latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim, dropout=args.dropout,
        use_cbam=use_cbam,
    ).to(device)
    alpha = make_alpha(datasets["train"].y).to(device) if use_focal else None
    criterion = FocalLoss(args.focal_gamma, alpha) if use_focal else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Variant: {args.variant}; CBAM={use_cbam}; focal_loss={use_focal}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    if alpha is not None:
        print("Focal alpha (class order 0..8):", [round(float(v), 4) for v in alpha.cpu()])

    best_val_f1 = -1.0
    history_path = out_dir / "training_history.csv"
    history_fields = ["epoch", "train_loss", "train_cross_entropy", "train_accuracy",
                      "train_macro_f1", "train_weighted_f1", "val_loss",
                      "val_cross_entropy", "val_accuracy", "val_macro_f1",
                      "val_weighted_f1", "epoch_seconds"]
    start_all = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, loaders["train"], criterion, device, optimizer)
            va = run_epoch(model, loaders["validation"], criterion, device)
            elapsed = time.time() - t0
            writer.writerow({
                "epoch": epoch, "train_loss": tr["loss"],
                "train_cross_entropy": tr["cross_entropy"],
                "train_accuracy": tr["accuracy"], "train_macro_f1": tr["macro_f1"],
                "train_weighted_f1": tr["weighted_f1"], "val_loss": va["loss"],
                "val_cross_entropy": va["cross_entropy"],
                "val_accuracy": va["accuracy"], "val_macro_f1": va["macro_f1"],
                "val_weighted_f1": va["weighted_f1"], "epoch_seconds": elapsed,
            })
            f.flush()
            print(f"Epoch {epoch:02d}/{args.epochs} | "
                  f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} macroF1 {tr['macro_f1']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} macroF1 {va['macro_f1']:.4f} | "
                  f"{elapsed:.1f}s")
            if va["macro_f1"] > best_val_f1:
                best_val_f1 = va["macro_f1"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "args": vars(args), "best_val_macro_f1": best_val_f1,
                    "epoch": epoch, "alpha": alpha.detach().cpu() if alpha is not None else None,
                }, out_dir / "best_model.pt")

    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = run_epoch(model, loaders["test"], criterion, device)
    labels = list(range(NUM_CLASSES))
    report = classification_report(
        test["y_true"], test["y_pred"], labels=labels,
        target_names=LABEL_NAMES, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(test["y_true"], test["y_pred"], labels=labels)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    with (out_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    metrics = {
        "variant": args.variant,
        "device": str(device),
        "epochs_requested": args.epochs,
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(checkpoint["best_val_macro_f1"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_weighted_f1": float(test["weighted_f1"]),
        "test_macro_precision": float(precision_score(
            test["y_true"], test["y_pred"], average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(
            test["y_true"], test["y_pred"], average="macro", zero_division=0)),
        "test_objective_loss": float(test["loss"]),
        "test_cross_entropy": float(test["cross_entropy"]),
        "trainable_parameters": count_parameters(model),
        "total_training_seconds": time.time() - start_all,
        "class_names": LABEL_NAMES,
        "cbam_enabled": use_cbam,
        "focal_enabled": use_focal,
        "focal_gamma": args.focal_gamma if use_focal else None,
        "focal_alpha": alpha.detach().cpu().tolist() if alpha is not None else None,
        "selection_metric": "validation macro-F1",
        "note": "Same fixed image NPZ splits as baseline; do not tune using test results.",
    }
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nTEST RESULTS")
    for key in ["test_accuracy", "test_macro_f1", "test_weighted_f1",
                "test_macro_precision", "test_macro_recall", "test_cross_entropy"]:
        print(f"{key}: {metrics[key]:.4f}")
    print(f"Best checkpoint epoch: {checkpoint['epoch']}")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
