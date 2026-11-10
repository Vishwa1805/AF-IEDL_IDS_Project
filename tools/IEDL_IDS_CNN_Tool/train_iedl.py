"""Train and evaluate the IEDL-IDS CNN-encoder + MLP baseline.

Input NPZ files are expected to contain arrays named X and y, with X stored NHWC
(N,21,21,3), as produced by prepare_iedl_images.py.
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
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)

from model import IEDLIDS, count_parameters


class NPZImageDataset(Dataset):
    def __init__(self, npz_path: Path) -> None:
        with np.load(npz_path, allow_pickle=False) as data:
            if not {"X", "y"}.issubset(data.files):
                raise ValueError(f"{npz_path} must contain X and y; found {data.files}")
            x = data["X"]
            y = data["y"]
        if x.ndim != 4 or x.shape[1:] != (21, 21, 3):
            raise ValueError(f"Bad image shape in {npz_path}: {x.shape}")
        if len(x) != len(y):
            raise ValueError(f"Image/label length mismatch in {npz_path}")
        if not np.issubdtype(y.dtype, np.integer):
            y = y.astype(np.int64)
        # Convert NHWC uint8 RGB [0,255] to NCHW float32 [0,1].
        x = np.ascontiguousarray(x.transpose(0, 3, 1, 2), dtype=np.float32) / 255.0
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(np.asarray(y, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, all_y, all_pred = 0.0, [], []
    n_seen = 0
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
        total_loss += float(loss.item()) * bs
        n_seen += bs
        all_y.extend(yb.detach().cpu().numpy().tolist())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
    avg_loss = total_loss / max(n_seen, 1)
    return {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_y, all_pred),
        "macro_f1": f1_score(all_y, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_y, all_pred, average="weighted", zero_division=0),
        "y_true": np.asarray(all_y),
        "y_pred": np.asarray(all_pred),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with images_train/validation/test.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "train": data_dir / "images_train.npz",
        "validation": data_dir / "images_validation.npz",
        "test": data_dir / "images_test.npz",
    }
    for split, path in expected.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} file: {path}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    datasets = {k: NPZImageDataset(v) for k, v in expected.items()}
    for k, ds in datasets.items():
        labels, counts = np.unique(ds.y.numpy(), return_counts=True)
        print(f"{k}: X={tuple(ds.x.shape)}, y={tuple(ds.y.shape)}, labels={dict(zip(labels.tolist(), counts.tolist()))}")

    generators = torch.Generator().manual_seed(args.seed)
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
                            generator=generators),
        "validation": DataLoader(datasets["validation"], batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=(device.type == "cuda")),
    }

    model = IEDLIDS(num_classes=9, latent_dim=args.latent_dim,
                    hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    history_path = out_dir / "training_history.csv"
    fields = ["epoch", "train_loss", "train_accuracy", "train_macro_f1", "train_weighted_f1",
              "val_loss", "val_accuracy", "val_macro_f1", "val_weighted_f1", "epoch_seconds"]
    start_all = time.time()
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, loaders["train"], criterion, device, optimizer)
            va = run_epoch(model, loaders["validation"], criterion, device)
            elapsed = time.time() - t0
            row = {"epoch": epoch, "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
                   "train_macro_f1": tr["macro_f1"], "train_weighted_f1": tr["weighted_f1"],
                   "val_loss": va["loss"], "val_accuracy": va["accuracy"],
                   "val_macro_f1": va["macro_f1"], "val_weighted_f1": va["weighted_f1"],
                   "epoch_seconds": elapsed}
            writer.writerow(row)
            f.flush()
            print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} macroF1 {tr['macro_f1']:.4f} | val loss {va['loss']:.4f} acc {va['accuracy']:.4f} macroF1 {va['macro_f1']:.4f} | {elapsed:.1f}s")
            # Select checkpoint by validation macro-F1 because class imbalance is substantial.
            if va["macro_f1"] > best_val_f1:
                best_val_f1 = va["macro_f1"]
                torch.save({"model_state_dict": model.state_dict(),
                            "args": vars(args), "best_val_macro_f1": best_val_f1,
                            "epoch": epoch}, out_dir / "best_model.pt")

    # Restore the best validation checkpoint before one final test evaluation.
    checkpoint = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = run_epoch(model, loaders["test"], criterion, device)
    label_names = ["backdoor", "ddos", "dos", "injection", "normal", "password", "ransomware", "scanning", "xss"]
    labels = list(range(9))
    report = classification_report(test["y_true"], test["y_pred"], labels=labels,
                                   target_names=label_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(test["y_true"], test["y_pred"], labels=labels)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    with (out_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    metrics = {
        "device": str(device), "epochs_requested": args.epochs,
        "best_checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(checkpoint["best_val_macro_f1"]),
        "test_loss": float(test["loss"]), "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["macro_f1"]), "test_weighted_f1": float(test["weighted_f1"]),
        "test_macro_precision": float(precision_score(test["y_true"], test["y_pred"], average="macro", zero_division=0)),
        "test_macro_recall": float(recall_score(test["y_true"], test["y_pred"], average="macro", zero_division=0)),
        "trainable_parameters": count_parameters(model), "total_training_seconds": time.time()-start_all,
        "class_names": label_names, "note": "Baseline implementation; compare against this checkpoint using the same splits and preprocessing."
    }
    with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("\nTEST RESULTS")
    for k in ["test_accuracy", "test_macro_f1", "test_weighted_f1", "test_macro_precision", "test_macro_recall"]:
        print(f"{k}: {metrics[k]:.4f}")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
