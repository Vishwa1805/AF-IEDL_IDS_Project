
"""Diagnose an existing IEDL-IDS baseline without modifying it."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from model import IEDLIDS

CLASSES = [
    "backdoor", "ddos", "dos", "injection", "normal",
    "password", "ransomware", "scanning", "xss"
]


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        x = z["X"]
        y = z["y"].astype(np.int64)

    if x.ndim != 4 or x.shape[1:] != (21, 21, 3):
        raise ValueError(f"Unexpected image shape: {x.shape}")

    if len(x) != len(y):
        raise ValueError("Image and label counts do not match.")

    return x, y


def evaluate(model, x, y, device, batch_size):
    all_probs, all_preds, all_losses, all_latents = [], [], [], []
    criterion = nn.CrossEntropyLoss(reduction="none")

    model.eval()

    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]

            xb = np.ascontiguousarray(xb.transpose(0, 3, 1, 2))
            xb = torch.from_numpy(xb).float().to(device) / 255.0
            yb = torch.from_numpy(yb).long().to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            losses = criterion(logits, yb)
            latents = model.encode(xb)

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_losses.append(losses.cpu().numpy())
            all_latents.append(latents.cpu().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_losses),
        np.concatenate(all_latents)
    )


def get_class_metrics(y, preds, probs, losses, split):
    rows = []

    for c, name in enumerate(CLASSES):
        mask = y == c

        if not mask.any():
            continue

        tp = int(np.sum(mask & (preds == c)))
        fp = int(np.sum((y != c) & (preds == c)))
        fn = int(np.sum(mask & (preds != c)))

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        correct = mask & (preds == y)
        wrong = mask & (preds != y)

        rows.append({
            "split": split,
            "class_id": c,
            "class": name,
            "support": int(mask.sum()),
            "correct": tp,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_cross_entropy": float(losses[mask].mean()),
            "mean_confidence_correct": (
                float(probs[correct].max(axis=1).mean())
                if correct.any() else None
            ),
            "mean_confidence_wrong": (
                float(probs[wrong].max(axis=1).mean())
                if wrong.any() else None
            )
        })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model_args = checkpoint.get("args", {})

    model = IEDLIDS(
        num_classes=9,
        latent_dim=int(model_args.get("latent_dim", 20)),
        hidden_dim=int(model_args.get("hidden_dim", 256)),
        dropout=float(model_args.get("dropout", 0.5))
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    summary = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "splits": {}
    }

    split_outputs = {}
    all_class_rows = []

    for split in ["train", "validation", "test"]:
        print(f"Diagnosing {split} split...")

        x, y = load_npz(data_dir / f"images_{split}.npz")

        probs, preds, losses, latents = evaluate(
            model, x, y, device, args.batch_size
        )

        correct = preds == y
        wrong = ~correct

        split_summary = {
            "images": int(len(y)),
            "accuracy": float(correct.mean()),
            "mean_cross_entropy": float(losses.mean()),
            "mean_max_probability": float(probs.max(axis=1).mean()),
            "mean_confidence_correct": (
                float(probs[correct].max(axis=1).mean())
                if correct.any() else None
            ),
            "mean_confidence_wrong": (
                float(probs[wrong].max(axis=1).mean())
                if wrong.any() else None
            ),
            "mean_pixel": float((x.astype(np.float32) / 255).mean()),
            "std_pixel": float((x.astype(np.float32) / 255).std()),
            "mean_latent_norm": float(
                np.linalg.norm(latents, axis=1).mean()
            )
        }

        summary["splits"][split] = split_summary

        all_class_rows.extend(
            get_class_metrics(y, preds, probs, losses, split)
        )

        split_outputs[split] = {
            "y": y,
            "latents": latents
        }

    shift_rows = []

    train_y = split_outputs["train"]["y"]
    train_z = split_outputs["train"]["latents"]
    val_y = split_outputs["validation"]["y"]
    val_z = split_outputs["validation"]["latents"]
    test_y = split_outputs["test"]["y"]
    test_z = split_outputs["test"]["latents"]

    for c, name in enumerate(CLASSES):
        tr = train_z[train_y == c]
        va = val_z[val_y == c]
        te = test_z[test_y == c]

        if len(tr) and len(va) and len(te):
            scale = np.std(tr, axis=0) + 1e-6

            shift_rows.append({
                "class": name,
                "train_vs_validation_shift": float(
                    np.mean(np.abs(tr.mean(0) - va.mean(0)) / scale)
                ),
                "train_vs_test_shift": float(
                    np.mean(np.abs(tr.mean(0) - te.mean(0)) / scale)
                ),
                "validation_vs_test_shift": float(
                    np.mean(np.abs(va.mean(0) - te.mean(0)) / scale)
                )
            })

    summary["interpretation_note"] = (
        "Latent shifts are descriptive measurements, not formal "
        "statistical tests. A large shift does not prove its cause."
    )

    with open(
        out_dir / "diagnostic_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    with open(
        out_dir / "class_metrics_by_split.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(all_class_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_class_rows)

    with open(
        out_dir / "latent_shift_by_class.csv",
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(shift_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(shift_rows)

    print("\nDIAGNOSTICS COMPLETED")
    print(json.dumps(summary, indent=2))
    print(f"\nOutput directory: {out_dir}")


if __name__ == "__main__":
    main()
