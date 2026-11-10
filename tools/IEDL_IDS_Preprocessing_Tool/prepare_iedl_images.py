#!/usr/bin/env python3
"""Prepare class-homogeneous IEDL-IDS RGB images from the sampled ToN_IoT CSV.

This script intentionally operates on the 10% sampled CSV, not the 23 original
full-size shards. It makes the following explicit, reproducible choices where
IEDL-IDS is underspecified:
  * 21-feature set reconstructed from the UNSW feature dictionary + audit.
  * group rows within each class, ordered by timestamp, into windows of 63.
  * assign each image its homogeneous class label.
  * chronological per-class image split: 80% train, 10% validation, remainder test.
  * fit categorical vocabularies and min-max ranges using training flows only.

No source CSV is modified. Outputs are compressed NPZ files and JSON/CSV reports.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

FEATURES = [
    "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service",
    "duration", "src_bytes", "dst_bytes", "conn_state", "src_pkts",
    "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_AA", "dns_RD",
    "dns_RA", "dns_rejected", "ssl_resumed", "ssl_established", "weird_notice",
]
TIMESTAMP = "ts"
TARGET = "type"
BINARY_TARGET = "label"
CLASSES = ["backdoor", "ddos", "dos", "injection", "normal", "password", "ransomware", "scanning", "xss"]
CATEGORICAL = ["src_ip", "dst_ip", "proto", "service", "conn_state"]
NUMERIC = ["src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
BOOLEAN = ["dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_resumed", "ssl_established", "weird_notice"]
GROUP_FLOWS = 63
IMAGE_SIZE = 21


def norm_text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("<MISSING>").str.strip().replace({"": "<MISSING>"})


def load_class(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df[TIMESTAMP] = pd.to_numeric(df[TIMESTAMP], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP])
    # Stable ordering preserves input order for timestamp ties.
    return df.sort_values(TIMESTAMP, kind="mergesort").reset_index(drop=True)


def numeric_frame(df: pd.DataFrame, cat_maps: Dict[str, Dict[str, int]], medians: Dict[str, float]) -> np.ndarray:
    n = len(df)
    out = np.empty((n, len(FEATURES)), dtype=np.float32)
    pos = {f: i for i, f in enumerate(FEATURES)}

    for col in CATEGORICAL:
        vals = norm_text(df[col])
        mapping = cat_maps[col]
        out[:, pos[col]] = vals.map(mapping).fillna(0).to_numpy(dtype=np.float32)

    for col in NUMERIC:
        vals = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = vals.fillna(medians[col])
        out[:, pos[col]] = vals.to_numpy(dtype=np.float32)

    for col in BOOLEAN:
        vals = norm_text(df[col]).str.upper()
        # Three states: absent/not-applicable=-1, explicitly false=0, true=1.
        mapped = vals.map({"-": -1.0, "<MISSING>": -1.0, "F": 0.0, "FALSE": 0.0, "0": 0.0,
                           "T": 1.0, "TRUE": 1.0, "1": 1.0})
        out[:, pos[col]] = mapped.fillna(-1.0).to_numpy(dtype=np.float32)
    return out


def build(args: argparse.Namespace) -> None:
    src = Path(args.input_csv).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Sampled CSV not found: {src}")
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_class_partitions"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    required = [TIMESTAMP, *FEATURES, BINARY_TARGET, TARGET]
    # Header check before the one pass over the sampled data.
    header = pd.read_csv(src, nrows=0).columns.tolist()
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    print(f"Partitioning sampled CSV: {src}")
    print("Only 9 multiclass categories are used; MITM is written to an exclusion report.")
    wrote = set()
    rows_by_class = {c: 0 for c in CLASSES}
    mitm_rows = 0
    total_input = 0
    usecols = required

    for ci, chunk in enumerate(pd.read_csv(src, usecols=usecols, chunksize=args.chunksize, low_memory=False), 1):
        total_input += len(chunk)
        chunk[TARGET] = chunk[TARGET].astype("string").str.strip().str.lower()
        if BINARY_TARGET in chunk:
            # The paper's type field is the multiclass target; ensure binary label agrees.
            expected_binary = (chunk[TARGET] != "normal").astype("int8")
            actual_binary = pd.to_numeric(chunk[BINARY_TARGET], errors="coerce")
            mismatch = actual_binary.notna() & (actual_binary.astype("Int64") != expected_binary.astype("Int64"))
            if mismatch.any():
                raise ValueError(f"label/type inconsistency found in input chunk {ci}: {int(mismatch.sum())} rows")
        mitm_rows += int((chunk[TARGET] == "mitm").sum())
        for cls in CLASSES:
            part = chunk.loc[chunk[TARGET] == cls, required]
            if part.empty:
                continue
            p = tmp / f"{cls}.csv"
            part.to_csv(p, mode="a", index=False, header=(cls not in wrote))
            wrote.add(cls)
            rows_by_class[cls] += len(part)
        if ci % 5 == 0 or len(chunk) < args.chunksize:
            print(f"  chunks processed: {ci}; rows read: {total_input:,}")

    # Determine usable image groups and chronological split sizes per class.
    group_info = {}
    for cls in CLASSES:
        p = tmp / f"{cls}.csv"
        if not p.exists():
            raise ValueError(f"No records found for expected class: {cls}")
        count = rows_by_class[cls]
        groups = count // GROUP_FLOWS
        n_train = int(groups * 0.80)
        n_val = int(groups * 0.10)
        n_test = groups - n_train - n_val
        group_info[cls] = {
            "rows": count, "complete_images": groups, "discarded_remainder_flows": count % GROUP_FLOWS,
            "train_images": n_train, "validation_images": n_val, "test_images": n_test,
        }

    # Fit category vocabularies and numeric imputation/min-max parameters on training flows only.
    vocab_sets = {c: set() for c in CATEGORICAL}
    medians = {c: [] for c in NUMERIC}
    mins = np.full(len(FEATURES), np.inf, dtype=np.float64)
    maxs = np.full(len(FEATURES), -np.inf, dtype=np.float64)
    print("Fitting encoders and min-max ranges on training portions only...")
    for cls in CLASSES:
        df = load_class(tmp / f"{cls}.csv")
        info = group_info[cls]
        train_rows = info["train_images"] * GROUP_FLOWS
        tr = df.iloc[:train_rows]
        for col in CATEGORICAL:
            vocab_sets[col].update(norm_text(tr[col]).unique().tolist())
        for col in NUMERIC:
            vals = pd.to_numeric(tr[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            med = float(vals.median()) if vals.notna().any() else 0.0
            medians[col].append(med)
        # Temporary maps for categorical values; IDs are 1..K, unknown=0.
        local_maps = {c: {v: i + 1 for i, v in enumerate(sorted(vocab_sets[c]))} for c in CATEGORICAL}
        # We cannot use partially accumulated vocab maps for global numeric ranges;
        # category dimensions are later normalized by their global vocab bounds.
        arr = numeric_frame(tr, local_maps, {c: (medians[c][-1] if medians[c] else 0.0) for c in NUMERIC})
        if len(arr):
            mins = np.minimum(mins, np.nanmin(arr, axis=0))
            maxs = np.maximum(maxs, np.nanmax(arr, axis=0))

    # Recalculate categorical feature bounds from the complete training vocabularies.
    cat_maps = {c: {v: i + 1 for i, v in enumerate(sorted(vocab_sets[c]))} for c in CATEGORICAL}
    for c in NUMERIC:
        medians[c] = float(np.median(medians[c])) if medians[c] else 0.0
    for c in CATEGORICAL:
        idx = FEATURES.index(c)
        mins[idx] = 1.0 if cat_maps[c] else 0.0
        maxs[idx] = float(max(1, len(cat_maps[c])))
    # Recompute numeric minima/maxima robustly with the global maps and medians.
    mins = np.full(len(FEATURES), np.inf, dtype=np.float64)
    maxs = np.full(len(FEATURES), -np.inf, dtype=np.float64)
    for cls in CLASSES:
        df = load_class(tmp / f"{cls}.csv")
        tr = df.iloc[:group_info[cls]["train_images"] * GROUP_FLOWS]
        arr = numeric_frame(tr, cat_maps, medians)
        if len(arr):
            mins = np.minimum(mins, arr.min(axis=0))
            maxs = np.maximum(maxs, arr.max(axis=0))
    mins[~np.isfinite(mins)] = 0.0
    maxs[~np.isfinite(maxs)] = mins[~np.isfinite(maxs)]

    class_to_id = {c: i for i, c in enumerate(CLASSES)}
    split_images: Dict[str, List[np.ndarray]] = {s: [] for s in ["train", "validation", "test"]}
    split_labels: Dict[str, List[np.ndarray]] = {s: [] for s in ["train", "validation", "test"]}
    split_class_counts = {s: {c: 0 for c in CLASSES} for s in split_images}
    print("Generating RGB images (63 rows -> 21x21x3)...")
    for cls in CLASSES:
        df = load_class(tmp / f"{cls}.csv")
        info = group_info[cls]
        usable_rows = info["complete_images"] * GROUP_FLOWS
        df = df.iloc[:usable_rows]
        ntr = info["train_images"] * GROUP_FLOWS
        nval = info["validation_images"] * GROUP_FLOWS
        segments = {
            "train": df.iloc[:ntr],
            "validation": df.iloc[ntr:ntr+nval],
            "test": df.iloc[ntr+nval:usable_rows],
        }
        for split, part in segments.items():
            if part.empty:
                continue
            arr = numeric_frame(part, cat_maps, medians)
            scale = np.where((maxs - mins) > 0, (maxs - mins), 1.0)
            arr = np.clip((arr - mins) / scale, 0.0, 1.0)
            arr = np.rint(arr * 255.0).astype(np.uint8)
            # (N*63,21) -> (N,63,21) -> channels as first/middle/last 21 flows.
            blocks = arr.reshape(-1, GROUP_FLOWS, IMAGE_SIZE)
            rgb = np.stack((blocks[:, 0:21, :], blocks[:, 21:42, :], blocks[:, 42:63, :]), axis=-1)
            split_images[split].append(rgb)
            split_labels[split].append(np.full((len(rgb),), class_to_id[cls], dtype=np.int64))
            split_class_counts[split][cls] += len(rgb)
        print(f"  {cls}: {info['complete_images']:,} images; discarded {info['discarded_remainder_flows']} tail flows")

    for split in split_images:
        if not split_images[split]:
            raise RuntimeError(f"No images generated for split: {split}")
        X = np.concatenate(split_images[split], axis=0)
        y = np.concatenate(split_labels[split], axis=0)
        # Fixed seed shuffle within training only; validation/test ordering remains deterministic.
        if split == "train":
            rng = np.random.default_rng(args.seed)
            order = rng.permutation(len(y))
            X, y = X[order], y[order]
        np.savez_compressed(out / f"images_{split}.npz", X=X, y=y)
        print(f"Saved {split}: X={X.shape}, y={y.shape}")

    summary = {
        "input_csv": str(src),
        "input_rows_read": total_input,
        "multiclass_classes": CLASSES,
        "mitm_rows_excluded_from_multiclass": mitm_rows,
        "feature_selection": FEATURES,
        "feature_selection_note": (
            "Reconstructed selection: 14 populated connection/statistical features plus seven documented Boolean flags. "
            "The IEDL-IDS paper states 21 dimensions but does not enumerate feature names."
        ),
        "categorical_encoding": "Ordinal codes fitted on training flows; unknown validation/test categories map to 0.",
        "boolean_encoding": "T=1, F=0, '-' or blank=-1 before train-fitted min-max scaling.",
        "normalization": "Per-feature min-max fitted on training flows only; values quantized to uint8 [0,255].",
        "timestamp_ordering": "Stable ascending ts sort performed separately within each class.",
        "image_label_policy": "Class-homogeneous groups: each 63-flow image contains one type; this is an explicit implementation assumption because the paper does not specify mixed-label image assignment for multiclass.",
        "split_policy": "Chronological within-class split by complete image windows: 80% train, 10% validation, 10% test (rounded by class). This resolves the paper's inconsistent reported 80/20/20 percentages.",
        "flows_per_image": GROUP_FLOWS,
        "image_shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
        "class_to_id": class_to_id,
        "per_class": group_info,
        "split_image_counts": split_class_counts,
        "feature_min": {f: float(mins[i]) for i, f in enumerate(FEATURES)},
        "feature_max": {f: float(maxs[i]) for i, f in enumerate(FEATURES)},
        "categorical_vocab_sizes": {c: len(cat_maps[c]) for c in CATEGORICAL},
        "training_numeric_medians": medians,
        "seed": args.seed,
    }
    with open(out / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    rows = []
    for cls, info in group_info.items():
        rows.append({"class": cls, **info,
                     "train_images": split_class_counts["train"][cls],
                     "validation_images": split_class_counts["validation"][cls],
                     "test_images": split_class_counts["test"][cls]})
    pd.DataFrame(rows).to_csv(out / "image_generation_summary.csv", index=False)
    with open(out / "feature_selection.csv", "w", encoding="utf-8") as f:
        f.write("feature,selection_reason\n")
        for feat in FEATURES:
            reason = "Boolean indicator retained as semantic T/F flag" if feat in BOOLEAN else "Retained populated network attribute"
            f.write(f"{feat},{reason}\n")

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nPREPARATION COMPLETE")
    print(f"Output directory: {out}")
    print("Files: images_train.npz, images_validation.npz, images_test.npz, preprocessing_metadata.json, image_generation_summary.csv, feature_selection.csv")
    print("Important: review preprocessing_metadata.json before describing these assumptions as the exact authors' implementation.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", required=True, help="Path to sampled_raw_10pct.csv")
    ap.add_argument("--out-dir", required=True, help="Output directory for RGB image arrays and reports")
    ap.add_argument("--chunksize", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args)

if __name__ == "__main__":
    main()
