#!/usr/bin/env python3
"""Create a reproducible, chunk-wise 10% class-stratified sample of ToN_IoT shards.

The script reads the 23 Network_dataset_*.csv files once, keeps all source
features (including ts), drops uid when present because it is an identifier
and appears only in one shard, and writes one sampled CSV plus audit reports.
It does NOT perform feature selection, encoding, normalization, sorting, or
image generation. Those are intentionally postponed until the feature rule
is resolved against the paper and metadata.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SHARD_RE = re.compile(r"^Network_dataset_(\d+)\.csv$", re.IGNORECASE)
REQUIRED_COLUMNS = {"ts", "label", "type"}
IGNORED_IDENTIFIER_COLUMNS = {"uid"}


def ordered_shards(directory: Path) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = SHARD_RE.match(path.name)
        if path.is_file() and match:
            found.append((int(match.group(1)), path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise FileNotFoundError(f"No Network_dataset_*.csv files found in {directory}")
    indexes = [idx for idx, _ in found]
    if indexes != list(range(1, len(indexes) + 1)):
        raise ValueError(f"Shard numbering is not continuous: {indexes}")
    return [path for _, path in found]


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Folder containing Network_dataset_1.csv ... Network_dataset_23.csv")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output folder for sampled dataset and reports")
    parser.add_argument("--sample-frac", type=float, default=0.10, help="Sampling fraction within each class in each chunk (default: 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per read chunk (default: 100000)")
    parser.add_argument("--expected-total-rows", type=int, default=22_339_021, help="Expected total rows from the completed audit; set 0 to skip")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory contents")
    args = parser.parse_args()

    if not (0 < args.sample_frac < 1):
        parser.error("--sample-frac must be between 0 and 1")
    if args.chunksize < 1:
        parser.error("--chunksize must be positive")
    if not args.data_dir.is_dir():
        parser.error(f"Data directory does not exist: {args.data_dir}")

    shards = ordered_shards(args.data_dir)
    if len(shards) != 23:
        print(f"WARNING: found {len(shards)} shards, expected 23", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.out_dir / "sampled_raw_10pct.csv"
    report_paths = [
        output_csv,
        args.out_dir / "sampled_class_counts.csv",
        args.out_dir / "sampled_shard_counts.csv",
        args.out_dir / "sampling_manifest.json",
    ]
    existing = [p for p in report_paths if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists. Delete the previous output folder or rerun with --overwrite. "
            f"Existing: {', '.join(str(p) for p in existing)}"
        )
    if args.overwrite:
        for p in report_paths:
            if p.exists():
                p.unlink()

    canonical_header = read_header(shards[0])
    missing_required = REQUIRED_COLUMNS - set(canonical_header)
    if missing_required:
        raise ValueError(f"First shard missing required columns: {sorted(missing_required)}")
    canonical_header = [c for c in canonical_header if c not in IGNORED_IDENTIFIER_COLUMNS]

    rng = np.random.default_rng(args.seed)
    original_counts: Counter[str] = Counter()
    sampled_counts: Counter[str] = Counter()
    shard_rows: list[dict[str, int | str]] = []
    total_rows = 0
    total_sampled = 0
    wrote_header = False
    timestamp_min: int | None = None
    timestamp_max: int | None = None

    print(f"Found {len(shards)} shard(s) in: {args.data_dir}")
    print(f"Sampling {args.sample_frac:.1%} within each class per chunk; seed={args.seed}; chunksize={args.chunksize:,}")
    print("All feature columns will be retained for now; only uid is dropped as an identifier.")

    for shard_number, shard in enumerate(shards, start=1):
        header = read_header(shard)
        missing = set(canonical_header) - set(header)
        extra = set(header) - set(canonical_header) - IGNORED_IDENTIFIER_COLUMNS
        if missing or extra:
            raise ValueError(
                f"Schema mismatch in {shard.name}: missing={sorted(missing)}, extra={sorted(extra)}"
            )

        shard_original: Counter[str] = Counter()
        shard_sampled: Counter[str] = Counter()
        shard_row_count = 0

        reader = pd.read_csv(
            shard,
            chunksize=args.chunksize,
            dtype="string",
            keep_default_na=False,
            na_filter=False,
            low_memory=False,
        )
        for chunk in reader:
            if "uid" in chunk.columns:
                chunk = chunk.drop(columns=["uid"])
            # Keep a stable, identical column order across all shards.
            chunk = chunk.reindex(columns=canonical_header)
            chunk["type"] = chunk["type"].str.strip().str.lower()
            chunk["label"] = chunk["label"].str.strip()

            if chunk["type"].eq("").any():
                raise ValueError(f"Blank type label encountered in {shard.name}")
            if not set(chunk["type"].unique()).issubset({
                "normal", "backdoor", "ddos", "dos", "injection", "mitm",
                "password", "ransomware", "scanning", "xss"
            }):
                bad = sorted(set(chunk["type"].unique()) - {
                    "normal", "backdoor", "ddos", "dos", "injection", "mitm",
                    "password", "ransomware", "scanning", "xss"
                })
                raise ValueError(f"Unexpected type labels in {shard.name}: {bad}")

            counts = chunk["type"].value_counts(dropna=False)
            for label, count in counts.items():
                original_counts[str(label)] += int(count)
                shard_original[str(label)] += int(count)
            shard_row_count += len(chunk)
            total_rows += len(chunk)

            # Stratified sampling within each chunk: each class present in a chunk
            # is sampled independently at the requested fraction.
            sampled = chunk.groupby("type", sort=False, group_keys=False).sample(
                frac=args.sample_frac, random_state=rng
            )
            if not sampled.empty:
                # ts is deliberately preserved; later stages sort sampled rows
                # chronologically before constructing 63-flow image windows.
                ts_numeric = pd.to_numeric(sampled["ts"], errors="coerce")
                if ts_numeric.notna().any():
                    chunk_min = int(ts_numeric.min())
                    chunk_max = int(ts_numeric.max())
                    timestamp_min = chunk_min if timestamp_min is None else min(timestamp_min, chunk_min)
                    timestamp_max = chunk_max if timestamp_max is None else max(timestamp_max, chunk_max)

                sampled.to_csv(
                    output_csv,
                    mode="a",
                    header=not wrote_header,
                    index=False,
                    quoting=csv.QUOTE_MINIMAL,
                )
                wrote_header = True
                got = sampled["type"].value_counts()
                for label, count in got.items():
                    sampled_counts[str(label)] += int(count)
                    shard_sampled[str(label)] += int(count)
                total_sampled += len(sampled)

        shard_rows.append({
            "file": shard.name,
            "rows_read": shard_row_count,
            "rows_sampled": int(sum(shard_sampled.values())),
            "sampling_fraction_observed": (sum(shard_sampled.values()) / shard_row_count) if shard_row_count else 0.0,
        })
        print(
            f"[{shard_number:02d}/{len(shards):02d}] {shard.name}: "
            f"read {shard_row_count:,}; sampled {sum(shard_sampled.values()):,}"
        )

    if args.expected_total_rows and total_rows != args.expected_total_rows:
        raise ValueError(
            f"Total rows {total_rows:,} did not match audited total {args.expected_total_rows:,}. "
            "Output was produced but should not be used until reconciled."
        )
    if not wrote_header:
        raise RuntimeError("No rows were sampled; output CSV was not created")

    class_rows = []
    for label in sorted(original_counts):
        original = original_counts[label]
        selected = sampled_counts[label]
        class_rows.append({
            "type": label,
            "original_rows": original,
            "sampled_rows": selected,
            "sample_fraction_observed": selected / original if original else 0.0,
            "expected_rows_at_fraction": original * args.sample_frac,
        })
    pd.DataFrame(class_rows).to_csv(args.out_dir / "sampled_class_counts.csv", index=False)
    pd.DataFrame(shard_rows).to_csv(args.out_dir / "sampled_shard_counts.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_directory": str(args.data_dir.resolve()),
        "shards_found": len(shards),
        "total_rows_read": total_rows,
        "expected_total_rows": args.expected_total_rows or None,
        "total_rows_match_expected": (total_rows == args.expected_total_rows) if args.expected_total_rows else None,
        "sample_fraction_requested": args.sample_frac,
        "seed": args.seed,
        "chunksize": args.chunksize,
        "total_rows_sampled": total_sampled,
        "class_counts_original": dict(sorted(original_counts.items())),
        "class_counts_sampled": dict(sorted(sampled_counts.items())),
        "timestamp_min_sampled": timestamp_min,
        "timestamp_max_sampled": timestamp_max,
        "output_csv": str(output_csv.resolve()),
        "columns_written": canonical_header,
        "dropped_identifier_columns": sorted(IGNORED_IDENTIFIER_COLUMNS),
        "method_notes": [
            "Sampling is performed independently by class within each input chunk with one seeded NumPy Generator.",
            "Because each chunk/class group is rounded independently, per-class sample totals are close to, but may not be exactly, 10% of full-class totals.",
            "All predictors, ts, label, and type are retained. uid is dropped because the audit found it only in Network_dataset_6.csv and it is an identifier.",
            "The sample is not timestamp-sorted. A later stage must sort by ts before forming consecutive flow groups.",
            "MITM is retained in this raw sample for auditability; the multiclass experiment may exclude it as described in IEDL-IDS.",
            "No feature-selection rule is applied here because the paper's claimed 21 dimensions do not match either audited zero-only or zero/dash/blank threshold results.",
        ],
    }
    (args.out_dir / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== SAMPLING SUMMARY ===")
    print(f"Rows read: {total_rows:,}")
    print(f"Rows sampled: {total_sampled:,} ({total_sampled / total_rows:.3%})")
    print("Per-class sampled rows:")
    for row in class_rows:
        print(f"  {row['type']:<12} {row['sampled_rows']:>10,} / {row['original_rows']:>10,} ({row['sample_fraction_observed']:.3%})")
    print(f"\nOutput: {output_csv}")
    print(f"Reports: {args.out_dir}")
    print("NOTE: Feature selection and timestamp sorting have intentionally not been applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
