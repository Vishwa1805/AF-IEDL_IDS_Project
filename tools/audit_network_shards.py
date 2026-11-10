#!/usr/bin/env python3
"""Streaming audit for the 23 ToN_IoT Processed_Network_dataset CSV shards.

This script does NOT sample, shuffle, normalize, or modify any source CSV.
It validates shard completeness/schema, counts labels, checks label consistency,
and estimates the paper's >80%-zero/absent feature filter.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

EXPECTED_SHARDS = [f"Network_dataset_{i}.csv" for i in range(1, 24)]
EXPECTED_TOTAL_ROWS = 22_339_021
EXPECTED_CLASS_COUNTS = {
    "backdoor": 508_116,
    "ddos": 6_165_008,
    "dos": 3_375_328,
    "injection": 452_659,
    "mitm": 1_052,
    "password": 1_718_568,
    "ransomware": 72_805,
    "scanning": 7_140_161,
    "xss": 2_108_944,
    "normal": 796_380,
}


def norm_label(value: object) -> str:
    return str(value).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Folder containing Network_dataset_1.csv ... _23.csv")
    ap.add_argument("--out-dir", default="audit_output", help="Directory for small audit reports")
    ap.add_argument("--chunksize", type=int, default=200_000, help="Rows per CSV read chunk")
    ap.add_argument("--allow-incomplete", action="store_true", help="Audit available shards even if some of the 23 are missing")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not data_dir.is_dir():
        print(f"ERROR: data folder does not exist: {data_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in EXPECTED_SHARDS if not (data_dir / name).is_file()]
    found = [data_dir / name for name in EXPECTED_SHARDS if (data_dir / name).is_file()]
    # Case-insensitive recovery for Windows filesystems / renamed extensions.
    if not found:
        found = sorted(data_dir.glob("Network_dataset_*.csv"), key=lambda p: int(re.search(r"_(\d+)\.csv$", p.name, re.I).group(1)) if re.search(r"_(\d+)\.csv$", p.name, re.I) else 9999)
    if missing and not args.allow_incomplete:
        print("ERROR: Missing expected shards:")
        for name in missing:
            print(f"  - {name}")
        print("Do not continue to sampling until all 23 expected files are present.")
        return 2
    if not found:
        print("ERROR: No Network_dataset_*.csv files found.", file=sys.stderr)
        return 2

    print(f"Found {len(found)} shard(s) in: {data_dir}")
    class_counts: Counter[str] = Counter()
    shard_rows: list[dict] = []
    feature_total: Counter[str] = Counter()
    feature_absent: Counter[str] = Counter()
    label_mismatch = 0
    total_rows = 0
    canonical_cols: list[str] | None = None
    schema_mismatches: list[dict] = []
    min_ts = None
    max_ts = None

    for idx, path in enumerate(found, 1):
        print(f"[{idx}/{len(found)}] Scanning {path.name} ...", flush=True)
        rows_this_shard = 0
        shard_class_counts: Counter[str] = Counter()
        try:
            for chunk in pd.read_csv(
                path,
                chunksize=args.chunksize,
                dtype="string",
                keep_default_na=False,
                low_memory=False,
                encoding_errors="replace",
            ):
                cols = list(chunk.columns)
                if canonical_cols is None:
                    canonical_cols = cols
                    required = {"label", "type"}
                    if not required.issubset(cols):
                        raise ValueError(f"Missing required target columns {sorted(required)} in {path.name}")
                elif cols != canonical_cols:
                    schema_mismatches.append({
                        "file": path.name,
                        "expected_columns": canonical_cols,
                        "actual_columns": cols,
                    })

                chunk_rows = len(chunk)
                rows_this_shard += chunk_rows
                total_rows += chunk_rows

                labels = chunk["type"].map(norm_label)
                vc = labels.value_counts(dropna=False)
                for label, count in vc.items():
                    class_counts[str(label)] += int(count)
                    shard_class_counts[str(label)] += int(count)

                binary = pd.to_numeric(chunk["label"], errors="coerce")
                expected_binary = (labels != "normal").astype("int8")
                label_mismatch += int((binary.notna() & (binary.astype("Int64") != expected_binary)).sum())

                feature_cols = [c for c in cols if c not in {"ts", "label", "type"}]
                for col in feature_cols:
                    s = chunk[col].astype("string").str.strip()
                    # Treat literal '-' / blank as absent for categorical Zeek fields,
                    # and numeric 0 as the paper's zero-value criterion.
                    is_dash_or_blank = s.isin(["-", ""])
                    numeric = pd.to_numeric(s, errors="coerce")
                    is_zero = numeric.eq(0).fillna(False)
                    feature_absent[col] += int((is_dash_or_blank | is_zero).sum())
                    feature_total[col] += int(chunk_rows)

                if "ts" in chunk.columns:
                    ts = pd.to_numeric(chunk["ts"], errors="coerce")
                    if ts.notna().any():
                        lo, hi = int(ts.min()), int(ts.max())
                        min_ts = lo if min_ts is None else min(min_ts, lo)
                        max_ts = hi if max_ts is None else max(max_ts, hi)
        except Exception as exc:
            print(f"ERROR while reading {path.name}: {exc}", file=sys.stderr)
            return 3

        shard_rows.append({
            "file": path.name,
            "rows": rows_this_shard,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
            "class_counts": dict(sorted(shard_class_counts.items())),
        })
        print(f"    rows: {rows_this_shard:,}")

    if canonical_cols is None:
        print("ERROR: no CSV headers read.", file=sys.stderr)
        return 3

    predictor_cols = [c for c in canonical_cols if c not in {"ts", "label", "type"}]
    feature_rows = []
    removed = []
    retained = []
    for col in predictor_cols:
        denom = feature_total[col]
        rate = feature_absent[col] / denom if denom else 0.0
        is_removed = rate > 0.80
        (removed if is_removed else retained).append(col)
        feature_rows.append({
            "feature": col,
            "zero_or_dash_fraction": round(rate, 8),
            "zero_or_dash_percent": round(rate * 100, 4),
            "removed_if_threshold_gt_80_percent": is_removed,
        })

    class_counts_sorted = dict(sorted(class_counts.items()))
    rows_match = total_rows == EXPECTED_TOTAL_ROWS
    class_match = class_counts_sorted == dict(sorted(EXPECTED_CLASS_COUNTS.items()))
    report = {
        "data_directory": str(data_dir),
        "shard_count_found": len(found),
        "expected_shard_count": 23,
        "missing_shards": missing,
        "total_rows": total_rows,
        "expected_total_rows_from_metadata": EXPECTED_TOTAL_ROWS,
        "row_count_matches_metadata": rows_match,
        "columns": canonical_cols,
        "column_count": len(canonical_cols),
        "predictor_count_excluding_ts_and_labels": len(predictor_cols),
        "timestamp_present": "ts" in canonical_cols,
        "timestamp_min_epoch": min_ts,
        "timestamp_max_epoch": max_ts,
        "class_counts": class_counts_sorted,
        "class_counts_match_metadata": class_match,
        "label_type_mismatch_count": label_mismatch,
        "schema_mismatches": schema_mismatches,
        "zero_or_dash_threshold": 0.80,
        "removed_features_gt_80_percent": removed,
        "retained_features_le_80_percent": retained,
        "retained_feature_count": len(retained),
        "shards": shard_rows,
        "notes": [
            "This is an audit only; source files are not modified.",
            "The paper says features with >80% zero values are removed. For Zeek categorical fields, this audit additionally treats '-' and blank as absent placeholders; confirm this interpretation against the feature documentation before finalizing preprocessing.",
            "The timestamp is excluded from model features but retained for ordering flows before image construction.",
            "The paper does not fully specify the exact train/validation/test split ratio or image-label assignment policy; these must be documented as implementation decisions.",
        ],
    }

    pd.DataFrame(feature_rows).sort_values("zero_or_dash_percent", ascending=False).to_csv(out_dir / "feature_presence_audit.csv", index=False)
    pd.DataFrame([{"class": k, "records": v} for k, v in class_counts_sorted.items()]).to_csv(out_dir / "class_counts.csv", index=False)
    pd.DataFrame(shard_rows).drop(columns=["class_counts"]).to_csv(out_dir / "shard_counts.csv", index=False)
    (out_dir / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== AUDIT SUMMARY ===")
    print(f"Shards found: {len(found)}/23")
    print(f"Total rows: {total_rows:,} (metadata: {EXPECTED_TOTAL_ROWS:,}; match: {rows_match})")
    print(f"Columns: {len(canonical_cols)}; predictors excluding ts/labels: {len(predictor_cols)}")
    print(f"Timestamp present: {'ts' in canonical_cols}; range: {min_ts} .. {max_ts}")
    print("Class counts:")
    for label, count in class_counts_sorted.items():
        print(f"  {label:12s} {count:>12,}   ({count/total_rows:.3%})")
    print(f"Label/type disagreements: {label_mismatch:,}")
    print(f"Schema mismatches: {len(schema_mismatches)}")
    print(f"Features removed (>80% zero/dash): {len(removed)}")
    print("  " + ", ".join(removed))
    print(f"Features retained: {len(retained)}")
    print("  " + ", ".join(retained))
    print(f"Reports written to: {out_dir}")

    if missing or schema_mismatches or not rows_match or not class_match or label_mismatch:
        print("\nAUDIT STATUS: CHECK REQUIRED — do not begin sampling until discrepancies are reviewed.")
        return 1
    print("\nAUDIT STATUS: PASS — row/class totals match the metadata workbook.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
