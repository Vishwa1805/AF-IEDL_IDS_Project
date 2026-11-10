#!/usr/bin/env python3
"""IEDL-IDS feature audit v2.

Read-only audit of the 23 ToN_IoT Network_dataset_*.csv shards.
Separates numeric zero values from '-' placeholders and blank strings.
Treats optional 'uid' as an identifier, not a model feature, and checks schema once per file.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path
import pandas as pd

EXPECTED_SHARDS = [f"Network_dataset_{i}.csv" for i in range(1, 24)]
EXPECTED_TOTAL_ROWS = 22_339_021
EXPECTED_CLASSES = {
    "backdoor": 508_116, "ddos": 6_165_008, "dos": 3_375_328,
    "injection": 452_659, "mitm": 1_052, "normal": 796_380,
    "password": 1_718_568, "ransomware": 72_805,
    "scanning": 7_140_161, "xss": 2_108_944,
}
BOOL_FEATURES = {
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_resumed", "ssl_established", "weird_notice",
}
EXCLUDED_COLUMNS = {"ts", "uid", "label", "type"}


def norm_label(x: object) -> str:
    return str(x).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Folder containing Network_dataset_1.csv ... _23.csv")
    ap.add_argument("--out-dir", required=True, help="Output folder for audit reports")
    ap.add_argument("--chunksize", type=int, default=200_000)
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [n for n in EXPECTED_SHARDS if not (data_dir / n).is_file()]
    if missing:
        print("Missing expected shards:")
        print("\n".join(f" - {x}" for x in missing))
        return 2

    # Canonical schema is taken from shard 1; compare once per file, not per chunk.
    canonical = list(pd.read_csv(data_dir / EXPECTED_SHARDS[0], nrows=0, dtype="string", keep_default_na=False).columns)
    canonical_set = set(canonical)
    schema_report = []
    allowed_uid_shards = []
    for name in EXPECTED_SHARDS:
        cols = list(pd.read_csv(data_dir / name, nrows=0, dtype="string", keep_default_na=False).columns)
        cset = set(cols)
        if cset == canonical_set:
            continue
        extra, absent = sorted(cset - canonical_set), sorted(canonical_set - cset)
        if extra == ["uid"] and not absent:
            allowed_uid_shards.append(name)
            schema_report.append({"file": name, "status": "allowed_identifier_only", "extra_columns": extra, "missing_columns": absent})
        else:
            schema_report.append({"file": name, "status": "mismatch", "extra_columns": extra, "missing_columns": absent})

    feature_cols = [c for c in canonical if c not in EXCLUDED_COLUMNS]
    total_rows = 0
    class_counts: Counter[str] = Counter()
    binary_mismatch = 0
    zero_counts: Counter[str] = Counter()
    dash_counts: Counter[str] = Counter()
    blank_counts: Counter[str] = Counter()
    false_counts: Counter[str] = Counter()
    nonempty_counts: Counter[str] = Counter()
    shard_rows = []
    ts_min = ts_max = None

    for i, name in enumerate(EXPECTED_SHARDS, 1):
        path = data_dir / name
        print(f"[{i}/23] {name}", flush=True)
        rows = 0
        shard_classes: Counter[str] = Counter()
        for chunk in pd.read_csv(path, chunksize=args.chunksize, dtype="string", keep_default_na=False, low_memory=False):
            # uid may appear only on one shard. It is an identifier outside the documented feature schema.
            if "uid" in chunk.columns:
                chunk = chunk.drop(columns=["uid"])
            labels = chunk["type"].map(norm_label)
            vc = labels.value_counts(dropna=False)
            for k, v in vc.items():
                class_counts[str(k)] += int(v)
                shard_classes[str(k)] += int(v)
            binary = pd.to_numeric(chunk["label"], errors="coerce")
            expected = (labels != "normal").astype("int8")
            binary_mismatch += int((binary.notna() & (binary.astype("Int64") != expected)).sum())

            for col in feature_cols:
                s = chunk[col].astype("string").str.strip()
                dash_counts[col] += int(s.eq("-").fillna(False).sum())
                blank_counts[col] += int(s.eq("").fillna(False).sum())
                false_counts[col] += int(s.str.upper().eq("F").fillna(False).sum())
                num = pd.to_numeric(s, errors="coerce")
                zero_counts[col] += int(num.eq(0).fillna(False).sum())
                nonempty_counts[col] += int((~s.isin(["", "-"])).sum())

            if "ts" in chunk.columns:
                ts = pd.to_numeric(chunk["ts"], errors="coerce")
                if ts.notna().any():
                    lo, hi = int(ts.min()), int(ts.max())
                    ts_min = lo if ts_min is None else min(ts_min, lo)
                    ts_max = hi if ts_max is None else max(ts_max, hi)
            n = len(chunk)
            rows += n
            total_rows += n
        shard_rows.append({"file": name, "rows": rows, "class_counts": dict(sorted(shard_classes.items()))})

    feature_rows = []
    for col in feature_cols:
        denom = total_rows
        z = zero_counts[col] / denom
        d = dash_counts[col] / denom
        b = blank_counts[col] / denom
        f = false_counts[col] / denom
        combined = (zero_counts[col] + dash_counts[col] + blank_counts[col]) / denom
        feature_rows.append({
            "feature": col,
            "documented_boolean": col in BOOL_FEATURES,
            "numeric_zero_count": zero_counts[col],
            "numeric_zero_percent": round(z * 100, 5),
            "dash_count": dash_counts[col],
            "dash_percent": round(d * 100, 5),
            "blank_count": blank_counts[col],
            "blank_percent": round(b * 100, 5),
            "literal_F_count": false_counts[col],
            "literal_F_percent": round(f * 100, 5),
            "zero_only_gt_80_percent": z > 0.80,
            "zero_dash_blank_gt_80_percent": combined > 0.80,
        })

    classes_match = dict(sorted(class_counts.items())) == dict(sorted(EXPECTED_CLASSES.items()))
    report = {
        "shard_count": len(EXPECTED_SHARDS),
        "total_rows": total_rows,
        "expected_total_rows": EXPECTED_TOTAL_ROWS,
        "total_rows_match": total_rows == EXPECTED_TOTAL_ROWS,
        "canonical_columns": canonical,
        "canonical_column_count": len(canonical),
        "model_candidate_predictors_excluding_ts_uid_labels": len(feature_cols),
        "uid_extra_shards_treated_as_identifier": allowed_uid_shards,
        "schema_review": schema_report,
        "timestamp_range_epoch": [ts_min, ts_max],
        "class_counts": dict(sorted(class_counts.items())),
        "class_counts_match_metadata": classes_match,
        "label_type_mismatch_count": binary_mismatch,
        "feature_rule_note": "The paper states >80% zero values. This report separates numeric zero from '-' and blank placeholders; do not equate them without explicit methodological justification.",
        "feature_count_removed_by_numeric_zero_only": sum(r["zero_only_gt_80_percent"] for r in feature_rows),
        "feature_count_removed_by_zero_dash_blank_combined": sum(r["zero_dash_blank_gt_80_percent"] for r in feature_rows),
        "shards": shard_rows,
    }
    pd.DataFrame(feature_rows).sort_values(["numeric_zero_percent", "dash_percent"], ascending=False).to_csv(out_dir / "feature_presence_audit_v2.csv", index=False)
    pd.DataFrame([{"class": k, "records": v} for k, v in sorted(class_counts.items())]).to_csv(out_dir / "class_counts_v2.csv", index=False)
    pd.DataFrame([{ "file": s["file"], "rows": s["rows"] } for s in shard_rows]).to_csv(out_dir / "shard_counts_v2.csv", index=False)
    (out_dir / "audit_summary_v2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== AUDIT V2 SUMMARY ===")
    print(f"Rows: {total_rows:,} (expected {EXPECTED_TOTAL_ROWS:,})")
    print(f"Class counts match metadata: {classes_match}")
    print(f"Label/type disagreements: {binary_mismatch:,}")
    print(f"Allowed uid identifier shards: {allowed_uid_shards}")
    print(f"Actual schema mismatches: {sum(x['status']=='mismatch' for x in schema_report)}")
    print(f"Features >80% numeric zero ONLY: {report['feature_count_removed_by_numeric_zero_only']}")
    print(f"Features >80% zero OR dash OR blank: {report['feature_count_removed_by_zero_dash_blank_combined']}")
    print(f"Reports: {out_dir}")
    if total_rows != EXPECTED_TOTAL_ROWS or not classes_match or binary_mismatch or any(x["status"] == "mismatch" for x in schema_report):
        print("STATUS: CHECK REQUIRED")
        return 1
    print("STATUS: STRUCTURAL DATA CHECKS PASS; FEATURE RULE STILL REQUIRES METHODOLOGY REVIEW")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
