
import argparse
import tempfile
from pathlib import Path

import pandas as pd

CLASSES = [
    "backdoor", "ddos", "dos", "injection", "normal",
    "password", "ransomware", "scanning", "xss"
]

CATEGORICAL = [
    "src_ip", "dst_ip", "proto", "service", "conn_state"
]

NUMERIC = [
    "src_port", "dst_port", "duration", "src_bytes", "dst_bytes",
    "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"
]

FEATURES = CATEGORICAL + NUMERIC + [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected",
    "ssl_resumed", "ssl_established", "weird_notice"
]

GROUP_SIZE = 63


def norm(series):
    return (
        series.astype("string")
        .fillna("<MISSING>")
        .str.strip()
        .replace({"": "<MISSING>"})
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    source = Path(args.input_csv)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="iedl_drift_") as temp:
        temp = Path(temp)
        written = set()

        print("Reading sampled flows and partitioning by class...")

        for chunk in pd.read_csv(
            source,
            usecols=["ts", "type"] + FEATURES,
            chunksize=150000,
            low_memory=False
        ):
            chunk["type"] = (
                chunk["type"].astype("string").str.strip().str.lower()
            )
            chunk["ts"] = pd.to_numeric(chunk["ts"], errors="coerce")
            chunk = chunk.dropna(subset=["ts"])

            for cls in CLASSES:
                part = chunk.loc[chunk["type"] == cls,
                                 ["ts"] + FEATURES]
                if part.empty:
                    continue

                path = temp / f"{cls}.csv"
                part.to_csv(
                    path,
                    mode="a",
                    index=False,
                    header=(cls not in written)
                )
                written.add(cls)

        vocab = {c: set() for c in CATEGORICAL}
        boundaries = {}

        print("Calculating training vocabularies...")

        for cls in CLASSES:
            path = temp / f"{cls}.csv"
            if not path.exists():
                continue

            df = pd.read_csv(path, low_memory=False)
            df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)

            groups = len(df) // GROUP_SIZE
            usable = groups * GROUP_SIZE
            train_end = int(groups * 0.80) * GROUP_SIZE
            val_end = (
                int(groups * 0.80) + int(groups * 0.10)
            ) * GROUP_SIZE

            boundaries[cls] = (train_end, val_end, usable)

            train = df.iloc[:train_end]

            for col in CATEGORICAL:
                vocab[col].update(norm(train[col]).unique().tolist())

        time_rows = []
        category_rows = []
        numeric_rows = []

        print("Comparing chronological splits...")

        for cls in CLASSES:
            path = temp / f"{cls}.csv"
            if cls not in boundaries:
                continue

            df = pd.read_csv(path, low_memory=False)
            df = df.sort_values("ts", kind="mergesort").reset_index(drop=True)

            train_end, val_end, usable = boundaries[cls]

            segments = {
                "train": df.iloc[:train_end],
                "validation": df.iloc[train_end:val_end],
                "test": df.iloc[val_end:usable]
            }

            for split, part in segments.items():
                if part.empty:
                    continue

                start = pd.to_datetime(
                    part["ts"].iloc[0], unit="s", utc=True
                )
                end = pd.to_datetime(
                    part["ts"].iloc[-1], unit="s", utc=True
                )

                time_rows.append({
                    "class": cls,
                    "split": split,
                    "rows": len(part),
                    "start_utc": str(start),
                    "end_utc": str(end),
                    "duration_hours": (
                        part["ts"].iloc[-1] - part["ts"].iloc[0]
                    ) / 3600
                })

                for col in CATEGORICAL:
                    vals = norm(part[col])
                    unseen = ~vals.isin(vocab[col])

                    category_rows.append({
                        "class": cls,
                        "split": split,
                        "feature": col,
                        "total_values": len(vals),
                        "unseen_values": int(unseen.sum()),
                        "unseen_rate": float(unseen.mean()),
                        "distinct_values": int(vals.nunique()),
                        "distinct_unseen": int(vals[unseen].nunique())
                    })

                for col in NUMERIC:
                    vals = pd.to_numeric(
                        part[col], errors="coerce"
                    ).dropna()

                    if len(vals):
                        numeric_rows.append({
                            "class": cls,
                            "split": split,
                            "feature": col,
                            "median": float(vals.median()),
                            "p90": float(vals.quantile(0.90)),
                            "p99": float(vals.quantile(0.99))
                        })

        pd.DataFrame(time_rows).to_csv(
            output / "class_time_ranges.csv", index=False
        )
        pd.DataFrame(category_rows).to_csv(
            output / "unseen_category_rates.csv", index=False
        )
        pd.DataFrame(numeric_rows).to_csv(
            output / "numeric_feature_summary.csv", index=False
        )

    print("\nRAW SPLIT DRIFT AUDIT COMPLETED")
    print(f"Reports saved in: {output}")


if __name__ == "__main__":
    main()
