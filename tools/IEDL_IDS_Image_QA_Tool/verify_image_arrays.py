#!/usr/bin/env python3
"""Validate locally generated IEDL-IDS NPZ image arrays and save class previews.

This script reads only images_train.npz, images_validation.npz,
images_test.npz and preprocessing_metadata.json. It does NOT reopen the
original 23 CSV shards or the sampled CSV.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

SPLITS = ("train", "validation", "test")
EXPECTED_SHAPE = (21, 21, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="Directory containing images_*.npz and preprocessing_metadata.json")
    ap.add_argument("--out-dir", default=None, help="Preview/report directory (default: <data-dir>/qa)")
    ap.add_argument("--previews-per-class", type=int, default=1)
    args = ap.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else data_dir / "qa"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = data_dir / "preprocessing_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    id_to_class = {int(v): k for k, v in meta["class_to_id"].items()}
    report = {"data_dir": str(data_dir), "splits": {}, "checks": {}, "notes": []}
    all_ok = True

    for split in SPLITS:
        p = data_dir / f"images_{split}.npz"
        if not p.exists():
            raise FileNotFoundError(p)
        with np.load(p, allow_pickle=False) as z:
            if "X" not in z or "y" not in z:
                raise ValueError(f"{p.name} must contain arrays named X and y")
            X, y = z["X"], z["y"]
        checks = {
            "x_shape_ok": X.ndim == 4 and tuple(X.shape[1:]) == EXPECTED_SHAPE,
            "x_y_count_match": X.shape[0] == y.shape[0],
            "y_is_1d": y.ndim == 1,
            "dtype_is_uint8": X.dtype == np.uint8,
            "all_labels_known": bool(np.isin(y, list(id_to_class)).all()),
            "finite_pixels": bool(np.isfinite(X).all()),
            "pixel_range_0_255": int(X.min()) >= 0 and int(X.max()) <= 255,
        }
        ok = all(checks.values())
        all_ok &= ok
        unique, counts = np.unique(y, return_counts=True)
        class_counts = {id_to_class[int(k)]: int(v) for k, v in zip(unique, counts)}
        report["splits"][split] = {
            "images": int(X.shape[0]), "x_shape": list(X.shape),
            "x_dtype": str(X.dtype), "y_shape": list(y.shape),
            "pixel_min": int(X.min()), "pixel_max": int(X.max()),
            "class_counts": class_counts, "checks": checks, "passed": ok,
        }
        print(f"{split}: X={X.shape}, dtype={X.dtype}, y={y.shape}, "
              f"pixels=[{int(X.min())},{int(X.max())}], passed={ok}")
        print("  classes:", class_counts)

        # Save the first requested number of previews per class.
        try:
            from PIL import Image, ImageDraw
            for class_id, class_name in id_to_class.items():
                idxs = np.flatnonzero(y == class_id)[:args.previews_per_class]
                for n, idx in enumerate(idxs, start=1):
                    img = Image.fromarray(X[idx], mode="RGB")
                    img.save(out_dir / f"{split}_{class_name}_{n}.png")
        except ImportError:
            report["notes"].append("Pillow not installed; PNG previews were not generated. Install with: py -m pip install pillow")
            break

    report["checks"]["all_array_checks_passed"] = bool(all_ok)
    report["checks"]["expected_image_shape"] = list(EXPECTED_SHAPE)
    report["checks"]["expected_class_count"] = len(id_to_class)
    report["checks"]["class_maps_to_9_classes"] = len(id_to_class) == 9
    report["checks"]["all_splits_present"] = len(report["splits"]) == 3
    report["checks"]["overall_passed"] = bool(all_ok and len(report["splits"]) == 3 and len(id_to_class) == 9)
    (out_dir / "image_array_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nOVERALL:", "PASS" if report["checks"]["overall_passed"] else "CHECK REQUIRED")
    print("Report/previews:", out_dir)
    if not report["checks"]["overall_passed"]:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
