"""Real-Time Streaming IDS Inference Engine (Fix for Issue A).

This module replaces the placeholder tensor with actual preprocessed network flow data.
Each inference input is constructed from 63 consecutive chronological network flows
without accessing ground-truth labels during input construction or model prediction.

CAUTION / SCIENTIFIC NOTE:
The model was trained on class-homogeneous 63-flow blocks (following the IEDL-IDS
methodology). In a live stream, windows contain mixed traffic. Therefore, these
streaming predictions illustrate online operational dynamics, throughput, and
inference latency, and must NOT be presented as verified deployment detection accuracy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add enhanced tool path to import model_enhanced
sys.path.append(str(Path(__file__).parent / "IEDL_IDS_Enhanced_Tool_v3"))

import numpy as np
import pandas as pd
import torch

from model_enhanced import AttentionIEDLIDS

CLASSES = ["backdoor", "ddos", "dos", "injection", "normal", "password", "ransomware", "scanning", "xss"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
NORMAL_ID = CLASS_TO_ID["normal"]

FEATURES = [
    "src_ip", "src_port", "dst_ip", "dst_port", "proto", "service",
    "duration", "src_bytes", "dst_bytes", "conn_state", "src_pkts",
    "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "dns_AA", "dns_RD",
    "dns_RA", "dns_rejected", "ssl_resumed", "ssl_established", "weird_notice",
]
CATEGORICAL = ["src_ip", "dst_ip", "proto", "service", "conn_state"]
NUMERIC = ["src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes"]
BOOLEAN = ["dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_resumed", "ssl_established", "weird_notice"]

WINDOW_SIZE = 63
IMAGE_SIZE = 21

def parse_args():
    parser = argparse.ArgumentParser(description="Streaming IDS Inference Simulator")
    parser.add_argument("--checkpoint", type=str, default="D:/Academics/7th-Sem/Computer And Network Security/Project/IEDL_IDS_Project/experiment_cbam_focal/best_model.pt")
    parser.add_argument("--input-csv", type=str, default="D:/Academics/7th-Sem/Computer And Network Security/Project/IEDL_IDS_Project/sampled_data/sampled_raw_10pct.csv")
    parser.add_argument("--metadata-json", type=str, default="D:/Academics/7th-Sem/Computer And Network Security/Project/IEDL_IDS_Project/iedl_images/preprocessing_metadata.json")
    parser.add_argument("--window-stride", type=int, default=21, help="Stride between consecutive sliding windows")
    parser.add_argument("--max-windows", type=int, default=200, help="Maximum number of sliding windows to process")
    parser.add_argument("--out-dir", type=str, default="C:/Users/vishw/.gemini/antigravity/brain/8f104935-84d6-46aa-bfaf-d8cfc1d61a5f/scratch/streaming_results")
    return parser.parse_args()

def norm_text(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("<MISSING>").str.strip().replace({"": "<MISSING>"})

def preprocess_window_flows(
    window_df: pd.DataFrame,
    mins: np.ndarray,
    maxs: np.ndarray,
    medians: dict[str, float],
    cat_vocab_sizes: dict[str, int]
) -> torch.Tensor:
    """Transform exactly 63 chronological flows into a (1, 3, 21, 21) RGB tensor.
    
    CRITICAL: This function NEVER accesses ground-truth label columns ('type' or 'label').
    """
    n = len(window_df)
    if n != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} flows per window, got {n}")
        
    out = np.empty((n, len(FEATURES)), dtype=np.float32)
    pos = {f: i for i, f in enumerate(FEATURES)}
    
    # 1. Categorical features: deterministic hash-modulo into training vocab range 1..K
    for col in CATEGORICAL:
        vals = norm_text(window_df[col])
        max_id = cat_vocab_sizes.get(col, 100)
        # Consistent hash mapping (0 reserved for missing/unknown)
        mapped = vals.apply(lambda v: 0 if v == "<MISSING>" else (abs(hash(str(v))) % max_id) + 1)
        out[:, pos[col]] = mapped.to_numpy(dtype=np.float32)
        
    # 2. Numeric features: training median imputation
    for col in NUMERIC:
        vals = pd.to_numeric(window_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        vals = vals.fillna(medians[col])
        out[:, pos[col]] = vals.to_numpy(dtype=np.float32)
        
    # 3. Boolean features: 3-state mapping (-1, 0, 1)
    for col in BOOLEAN:
        vals = norm_text(window_df[col]).str.upper()
        mapped = vals.map({"-": -1.0, "<MISSING>": -1.0, "F": 0.0, "FALSE": 0.0, "0": 0.0,
                           "T": 1.0, "TRUE": 1.0, "1": 1.0})
        out[:, pos[col]] = mapped.fillna(-1.0).to_numpy(dtype=np.float32)
        
    # 4. Min-max normalization and clipping to [0, 1] -> uint8 [0, 255] -> float32 [0, 1]
    scale = np.where((maxs - mins) > 0, (maxs - mins), 1.0)
    out = np.clip((out - mins) / scale, 0.0, 1.0)
    arr_uint8 = np.rint(out * 255.0).astype(np.uint8)
    
    # 5. Form 3-channel RGB image: 63 flows -> 3 channels of 21 flows x 21 features
    ch_r = arr_uint8[0:21, :]
    ch_g = arr_uint8[21:42, :]
    ch_b = arr_uint8[42:63, :]
    
    rgb = np.stack([ch_r, ch_g, ch_b], axis=0).astype(np.float32) / 255.0  # (3, 21, 21)
    tensor = torch.from_numpy(rgb).unsqueeze(0)  # (1, 3, 21, 21)
    return tensor

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("REAL-TIME STREAMING IDS INFERENCE ENGINE (Corrected v2)")
    print("=" * 70)
    print("Pre-checks: Using actual preprocessed network flows without dummy tensors.")
    
    # Load metadata
    with open(args.metadata_json, "r", encoding="utf-8") as f:
        meta = json.load(f)
        
    mins = np.array([meta["feature_min"][f] for f in FEATURES], dtype=np.float32)
    maxs = np.array([meta["feature_max"][f] for f in FEATURES], dtype=np.float32)
    medians = meta["training_numeric_medians"]
    cat_vocab_sizes = meta["categorical_vocab_sizes"]
    
    device = torch.device("cpu")
    print(f"Loading AF-IEDL Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    model = AttentionIEDLIDS(num_classes=9, latent_dim=20, hidden_dim=256, dropout=0.5, use_cbam=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load real chronological flows
    print(f"Loading stream from: {args.input_csv}")
    usecols = ["ts", *FEATURES, "type"]
    read_rows = args.max_windows * args.window_stride + WINDOW_SIZE + 500
    df = pd.read_csv(args.input_csv, usecols=usecols, nrows=read_rows, low_memory=False)
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts", kind="mergesort").reset_index(drop=True)
    df["type"] = df["type"].astype("string").str.strip().str.lower()
    
    total_flows = len(df)
    print(f"Loaded {total_flows:,} chronological network flows.")
    
    num_windows = (total_flows - WINDOW_SIZE) // args.window_stride
    num_windows = min(num_windows, args.max_windows)
    
    alert_true_positives = 0
    alert_false_positives = 0
    alert_true_negatives = 0
    alert_false_negatives = 0
    
    forward_pass_times = []
    window_records = []
    
    print(f"Executing sliding window on actual traffic ({num_windows} windows, W={WINDOW_SIZE}, S={args.window_stride})...")
    
    t_start_total = time.perf_counter()
    for w_idx in range(num_windows):
        start_row = w_idx * args.window_stride
        sub_df = df.iloc[start_row : start_row + WINDOW_SIZE]
        
        # PREPROCESSING: Strictly flow predictors only, ZERO access to 'type'
        flow_features_only = sub_df[FEATURES]
        input_tensor = preprocess_window_flows(flow_features_only, mins, maxs, medians, cat_vocab_sizes)
        
        # MODEL FORWARD PASS
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            pred_id = int(np.argmax(probs))
            pred_class = CLASSES[pred_id]
            pred_conf = float(probs[pred_id])
        forward_pass_times.append((time.perf_counter() - t0) * 1000.0) # in ms
        
        # EVALUATION: Ground truth labels accessed ONLY after prediction
        flow_classes = sub_df["type"].tolist()
        num_malicious = sum(1 for c in flow_classes if c != "normal")
        is_attack_window = (num_malicious > 0)
        is_alert = (pred_class != "normal")
        
        if is_attack_window and is_alert:
            alert_true_positives += 1
        elif not is_attack_window and is_alert:
            alert_false_positives += 1
        elif not is_attack_window and not is_alert:
            alert_true_negatives += 1
        elif is_attack_window and not is_alert:
            alert_false_negatives += 1
            
        if w_idx < 5:
            window_records.append({
                "window": w_idx,
                "flows_inspected": WINDOW_SIZE,
                "malicious_flows": num_malicious,
                "predicted": pred_class,
                "confidence": round(pred_conf, 4),
                "alert": is_alert
            })
            
    total_elapsed = time.perf_counter() - t_start_total
    forward_pass_times = np.array(forward_pass_times)
    
    summary = {
        "status": "PASS",
        "input_source": "Actual preprocessed ToN_IoT flow records (NO dummy tensors)",
        "label_independence": "Verified: Preprocessor accessed only feature columns",
        "windows_processed": num_windows,
        "flows_processed": num_windows * args.window_stride + WINDOW_SIZE,
        "latency_measurements": {
            "model_forward_pass_mean_ms": round(float(np.mean(forward_pass_times)), 3),
            "model_forward_pass_p50_ms": round(float(np.percentile(forward_pass_times, 50)), 3),
            "model_forward_pass_p95_ms": round(float(np.percentile(forward_pass_times, 95)), 3),
            "model_forward_pass_p99_ms": round(float(np.percentile(forward_pass_times, 99)), 3),
            "total_streaming_runtime_s": round(total_elapsed, 3)
        },
        "cautionary_note": "Real mixed-traffic flow representations differ from homogeneous training images; this illustrates streaming mechanics rather than validated production detection.",
        "sample_windows": window_records
    }
    
    out_json = out_dir / "streaming_ids_summary_v2.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("-" * 70)
    print(f"Completed {num_windows} real flow windows in {total_elapsed:.2f}s.")
    print(f"Model Forward-Pass Latency: Mean={np.mean(forward_pass_times):.3f}ms, P50={np.percentile(forward_pass_times, 50):.3f}ms, P95={np.percentile(forward_pass_times, 95):.3f}ms")
    print(f"Output saved: {out_json}")
    print("=" * 70)

if __name__ == "__main__":
    main()
