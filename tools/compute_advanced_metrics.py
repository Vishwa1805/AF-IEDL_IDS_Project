"""Compute advanced security and calibration metrics: ECE, Brier score, FPR, FNR, Latency percentiles."""
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add model paths
TOOLS_DIR = Path("D:/Academics/7th-Sem/Computer And Network Security/Project/IEDL_IDS_Project/tools")
sys.path.append(str(TOOLS_DIR / "IEDL_IDS_CNN_Tool_v2"))
sys.path.append(str(TOOLS_DIR / "IEDL_IDS_Enhanced_Tool_v3"))

from model import IEDLIDS as BaselineModel
from model_enhanced import AttentionIEDLIDS as EnhancedModel

PROJECT_ROOT = Path("D:/Academics/7th-Sem/Computer And Network Security/Project/IEDL_IDS_Project")
TEST_NPZ = PROJECT_ROOT / "iedl_images" / "images_test.npz"
BASE_CKPT = PROJECT_ROOT / "baseline_iedl_v2_50ep" / "best_model.pt"
AF_CKPT = PROJECT_ROOT / "experiment_cbam_focal" / "best_model.pt"

CLASSES = ["backdoor", "ddos", "dos", "injection", "normal", "password", "ransomware", "scanning", "xss"]
NORMAL_IDX = CLASSES.index("normal")

def compute_ece(probs, targets, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == targets)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def compute_brier(probs, targets, n_classes=9):
    one_hot = np.zeros_like(probs)
    for i, t in enumerate(targets):
        one_hot[i, t] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X.transpose(0, 3, 1, 2)).float() / 255.0
        logits = model(x_tensor)
        probs = F.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
    return probs, preds

def get_binary_ids_metrics(preds, targets):
    # Binary: Normal (Negative) vs Attack (Positive)
    is_attack_true = (targets != NORMAL_IDX)
    is_attack_pred = (preds != NORMAL_IDX)
    
    tp = int(np.sum(is_attack_true & is_attack_pred))
    fp = int(np.sum((~is_attack_true) & is_attack_pred))
    tn = int(np.sum((~is_attack_true) & (~is_attack_pred)))
    fn = int(np.sum(is_attack_true & (~is_attack_pred)))
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "FPR": round(fpr, 4),
        "FNR": round(fnr, 4),
        "Detection_Rate_TPR": round(detection_rate, 4),
        "Specificity_TNR": round(specificity, 4),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

def main():
    print("Loading test dataset...")
    with np.load(TEST_NPZ) as data:
        X, y = data["X"], data["y"].astype(np.int64)
    print(f"Loaded {len(y)} test images.")
    
    # 1. Evaluate Baseline
    print("\n--- Evaluating Baseline Model ---")
    base_model = BaselineModel(num_classes=9, latent_dim=20, hidden_dim=256, dropout=0.5)
    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base_model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    base_probs, base_preds = evaluate_model(base_model, X, y)
    
    base_ece = compute_ece(base_probs, y)
    base_brier = compute_brier(base_probs, y)
    base_binary = get_binary_ids_metrics(base_preds, y)
    
    # 2. Evaluate AF-IEDL
    print("\n--- Evaluating AF-IEDL Model ---")
    af_model = EnhancedModel(num_classes=9, latent_dim=20, hidden_dim=256, dropout=0.5, use_cbam=True)
    af_ckpt = torch.load(AF_CKPT, map_location="cpu", weights_only=False)
    af_model.load_state_dict(af_ckpt["model_state_dict"] if "model_state_dict" in af_ckpt else af_ckpt)
    af_probs, af_preds = evaluate_model(af_model, X, y)
    
    af_ece = compute_ece(af_probs, y)
    af_brier = compute_brier(af_probs, y)
    af_binary = get_binary_ids_metrics(af_preds, y)
    
    # 3. Benchmark CSW Latency Percentiles
    print("\n--- Profiling CSW Latency Distribution ---")
    latencies = []
    x_single = torch.randn(1, 3, 21, 21, dtype=torch.float32)
    # Warmup
    for _ in range(50):
        _ = af_model(x_single)
    # Measure 1,000 runs
    for _ in range(1000):
        t0 = time.perf_counter()
        _ = af_model(x_single)
        latencies.append((time.perf_counter() - t0) * 1000.0) # in ms
        
    latencies = np.array(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_lat = np.mean(latencies)
    
    print("\n" + "=" * 60)
    print("ADVANCED SECURITY & CALIBRATION METRICS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Baseline':<12} | {'AF-IEDL':<12}")
    print("-" * 60)
    print(f"{'Expected Calibration Error (ECE)':<30} | {base_ece:<12.4f} | {af_ece:<12.4f}")
    print(f"{'Brier Score (Lower is Better)':<30} | {base_brier:<12.4f} | {af_brier:<12.4f}")
    print(f"{'Binary False Positive Rate (FPR)':<30} | {base_binary['FPR']:<12.4f} | {af_binary['FPR']:<12.4f}")
    print(f"{'Binary False Negative Rate (FNR)':<30} | {base_binary['FNR']:<12.4f} | {af_binary['FNR']:<12.4f}")
    print(f"{'Detection Rate (Attack Recall)':<30} | {base_binary['Detection_Rate_TPR']:<12.4f} | {af_binary['Detection_Rate_TPR']:<12.4f}")
    print(f"{'Specificity (Normal Recall)':<30} | {base_binary['Specificity_TNR']:<12.4f} | {af_binary['Specificity_TNR']:<12.4f}")
    print("-" * 60)
    print(f"CSW Streaming Inference Latency (CPU Single-Thread):")
    print(f"  Mean: {mean_lat:.3f} ms | P50 (Median): {p50:.3f} ms | P95: {p95:.3f} ms | P99: {p99:.3f} ms")
    print("=" * 60)

if __name__ == "__main__":
    main()
