"""Generate publication-ready figures for the AF-IEDL research paper."""
import csv
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set Matplotlib config dir to avoid warning
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/Temp"), "matplotlib_cache")

CLASSES = ["Backdoor", "DDoS", "DoS", "Injection", "Normal", "Password", "Ransomware", "Scanning", "XSS"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "Paper" / "latex" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "font.family": "sans-serif"
})

def plot_confusion_matrices():
    base_cm_file = PROJECT_ROOT / "baseline_iedl_v2_50ep" / "confusion_matrix.csv"
    af_cm_file = PROJECT_ROOT / "experiment_cbam_focal" / "confusion_matrix.csv"
    
    cm_base = np.loadtxt(base_cm_file, delimiter=",").astype(int)
    cm_af = np.loadtxt(af_cm_file, delimiter=",").astype(int)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Baseline
    im0 = axes[0].imshow(cm_base, cmap="Blues", interpolation="nearest")
    axes[0].set_title("(a) Baseline IEDL-IDS (Cross-Entropy)")
    axes[0].set_xticks(range(len(CLASSES)))
    axes[0].set_yticks(range(len(CLASSES)))
    axes[0].set_xticklabels(CLASSES, rotation=45, ha="right")
    axes[0].set_yticklabels(CLASSES)
    axes[0].set_xlabel("Predicted Class")
    axes[0].set_ylabel("True Class")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm_base[i, j]
            color = "white" if val > cm_base.max() / 2 else "black"
            axes[0].text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)
            
    # AF-IEDL
    im1 = axes[1].imshow(cm_af, cmap="Purples", interpolation="nearest")
    axes[1].set_title("(b) Proposed AF-IEDL (CBAM + Focal Loss)")
    axes[1].set_xticks(range(len(CLASSES)))
    axes[1].set_yticks(range(len(CLASSES)))
    axes[1].set_xticklabels(CLASSES, rotation=45, ha="right")
    axes[1].set_yticklabels(CLASSES)
    axes[1].set_xlabel("Predicted Class")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm_af[i, j]
            color = "white" if val > cm_af.max() / 2 else "black"
            axes[1].text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)
            
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = FIGURES_DIR / "confusion_matrix_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_per_class_f1():
    # Per class F1 data from experiments
    f1_data = {
        "Baseline": [0.8865, 0.8345, 0.8410, 0.8054, 0.9105, 0.9751, 0.8462, 0.9769, 0.9655],
        "CBAM":     [0.8865, 0.8387, 0.8463, 0.7683, 0.8696, 0.9856, 0.5600, 0.9764, 0.9528],
        "Focal":    [0.8770, 0.8368, 0.8457, 0.7179, 0.8881, 0.9643, 0.8696, 0.9737, 0.9592],
        "AF-IEDL":  [0.8962, 0.8337, 0.8417, 0.7302, 0.9254, 0.9687, 0.8696, 0.9800, 0.9439]
    }
    
    x = np.arange(len(CLASSES))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    
    for idx, (model_name, scores) in enumerate(f1_data.items()):
        ax.bar(x + (idx - 1.5) * width, scores, width, label=model_name, color=colors[idx], alpha=0.9)
        
    ax.set_ylabel("Test F1-Score")
    ax.set_title("Per-Class Test F1-Score Across Evaluated Architectures")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax.set_ylim(0.4, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "per_class_f1_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_learning_curves():
    base_hist = PROJECT_ROOT / "baseline_iedl_v2_50ep" / "training_history.csv"
    af_hist = PROJECT_ROOT / "experiment_cbam_focal" / "training_history.csv"
    
    base_rows = []
    with open(base_hist, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            base_rows.append(r)
            
    af_rows = []
    with open(af_hist, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            af_rows.append(r)
            
    epochs = [int(r["epoch"]) for r in base_rows]
    base_val_f1 = [float(r["val_macro_f1"]) for r in base_rows]
    af_val_f1 = [float(r["val_macro_f1"]) for r in af_rows]
    
    base_train_loss = [float(r["train_loss"]) for r in base_rows]
    af_train_loss = [float(r["train_loss"]) for r in af_rows]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    
    # Loss curves
    axes[0].plot(epochs, base_train_loss, label="Baseline (Cross-Entropy)", color="#2b5c8f", lw=2)
    axes[0].plot(epochs, af_train_loss, label="AF-IEDL (Focal Loss, gamma=2)", color="#1b9e77", lw=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("(a) Training Loss Trajectory")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    # Validation Macro-F1
    axes[1].plot(epochs, base_val_f1, label="Baseline (Best epoch 45: 0.9678)", color="#2b5c8f", lw=2)
    axes[1].plot(epochs, af_val_f1, label="AF-IEDL (Best epoch 23: 0.9720)", color="#1b9e77", lw=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Macro-F1")
    axes[1].set_title("(b) Validation Macro-F1 Convergence")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "learning_curves.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_calibration_and_metrics():
    # Comparing macro-F1, accuracy, cross-entropy loss, and parameter count
    models = ["Baseline", "CBAM", "Focal Loss", "AF-IEDL"]
    macro_f1 = [0.8935, 0.8538, 0.8814, 0.8877]
    ce_loss = [3.068, 1.585, 1.165, 0.8926]
    inj_recall = [0.8219, 0.7671, 0.8904, 0.9452]
    
    x = np.arange(len(models))
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 1. Macro-F1
    axes[0].bar(x, macro_f1, color=["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"], width=0.55)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20)
    axes[0].set_ylabel("Macro-F1 Score")
    axes[0].set_title("Overall Test Macro-F1")
    axes[0].set_ylim(0.8, 0.92)
    for i, v in enumerate(macro_f1):
        axes[0].text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    
    # 2. Test Cross-Entropy (Calibration)
    axes[1].bar(x, ce_loss, color=["#e41a1c", "#ff7f00", "#984ea3", "#4daf4a"], width=0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20)
    axes[1].set_ylabel("Test Cross-Entropy Loss (Lower is Better)")
    axes[1].set_title("Confidence Calibration & Test Cross-Entropy")
    for i, v in enumerate(ce_loss):
        axes[1].text(i, v + 0.07, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    
    # 3. Injection Attack Recall
    axes[2].bar(x, inj_recall, color=["#377eb8", "#ff7f00", "#984ea3", "#2ca02c"], width=0.55)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20)
    axes[2].set_ylabel("Recall Rate")
    axes[2].set_title("High-Risk Injection Attack Recall")
    axes[2].set_ylim(0.7, 1.0)
    for i, v in enumerate(inj_recall):
        axes[2].text(i, v + 0.008, f"{v*100:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    out_path = FIGURES_DIR / "tradeoff_summary.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    plot_confusion_matrices()
    plot_per_class_f1()
    plot_learning_curves()
    plot_calibration_and_metrics()
    print("All paper figures successfully generated!")
