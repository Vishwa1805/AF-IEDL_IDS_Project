# AF-IEDL: Attention-Augmented and Focal-Calibrated Deep Learning for Robust Network Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Capstone](https://img.shields.io/badge/Coursework-CNS%20Capstone%20Project-success.svg)]()

> **Authors:** **Vishwa S** (CS23I1043) and **Praveen Chouthri** (CS23I1044)  
> **Affiliation:** Department of Computer Science and Engineering  
> **Coursework:** Computer and Network Security (CNS), Academic Semester VII  
> **Reference Base Paper:** *IEDL-IDS: An Image-Enhanced Encoder-Based Deep Learning Scheme for Intrusion Detection Systems* (ACM TOPS, Feb 2026)

---

## 📌 Executive Summary

This repository presents the official reproduction audit, architectural critique, and implementation of **AF-IEDL** (Attention-Augmented and Focal-Calibrated Image-Enhanced Deep Learning), an advanced deep learning framework for Network Intrusion Detection Systems (NIDS).

Recent visual NIDS literature—exemplified by IEDL-IDS (Wang et al., ACM TOPS 2026)—advocates reshaping tabular network flows into two-dimensional visual representations ($21 \times 21 \times 3$ RGB image tensors) to exploit computer vision architectures (CNNs, CBAM). Through a full-scale autonomous audit of all **23 CSV shards (22,339,021 flows)** of the ToN_IoT benchmark, our investigation reveals that published near-perfect metrics rely on significant methodological vulnerabilities. 

To overcome these limitations, **AF-IEDL** integrates:
1. **Dual-Stage Convolutional Block Attention Modules (CBAM)** to adaptively reweight non-spatial tabular coordinates and optical flow channels;
2. **Class-Balanced Multi-Class Focal Loss ($\gamma = 2.0$)** to suppress gradient dominance of majority classes and resolve severe test miscalibration under temporal drift;
3. **Continuous Streaming Window (CSW) Inference Engine** capable of label-independent sliding-window evaluation at line-rate forward speeds ($P_{50} = 1.388\text{ ms}$).

---

## 🔬 Core Innovations & Comparison with Base Paper

| Design Dimension | Base Paper: IEDL-IDS (Wang et al., ACM TOPS 2026) | Proposed Framework: AF-IEDL (Ours) |
| :--- | :--- | :--- |
| **Image Generation** | Segregates flows by attack class before grouping; drops minority classes that cannot fill blocks (e.g., MITM). | Strict chronological class-wise grouping for offline training; supplemented by Continuous Streaming Window (CSW) for live traffic. |
| **Tabular Coordinates** | Assumes 2D spatial locality across arbitrary CSV column headers; slices flows into RGB channels without justification. | Introduces dual-stage **CBAM** (Channel + Spatial Attention) to dynamically reweight non-spatial coordinates and channel dependencies. |
| **Feature Filtering** | Claims $>80\%$ zeros left 21 features; conflates Zeek hyphens (`'-'`) with zeros; omits feature names. | Audits 22.3M raw rows; formalizes canonical 21-feature schema with median imputation and 3-state boolean encoding. |
| **Optimization Objective** | Standard categorical cross-entropy; susceptible to majority-class gradient dominance and miscalibration. | **Class-Balanced Multi-Class Focal Loss** ($\gamma = 2.0$); down-weights easy majority instances and boosts minority gradients. |
| **Confidence Calibration** | Severe test overconfidence under temporal distribution drift (Test cross-entropy $= 3.068$; $97.8\%$ confidence on incorrect predictions). | **70.9% reduction in test cross-entropy** ($0.8926$); Expected Calibration Error (ECE) suppressed to **$0.0854$**. |
| **False Positive Rate** | High false alarm rate on benign traffic ($\text{FPR} = 7.87\%$), inducing SOC alert fatigue. | **70.0% reduction in FPR** ($\text{FPR} = 2.36\%$, benign specificity $97.64\%$), saving security analyst bandwidth. |
| **Critical Threat Recall** | Injection recall suppressed to $82.19\%$ due to minority class penalties. | **Injection recall surges to $94.52\%$** ($69/73$ test instances detected), preventing high-risk arbitrary code execution. |
| **Online Streaming** | No streaming evaluation; requires an oracle to group 63 flows of the identical attack type at line rate. | Implements real-flow CSW circular-buffer engine; profiles forward latency ($P_{50} = 1.388\text{ ms}$, $P_{99} = 2.723\text{ ms}$). |

---

## 📊 Comprehensive Experimental Results

All experiments were conducted on an audited 10% stratified sample ($2,233,902$ raw flows) split strictly chronologically by epoch timestamp (80% Train, 10% Validation, 10% Held-out Test) across 9 target classes:

### 1. Overall Test Set Performance & Architectural Complexity
| Model Architecture | Accuracy | Macro-Prec | Macro-Rec | Macro-F1 | Weighted-F1 | Test Cross-Entropy | Parameters | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Cross-Entropy)** | **90.86%** | **87.53%** | 92.52% | **0.8935** | **0.9075** | 3.0680 | 109,757 | 18.3 min |
| **CBAM Only** | 90.63% | 83.11% | 89.23% | 0.8538 | 0.9057 | 1.5850 | 111,233 | 56.3 min |
| **Focal Loss Only** | 90.43% | 86.47% | 91.43% | 0.8814 | 0.9036 | 1.1650 | 109,757 | 14.7 min |
| **AF-IEDL (CBAM + Focal)** | 90.52% | 87.25% | **92.53%** | 0.8877 | 0.9051 | **0.8926** | 111,233 | 24.3 min |

### 2. Operational Security & Calibration Metrics
| Operational Metric | Baseline IEDL-IDS | AF-IEDL (Ours) | Relative Impact |
| :--- | :---: | :---: | :---: |
| **False Positive Rate (FPR) $\downarrow$** | 7.87% (10/127 errors) | **2.36% (3/127 errors)** | **70.0% reduction in false alarms** |
| **Benign Specificity (TNR) $\uparrow$** | 92.13% | **97.64%** | **+5.51% absolute gain** |
| **Test Cross-Entropy Loss $\downarrow$** | 3.0680 | **0.8926** | **70.9% reduction in prediction error** |
| **Expected Calibration Error (ECE) $\downarrow$** | 0.0882 | **0.0854** | Well-calibrated Bayesian confidence |
| **Injection Attack Recall $\uparrow$** | 82.19% | **94.52%** | **+12.33% absolute gain ($69/73$ detected)** |
| **Ransomware Recall $\uparrow$** | 91.67% | **100.00%** | Zero false negatives on ransomware test split |
| **Forward Pass Latency ($P_{50}$)** | — | **1.388 ms** | Sub-2ms real-time edge viability |

---

## 📁 Repository Directory Structure

```text
IEDL_IDS_Project/
|
├── tools/                                # Modular Engineering & Reproduction Pipeline
│   ├── audit_network_features_v2.py      # Memory-efficient streaming auditor for 22.3M rows
│   ├── audit_network_shards.py           # Shard-level verification script
│   ├── compute_advanced_metrics.py       # ECE, Brier score, latency, and operational FPR calculator
│   ├── generate_paper_assets.py          # Automated figure generator using Seaborn & Matplotlib
│   ├── package_overleaf.py               # Generates one-click Overleaf distribution zip
│   ├── streaming_ids_inference.py        # CSW sliding-window real-time inference engine
│   │
│   ├── IEDL_IDS_Sampling_Tool/           # 10% class-stratified sampling pipeline
│   │   └── prepare_sample.py
│   ├── IEDL_IDS_Preprocessing_Tool/      # 21-feature extraction and 3-channel RGB image tensor builder
│   │   └── prepare_iedl_images.py
│   ├── IEDL_IDS_Image_QA_Tool/           # Verification script for .npz tensor integrity
│   │   └── verify_image_arrays.py
│   ├── IEDL_IDS_CNN_Tool/                # Original baseline implementation
│   │   ├── model.py
│   │   └── train_iedl.py
│   ├── IEDL_IDS_CNN_Tool_v2/             # Audited chronological baseline with drift diagnostics
│   │   ├── model.py
│   │   ├── train_iedl.py
│   │   └── diagnose_raw_split_drift.py
│   └── IEDL_IDS_Enhanced_Tool_v3/        # Proposed AF-IEDL Architecture
│       ├── model_enhanced.py             # CNN + CBAM (Channel & Spatial Attention)
│       ├── train_iedl_enhanced.py        # Focal Loss training loop with Adam optimizer
│       └── predict_demo.py               # Single-sample inference test
│
├── audit_output_v2/                      # Master 22.3M Flow Audit Manifests
│   ├── audit_summary_v2.json             # Global column profiles, zeros, hyphens
│   ├── class_counts_v2.csv               # Ground-truth flow counts per class
│   ├── feature_presence_audit_v2.csv     # Missingness and sparsity statistics
│   └── shard_counts_v2.csv               # Row count validation across 23 CSV shards
│
├── iedl_images/                          # Preprocessed Image Tensors (.npz)
│   ├── feature_selection.csv             # 21 canonical selected features
│   ├── images_train.npz                  # 28,358 training RGB images (21x21x3)
│   ├── images_validation.npz             # 3,540 validation RGB images
│   └── images_test.npz                   # 3,554 held-out test RGB images
│
├── sampled_data/                         # Stratified Sample Metadata
│   ├── sampling_manifest.json            # Sampling configuration and random seed state
│   ├── sampled_class_counts.csv          # Exact counts in the 2.23M flow sample
│   └── sampled_shard_counts.csv          # Distribution across shards
│
├── baseline_iedl_v2_50ep/                # Baseline Model Checkpoint & Logs (50 Epochs)
│   ├── best_model.pt                     # Best validation checkpoint (0.42 MB)
│   ├── classification_report.json
│   └── confusion_matrix.csv
│
├── experiment_cbam/                      # CBAM Only Ablation Results
│   ├── best_model.pt                     # Checkpoint (0.43 MB)
│   ├── classification_report.json
│   └── confusion_matrix.csv
│
├── experiment_focal/                     # Focal Loss Only Ablation Results
│   ├── best_model.pt                     # Checkpoint (0.42 MB)
│   ├── classification_report.json
│   └── confusion_matrix.csv
│
├── experiment_cbam_focal/                # Proposed AF-IEDL Checkpoint & Results
│   ├── best_model.pt                     # Final AF-IEDL weights (0.43 MB)
│   ├── classification_report.json
│   └── confusion_matrix.csv
│
├── .gitignore                            # Excludes large raw CSVs (>100MB) & cache files
├── requirements.txt                      # Pinned Python package dependencies
└── README.md                             # Academic project documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
Clone this repository and install the verified dependencies using Python 3.10 or 3.11:
```bash
git clone https://github.com/Vishwa1805/AF-IEDL_IDS_Project
cd AF-IEDL_IDS_Project
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Verify Tensor Preprocessing & QA
To verify that the preprocessed image tensors match the canonical $21 \times 21 \times 3$ specification:
```bash
python tools/IEDL_IDS_Image_QA_Tool/verify_image_arrays.py
```

### 3. Train the Proposed AF-IEDL Model
To train the CNN backbone with dual-stage CBAM attention and Class-Balanced Focal Loss ($\gamma = 2.0$):
```bash
python tools/IEDL_IDS_Enhanced_Tool_v3/train_iedl_enhanced.py \
  --images_dir iedl_images \
  --output_dir experiment_cbam_focal \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001 \
  --gamma 2.0 \
  --use_cbam
```

### 4. Compute Advanced Calibration & Operational Metrics
Evaluate Expected Calibration Error (ECE), Brier Score, and line-rate forward latency across all 4 ablations:
```bash
python tools/compute_advanced_metrics.py
```

### 5. Run Continuous Streaming Window (CSW) Inference
Simulate an online network gateway evaluating live incoming flows with a sliding circular FIFO buffer ($W=63$, Stride $S=21$) with strict label independence:
```bash
python tools/streaming_ids_inference.py
```

### 6. Re-generate Paper Figures
To regenerate the publication-grade figures in `Paper/latex/figures/`:
```bash
python tools/generate_paper_assets.py
```

---

## 📦 Dataset & File Management Policy for GitHub

* **Repository Size Control**: GitHub enforces a strict **100 MB per-file limit**. The raw 10% sampled traffic file (`sampled_data/sampled_raw_10pct.csv`, 319.8 MB) is appropriately excluded in `.gitignore`.
* **Lightweight Weights Included**: All four trained PyTorch model checkpoints (`best_model.pt`) are compact (**$\approx 0.43\text{ MB}$ each**) and included in the repository, enabling immediate offline reproduction without requiring hours of re-training.
* **Preprocessed Tensors Included**: The converted image tensors in `iedl_images/` total only **$\approx 5.6\text{ MB}$**, ensuring anyone cloning this repository can directly run training and evaluation scripts out-of-the-box.
* **Full Dataset Source**: The complete 22.3M flow raw ToN_IoT dataset is publicly hosted by the Australian Centre for Cyber Security (ACCS) / UNSW Canberra on [IEEE DataPort](https://ieee-dataport.org/documents/toniot-datasets).

---

## 📖 Citation

If you find this work, audit, or codebase useful for your research, please cite our study:

```bibtex
@article{vishwa2026afiedl,
  title={AF-IEDL: Attention-Augmented and Focal-Calibrated Deep Learning for Robust Network Intrusion Detection},
  author={VISHWA S and PRAVEEN CHOUTHRI},
  journal={Academic Capstone Project, Department of Computer Science and Engineering},
  year={2026},
  institution={INDIAN INSTITUTE OF INFORMATION TECHNOLOGY, DESIGN AND MANUFACTURING, KANCHEEPURAM}
}
```

---
*Developed as part of the 7th-Semester Capstone Project for Computer and Network Security (CNS).*
