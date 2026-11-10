"""Predict the class of one saved IEDL-IDS traffic-image sample."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model_enhanced import AttentionIEDLIDS


LABEL_NAMES = [
    "backdoor", "ddos", "dos", "injection", "normal",
    "password", "ransomware", "scanning", "xss",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Folder containing images_train/validation/test.npz")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to best_model.pt from an enhanced-model run")
    parser.add_argument("--split", choices=["train", "validation", "test"],
                        default="validation")
    parser.add_argument("--index", type=int, default=0,
                        help="Zero-based image index within the selected split")
    args = parser.parse_args()

    npz_path = Path(args.data_dir) / f"images_{args.split}.npz"
    checkpoint_path = Path(args.checkpoint)
    if not npz_path.is_file():
        parser.error(f"Data split not found: {npz_path}")
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    with np.load(npz_path, allow_pickle=False) as data:
        if not {"X", "y"}.issubset(data.files):
            parser.error(f"{npz_path} must contain arrays X and y")
        images, labels = data["X"], data["y"]

    if images.ndim != 4 or images.shape[1:] != (21, 21, 3):
        parser.error(f"Expected NHWC images with shape (N,21,21,3), got {images.shape}")
    if len(images) != len(labels) or not 0 <= args.index < len(labels):
        parser.error(f"Index must be between 0 and {len(labels) - 1}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    variant = saved_args.get("variant", "cbam_focal")
    use_cbam = variant in ("cbam", "cbam_focal")
    model = AttentionIEDLIDS(
        num_classes=len(LABEL_NAMES),
        latent_dim=saved_args.get("latent_dim", 20),
        hidden_dim=saved_args.get("hidden_dim", 256),
        dropout=saved_args.get("dropout", 0.5),
        use_cbam=use_cbam,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = np.ascontiguousarray(images[args.index].transpose(2, 0, 1), dtype=np.float32) / 255.0
    x = torch.from_numpy(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(x), dim=1)[0]
        scores, ids = torch.topk(probabilities, k=3)

    truth_id = int(labels[args.index])
    predicted_id = int(ids[0])
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Variant: {variant} | device: {device}")
    print(f"Sample: {args.split}[{args.index}] (one 21x21 RGB image representing 63 flows)")
    print(f"Known label: {LABEL_NAMES[truth_id]}")
    print(f"Prediction:  {LABEL_NAMES[predicted_id]} ({float(scores[0]) * 100:.2f}% confidence)")
    print("Top three:")
    for class_id, score in zip(ids.tolist(), scores.tolist()):
        print(f"  {LABEL_NAMES[class_id]:12s} {score * 100:6.2f}%")
    print(f"Demo match:  {predicted_id == truth_id}")


if __name__ == "__main__":
    main()
