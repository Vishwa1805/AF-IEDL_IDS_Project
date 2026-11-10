# IEDL-IDS CNN + MLP baseline (v2)

This update keeps the smoke-tested data loader, training loop, metrics and outputs, but revises the classifier to better follow the MLP description in Section 4.2.1 of the assigned paper.

## Architecture
- Input: RGB 21x21x3 (converted to NCHW float32 in [0,1]).
- Encoder: Conv2d(3,64,3,padding=1) -> ReLU -> 2x2 MaxPool -> Conv2d(64,32,3,padding=1) -> ReLU -> 2x2 MaxPool.
- Flatten size: 800.
- Latent projection: 800 -> 20 (20 < 21 as stated in the paper).
- MLP: 20 -> 256 -> Dropout(0.5) -> 256 -> 9 class logits.

The paper does not specify the exact latent width or fully enumerate the downstream MLP's dimensions. Latent width 20 and the precise placement of the two 256-unit dense layers are documented reconstruction choices. This implementation is paper-guided, not the authors' released source code.

## Training defaults
Adam, learning rate 0.001, batch size 64, 50 epochs, dropout 0.5. CrossEntropyLoss is used with raw logits.

## Install / verify
```powershell
py -m pip install -r requirements.txt
py -c "import numpy, sklearn, torch; print(numpy.__version__, sklearn.__version__, torch.__version__)"
```

## Run
```powershell
py train_iedl.py --data-dir "D:\path\to\IEDL_IDS_Project\iedl_images" --out-dir "D:\path\to\IEDL_IDS_Project\baseline_results_v2" --epochs 50 --batch-size 64 --num-workers 0
```

Use a new output directory so the earlier 1-epoch smoke-test checkpoint is preserved separately.
