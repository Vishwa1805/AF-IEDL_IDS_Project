# IEDL-IDS Enhanced Experiment Tool (v3)

This package adds a controlled extension to the existing IEDL-IDS v2 course-project baseline. It does not alter the 21x21 RGB image generation or the saved train/validation/test NPZ files.

## Proposed extension: AF-IEDL (Attention-and-Focal IEDL)

Two independently motivated changes are evaluated:

1. **CBAM attention** after each convolution+ReLU stage: sequential channel attention and spatial attention refine convolutional feature maps.
2. **Multi-class focal loss** with gamma=2 and normalized inverse-square-root class weights computed from the training labels only. This reduces the relative emphasis on easy examples and modestly compensates for class imbalance.

The combined model is a proposed course-project extension, not a claim of being the first paper to combine these components.

## Ablation variants

| Variant | Encoder | Loss | Purpose |
|---|---|---|---|
| `cbam` | Baseline CNN + CBAM | Cross-entropy | Isolate attention contribution |
| `focal` | Baseline CNN | Class-balanced focal | Isolate loss contribution |
| `cbam_focal` | Baseline CNN + CBAM | Class-balanced focal | Proposed combined model |

The original v2 run is the no-CBAM, ordinary-cross-entropy reference. For fair comparison, all variants should use the same NPZ split files, seed, number of epochs, batch size, learning rate and validation-macro-F1 checkpoint selection. Do not change or use the test split for tuning.

## Requirements

Use the existing working Python launcher (`py`) and environment:

```powershell
py -m pip install numpy scikit-learn torch
```

## Example commands

Run from the extracted `IEDL_IDS_Enhanced_Tool_v3` directory. Change only the output folder and variant for each experiment.

**Experiment A — CBAM only**

```powershell
py train_iedl_enhanced.py --data-dir "D:\path\to\IEDL_IDS_Project\iedl_images" --out-dir "D:\path\to\IEDL_IDS_Project\experiment_cbam" --variant cbam --epochs 50 --batch-size 64 --num-workers 0 --seed 42
```

**Experiment B — focal loss only**

```powershell
py train_iedl_enhanced.py --data-dir "D:\path\to\IEDL_IDS_Project\iedl_images" --out-dir "D:\path\to\IEDL_IDS_Project\experiment_focal" --variant focal --epochs 50 --batch-size 64 --num-workers 0 --seed 42
```

**Experiment C — CBAM + focal loss (proposed model)**

```powershell
py train_iedl_enhanced.py --data-dir "D:\path\to\IEDL_IDS_Project\iedl_images" --out-dir "D:\path\to\IEDL_IDS_Project\experiment_cbam_focal" --variant cbam_focal --epochs 50 --batch-size 64 --num-workers 0 --seed 42
```

Each run produces `best_model.pt`, `training_history.csv`, `test_metrics.json`, `classification_report.json`, and `confusion_matrix.csv`.

## Notes for reporting

- Report accuracy, macro-F1, weighted-F1, macro-precision, macro-recall, per-class metrics, and parameter count.
- Compare macro-F1 and minority-class recall as well as accuracy.
- Focal-loss objective values are not directly comparable with cross-entropy objective values. The tool therefore also writes `test_cross_entropy` for reference.
- The baseline is selected by validation macro-F1, as are all variants.
- Do not claim improvement until all runs have completed and outputs have been inspected.

## References

Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. *Proceedings of the European Conference on Computer Vision*, 3–19. https://doi.org/10.1007/978-3-030-01234-2_1

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *Proceedings of the IEEE International Conference on Computer Vision*, 2980–2988. https://doi.org/10.1109/ICCV.2017.324

## BibTeX

```bibtex
@inproceedings{woo2018cbam,
  title={CBAM: Convolutional Block Attention Module},
  author={Woo, Sanghyun and Park, Jongchan and Lee, Joon-Young and Kweon, In So},
  booktitle={Proceedings of the European Conference on Computer Vision},
  pages={3--19},
  year={2018},
  doi={10.1007/978-3-030-01234-2_1}
}

@inproceedings{lin2017focal,
  title={Focal Loss for Dense Object Detection},
  author={Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2980--2988},
  year={2017},
  doi={10.1109/ICCV.2017.324}
}
```
