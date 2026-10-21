# Model Reproduction Results

This document records a local reproduction run performed on 21 October 2026. The values below are newly generated results, not metrics copied from an earlier report.

## Environment

- Windows
- Python 3.12.15
- TensorFlow 2.19.0
- scikit-learn 1.9.1
- NumPy 2.1.3
- CPU execution
- Matplotlib `Agg` backend

An isolated Conda environment named `gnss-classifier` was used. MATLAB and Octave were not available.

## Summary

| Model | Dataset | Evaluation observations | Accuracy | Macro F1 | Weighted F1 | Status |
|---|---|---:|---:|---:|---:|---|
| MLP | `TagDatabase/TagDatabase.csv` | 212,658 rows | 0.77 | 0.471 | 0.778 | Completed |
| Random Forest, 30-step windows | `TagDatabase/data_b_output.csv` | 68 windows | 0.9265 | 0.9223 | 0.9269 | Completed |
| Random Forest, grouped statistics | `TagDatabase/data_b_output.csv` | 6 satellite/site groups | 0.5000 | 0.4750 | 0.4667 | Completed, sample too small |
| BiLSTM | `TagDatabase/data_b_dataset.csv` | 2,100 time-step labels | 0.8871 | 0.8818 | 0.8869 | Completed |
| Existing SMConvLSTM checkpoint | original `System_ongoing/output/ublox_2_tag.csv` | 15,309 labelled rows | 0.8330 | 0.4548 | 0.8320 | Completed after compatibility fixes |

## MLP

The MLP uses only `CNR` and `Elevation`. It trained on a manually sampled set of 46,700 observations and evaluated on the remaining 212,658 observations. Early stopping ended training after 65 iterations.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| NLOS | 0.72 | 0.77 | 0.74 | 78,436 |
| LOS | 0.87 | 0.80 | 0.84 | 120,704 |
| NLOS_Refl | 0.05 | 0.62 | 0.10 | 357 |
| NLOS_Diff | 0.49 | 0.47 | 0.48 | 12,275 |
| LOS_Diff | 0.14 | 0.34 | 0.20 | 886 |

The overall accuracy is dominated by NLOS and LOS. Performance on the rare reflection and LOS-diffraction classes is poor.

## Random Forest with Time Windows

The program created 339 flattened 30-step windows and used 68 for validation.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Multi_Path | 1.0000 | 0.9375 | 0.9677 | 16 |
| LOS | 0.9412 | 0.8421 | 0.8889 | 19 |
| Reflection | 0.8000 | 0.9231 | 0.8571 | 13 |
| Diffraction | 0.9524 | 1.0000 | 0.9756 | 20 |

The result is strong, but the validation set is small. Windows are randomly split after construction, so overlapping or nearby observations may occur in both training and validation data. It should not be interpreted as site-independent generalization.

## Random Forest with Grouped Statistics

Only 29 site/PRN groups were produced and only six were used for validation. The resulting 50% accuracy is too unstable for a meaningful model comparison. This method needs more independent site/PRN groups or grouped cross-validation.

## BiLSTM

The BiLSTM created 349 sequences of 30 time steps from `data_b_dataset.csv`. Its classification results were:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Multi_Path | 0.9929 | 0.8235 | 0.9003 | 510 |
| LOS | 0.8540 | 0.9059 | 0.8792 | 510 |
| Reflection | 0.8926 | 0.8061 | 0.8471 | 330 |
| Diffraction | 0.8532 | 0.9533 | 0.9005 | 750 |

The script completed training and evaluation but initially failed only at the final save step because the hard-coded `System_ongoing/Bi_LSTM/` directory did not exist. The best checkpoint produced by `ModelCheckpoint` was recovered as `models/BiLSTM/best_gnss_model.h5`. As with the Random Forest, the random split of overlapping sequences can leak nearby temporal information into validation.

## Existing SMConvLSTM Checkpoint

The repository checkpoint loaded successfully with its custom `SMConvLSTM` layer. It was evaluated on the original inference target, `ublox_2_tag.csv`, after adding compatibility handling for a missing `Site` column and numeric GPS time.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 0 | 0.9950 | 0.8933 | 0.9414 | 11,168 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 1,104 |
| 2 | 0.0578 | 0.3160 | 0.0977 | 212 |
| 3 | 0.6571 | 0.9593 | 0.7800 | 2,825 |

The 83.30% accuracy is misleading because class 0 dominates the dataset and class 1 is never correctly recovered. Macro F1 (0.4548) is the more informative summary.

## Components Without Independent Training Runs

- `Base_convLSTM/` and `Time_masked_convLSTM/` contain reusable layer implementations, not complete training entry points.
- `models/BiLSTM/BiLSTM.m` could not be executed because MATLAB/Octave was unavailable and its helper functions are incomplete.
- `Gaussian_mixture_1.py` executed successfully, but it is an analytical probability-density experiment rather than a trained classifier; its original code displays a plot without saving an artifact.
- `predict_gnss_quality.m` remains incomplete because `predict_gnss.py` is missing.

## Reproducibility Warnings

1. The neural-network scripts do not consistently set all random seeds, so a rerun can produce different values.
2. Several scalers are fitted before the train/validation split, which leaks validation distribution information.
3. Overlapping temporal windows are randomly split rather than separated by site or collection session.
4. The four-class `Type` target and four-class `Uncertainty` target have different meanings.
5. The reported numbers are useful for code recovery and sanity checking, but they are not publication-quality benchmark results.
