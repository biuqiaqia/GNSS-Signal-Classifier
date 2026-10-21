# Unified Model Benchmark

This report contains the directly comparable benchmark run on 21 October 2026. Machine-readable results are in [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json), and the executable procedure is in [`models/unified_benchmark.py`](models/unified_benchmark.py).

## Experimental protocol

The benchmark removes the incompatible feature sets, targets, and random splits used by the historical scripts.

- Datasets: full `TagDatabase.csv` (259,358 rows) and balanced `data_b.csv` (9,208 rows).
- Features for every model: `Elevation`, `Azimuth`, and `CNR` only.
- Common classes: `0 NLOS/Multipath`, `1 LOS`, `2 Reflection`, and `3 Diffraction`. Class 4 (`LOS_Diff`) in the large database is merged into class 3.
- Samples: ten-observation sliding windows constructed independently for each PRN.
- Split: earliest 80% of each PRN's windows for training and latest 20% for testing. No random train/test shuffling.
- Evaluation: all four models predict the class of the final observation in exactly the same windows.
- Scaling: `StandardScaler` fitted only to training features.
- Imbalance handling: balanced class weights for neural networks and `balanced_subsample` for Random Forest.
- Seed: 42.

MLP and Random Forest receive the current observation. BiLSTM and SMConvLSTM receive the full ten-observation history. This keeps target cases identical while allowing temporal models to use sequential input.

## Main results

| Dataset | Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| Full `TagDatabase` | Random Forest | **0.7526** | 0.4202 | **0.4310** | **0.4171** | **0.7607** |
| Full `TagDatabase` | MLP | 0.7052 | 0.4098 | 0.4126 | 0.3985 | 0.7300 |
| Full `TagDatabase` | BiLSTM | 0.7290 | **0.4274** | 0.4186 | 0.4064 | 0.7537 |
| Full `TagDatabase` | SMConvLSTM | 0.6844 | 0.4058 | 0.3463 | 0.3650 | 0.7160 |
| Balanced `data_b` | Random Forest | **0.9742** | **0.9679** | **0.9802** | **0.9733** | **0.9744** |
| Balanced `data_b` | MLP | 0.9044 | 0.9115 | 0.9013 | 0.9035 | 0.9045 |
| Balanced `data_b` | BiLSTM | 0.9198 | 0.9280 | 0.9156 | 0.9177 | 0.9203 |
| Balanced `data_b` | SMConvLSTM | 0.6225 | 0.6512 | 0.6941 | 0.6098 | 0.5520 |

## Per-class F1

| Dataset | Model | NLOS/Multipath | LOS | Reflection | Diffraction |
|---|---|---:|---:|---:|---:|
| Full `TagDatabase` | Random Forest | 0.7915 | 0.7557 | 0.0000 | 0.1212 |
| Full `TagDatabase` | MLP | 0.7742 | 0.7092 | 0.0000 | 0.1104 |
| Full `TagDatabase` | BiLSTM | 0.8029 | 0.7293 | 0.0000 | 0.0932 |
| Full `TagDatabase` | SMConvLSTM | 0.7610 | 0.6989 | 0.0000 | 0.0000 |
| Balanced `data_b` | Random Forest | 0.9563 | 0.9736 | 0.9860 | 0.9772 |
| Balanced `data_b` | MLP | 0.9986 | 0.9193 | 0.8721 | 0.8239 |
| Balanced `data_b` | BiLSTM | 0.9845 | 0.9466 | 0.8829 | 0.8567 |
| Balanced `data_b` | SMConvLSTM | 0.6548 | 0.1714 | 0.7906 | 0.8223 |

## Interpretation

Random Forest is strongest under this protocol. On `data_b`, all four classes are represented well and RF reaches 0.9733 macro F1. BiLSTM ranks second and improves on MLP, indicating that recent PRN history contains useful information, although it does not exceed RF.

Accuracy on the full database is not sufficient evidence of good four-class performance. Its row distribution is 98,436 class 0, 140,704 class 1, 1,557 class 2, and 18,661 combined class 3/4. None of the models identifies reflection in the chronological test period, so macro F1 remains near 0.40 even when accuracy exceeds 0.70. The contrast with `data_b` shows why the balanced set is more suitable for comparing minority-class recognition.

SMConvLSTM underperforms here, especially on `data_b`. For a strict common-target comparison, windows are built separately for each PRN; consequently, its spatial dimensions are `1 × 1` and the spatial mask contains valid observations only. This run tests the repository's masked recurrent layer under unified inputs, but it cannot exploit simultaneous relationships between satellites. A future spatial benchmark should build time × PRN grids and compare all models on the same valid grid cells.

## Reproduction

```bash
python models/unified_benchmark.py
```

Tested environment: Python 3.12.15, TensorFlow 2.19.0, scikit-learn 1.9.1, and NumPy 2.1.3. Neural training used early stopping with up to 15 epochs. Exact values may vary on another TensorFlow backend despite the fixed seed.
