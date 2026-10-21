# GNSS Signal Classifier

Research code for classifying GNSS signal propagation conditions in urban environments. The project combines RINEX observations, broadcast ephemerides, reference trajectories, and 3D building models to generate per-satellite features and ray-tracing labels. MLP, Random Forest, BiLSTM, and spatially masked ConvLSTM (SMConvLSTM) models are compared under a unified benchmark.

> This is a research archive. Some historical scripts contain machine-specific paths and experimental implementations. Raw RINEX data, personal trajectories, and most generated training CSV files are not included.

The latest local execution results and reproducibility limitations are documented in [`REPRODUCTION_RESULTS.md`](REPRODUCTION_RESULTS.md). Exact machine-readable metrics are stored in [`BENCHMARK_RESULTS.json`](BENCHMARK_RESULTS.json).

## Table of Contents

1. [Research Objectives](#1-research-objectives)
2. [Signal Classes](#2-signal-classes)
3. [Dataset Schema](#3-dataset-schema)
4. [`DataProcess/`](#4-dataprocess)
5. [`DataTag/`](#5-datatag)
6. [Data, Baselines, and Visualization](#6-data-baselines-and-visualization)
7. [`models/`](#7-models)
8. [Recommended Reading Order](#8-recommended-reading-order)
9. [Dependencies](#9-dependencies)
10. [Known Issues](#10-known-issues)
11. [Third-party Code](#11-third-party-code)

## 1. Research Objectives

Buildings in urban canyons block or alter satellite signals, producing LOS, reflected, diffracted, multipath, or fully NLOS observations. This project was developed to:

1. generate epoch-level, satellite-level features from raw GNSS data;
2. derive propagation labels using building KML files, sky masks, and ray tracing;
3. train conventional and temporal classifiers; and
4. provide signal-quality predictions to weighted GNSS positioning and GNSS/INS integration systems.



## 2. Signal Classes

The earlier labelling code uses five classes:

| Value | Class | Meaning |
|---:|---|---|
| 0 | NLOS | Direct path is blocked; no valid reflection or diffraction path is confirmed |
| 1 | LOS | Direct line-of-sight signal |
| 2 | NLOS_Refl | NLOS signal arriving through reflection |
| 3 | NLOS_Diff | NLOS signal arriving through diffraction |
| 4 | LOS_Diff | Direct component accompanied by diffraction effects |

The unified benchmark uses four outputs. It retains classes 0–2 and merges class 4 (`LOS_Diff`) into class 3 (`Diffraction`). The name `NLOS/Multipath` is used for class 0 because the large archive calls it NLOS while `data_b` calls it multipath.

## 3. Dataset Schema

`TagDatabase/TagDatabase.csv` contains:

| Field | Description |
|---|---|
| `GPS_Time` | GPS time-of-week, or the experiment's GPS time convention |
| `PRN` | Satellite identifier |
| `Elevation`, `Azimuth` | Satellite elevation and azimuth |
| `CNR` | Carrier-to-noise ratio / signal strength |
| `Pr_Residual` | Pseudorange residual |
| `LS_Latitude`, `LS_Longitude`, `LS_Height` | Least-squares position |
| `GT_Latitude`, `GT_Longitude`, `GT_Height` | Ground-truth/reference position |
| `Doppler` | Doppler observation |
| `SaVX`, `SaVY`, `SaVZ` | Satellite ECEF velocity |
| `lambda` | Signal wavelength |
| `corrected_Pr` | Pseudorange after clock, ionospheric, tropospheric, and related corrections |
| `SaX`, `SaY`, `SaZ` | Satellite ECEF position |
| `Max_PRN`, `Max_Residual` | Reference satellite and residual used in differencing-related processing |
| `Visibility` | Sky-mask result, normally `Visible` or `Blocked` |
| `ReflDist`, `DiffDist` | Extra reflected and diffracted path lengths from ray tracing |
| `D_Coeff` | Diffraction coefficient |
| `Type` | Ray-tracing signal-type label |

The balanced dataset additionally uses `Site`, `CNR_rate`, `Diff_DelPr_PrRate`, and `SecDeri_pr`. These extra fields are retained as research data but are excluded from the unified comparison.


## 4. `DataProcess/`

### `DataProcess/RINEX_processing_main_f.m`

Main raw-GNSS processor. It reads RINEX observations and broadcast navigation, calculates satellite position/velocity/clock corrections, produces elevation, azimuth, C/N0, Doppler, and corrected pseudorange, performs least-squares positioning, aligns results with a reference trajectory, calculates residuals, and exports per-satellite CSV data. Embedded input paths and experimental coordinates must be changed for new data.

### `DataProcess/goGPS/`

Third-party goGPS MATLAB library. `input_output/` handles RINEX, RTCM, UBX, NMEA, and SP3; `positioning/` contains satellite, correction, least-squares, and Kalman routines; `utility/` contains coordinate/time/GNSS helpers; `obj/` contains classes; `settings/` contains examples; and `plot/` contains visualization functions.

### `DataProcess/gtsam_toolbox-master/`

Third-party GTSAM MATLAB toolbox for factor graphs, navigation, and optimization. It is retained as part of the historical environment; current classifier scripts do not directly use it as their model framework.

### `DataProcess/fsaxen-ParforProgMon-f99c6f0/`

Third-party MATLAB parallel-loop progress utility. `ParforProgressbar.m` is the `parfor` progress class, `progressbar.m` implements the bar, `testParforProgressbar.m` is an example, and `Readme.md`, `license.txt`, and `progress.png` are upstream documentation assets.

## 5. `DataTag/`

### `DataTag/zzy_skymask_generation_f.m`

Main label generator. It reads the per-satellite CSV and building KML, creates a sky mask, runs reflection/diffraction ray tracing, and appends `Visibility`, `ReflDist`, `DiffDist`, `D_Coeff`, and `Type`. Output is normally `<experiment>_tag.csv`, with optional rejected-record output.

### `DataTag/Attachment/`

Historical support code for building-KML parsing, sky-mask creation/merging, LOS/NLOS decisions, reflection and diffraction ray tracing, UTD diffraction coefficients, GNSS RUMS examples, and RINEX/positioning helpers. It also contains copies of goGPS, GTSAM, and satnav3p01. These preserve the original environment but should eventually be deduplicated and documented as dependencies.

## 6. Data, Baselines, and Visualization

### `TagDatabase/TagDatabase.csv`

The original consolidated labelled database and the largest dataset in this repository:

- **259,358 observations, 28 columns, approximately 60.17 MB**;
- covers multiple experiments over a broad GPS-time range;
- contains geometry, C/N0, pseudorange residuals, least-squares and ground-truth positions, satellite state, ray-tracing results, and `Type`;
- uses the original five-class `Type` definition (`0`–`4`); and
- is the dataset explicitly referenced by the early `MLP/MLP_1.py` baseline.

Its class distribution is highly imbalanced: 98,436 class-0, 140,704 class-1, 1,557 class-2, 16,275 class-3, and 2,386 class-4 observations. It should therefore not be treated as a balanced training set without an explicit sampling or weighting strategy.

### `TagDatabase/data_b.csv`

The single canonical balanced four-class dataset used by the unified benchmark:

- **9,208 observations and 30 columns**;
- adds `CNR_rate`, `Diff_DelPr_PrRate`, `SecDeri_pr`, and `Site` for later research;
- uses weighted least-squares position fields (`WLS_*`) instead of the original `LS_*` fields;
- maps `Type` to `0: Multi_Path`, `1: LOS`, `2: Reflection`, and `3: Diffraction`; and
- has class counts of 2,052, 2,477, 2,076, and 2,603 respectively.

The extra engineered features are not used in the current comparison. Every model receives only `Elevation`, `Azimuth`, and `CNR`, ensuring that differences in results reflect the models rather than different input columns.

### Dataset lineage and model association

```text
TagDatabase.csv -> full five-class archive -> mapped to four classes -> all four models
data_b.csv      -> balanced four-class set -> all four models
```

These datasets may reveal experimental locations and trajectories. They should be anonymized or removed before the repository is made public.

### Current unified results

All values below come from the same features (`Elevation`, `Azimuth`, `CNR`), ten-observation PRN windows, four-class mapping, and chronological 80/20 split.

| Dataset | Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| Full `TagDatabase` | Random Forest | **0.7526** | **0.4171** |
| Full `TagDatabase` | MLP | 0.7052 | 0.3985 |
| Full `TagDatabase` | BiLSTM | 0.7290 | 0.4064 |
| Full `TagDatabase` | SMConvLSTM | 0.6844 | 0.3650 |
| Balanced `data_b` | Random Forest | **0.9742** | **0.9733** |
| Balanced `data_b` | MLP | 0.9044 | 0.9035 |
| Balanced `data_b` | BiLSTM | 0.9198 | 0.9177 |
| Balanced `data_b` | SMConvLSTM | 0.6225 | 0.6098 |

The full database's apparently reasonable accuracy is dominated by classes 0 and 1; all four models obtain zero F1 for the rare reflection class in the chronological test set. See [`REPRODUCTION_RESULTS.md`](REPRODUCTION_RESULTS.md) for per-class scores and limitations.

### `MLP/MLP_1.py`

Early multilayer-perceptron baseline using mainly `CNR` and `Elevation` to predict `Type`. It includes standardization, balancing, training, validation, and decision-boundary visualization. Its CSV path is a placeholder.

### `Visualized_Path/to kml.f.py`

Reads ECEF/positioning results from CSV, converts them to latitude, longitude, and height, and creates a Google Earth KML trajectory. A clearer future name is `export_trajectory_to_kml.py`.

## 7. `models/`

### Random Forest

- `models/RF/RF.py`: historical time-window experiment. It is retained for reference, but its original four-feature setup is not used for the current cross-model results.

### BiLSTM

- `models/BiLSTM/biLSTM.2.py`: the most complete Python training entry point. It creates overlapping sequences, standardizes inputs, builds bidirectional/standard LSTM and dropout layers, uses focal loss, reports metrics and permutation importance, plots training results, and saves the Keras model. Paths are hard-coded; `train_bilstm.py` would be a clearer name.
- `models/BiLSTM/best_gnss_model.h5`: checkpoint from an earlier, non-unified reproduction run. It is not the source of the current comparison metrics.
- `models/BiLSTM/BiLSTM.m`: MATLAB two-layer BiLSTM prototype with class weights and five-class output. It saves `trained_gnss_model.mat` but depends on missing helpers, and its focal-loss comments do not fully match the configuration.

### Base ConvLSTM

- `models/Base_convLSTM/base_conv_lstm.py`: low-level `ConvLSTMCell` and `ConvLSTM` implementation managing convolutional recurrent state, shapes, dropout, and initialization; it is a layer, not a training entry point.
- `models/Base_convLSTM/conv_lstm2d.py`: two-dimensional `ConvLSTM2D` wrapper providing a Keras-like interface for spatial sequences.

### Time-masked ConvLSTM

- `models/Time_masked_convLSTM/self-made masked_conv_LSTM.py`: early custom implementation that propagates a mask so zero padding for missing satellites/time steps is not interpreted as an observation.
- `models/Time_masked_convLSTM/masked_convLSTM2D_module.py`: another implementation defining `ConvLSTMCell`, `ConvLSTM`, and `MaskedConvLSTM2D`; it differs from the root-level version.
- `models/masked_convLSTM2D_module.py`: later copy placed at the model root for direct imports. Experiment records are needed to determine which version is authoritative.

### SMConvLSTM

- `models/Rho_module.py`: core custom `SMConvLSTMCell` and `SMConvLSTM` layers, jointly handling time, satellite/spatial dimensions, and observation masks.
- `models/SMconLSTM_training.py`: reshapes CSV data into `sample × time × PRN × feature`, masks unavailable satellites, and trains a four-class model with `Elevation`, `Azimuth`, and `CNR`. It saves an HDF5 checkpoint but uses `Uncertainty` as its target and contains hard-coded paths.
- `models/load_model_SMconvLSTM.py`: loads the HDF5 model and custom layers, predicts CSV records, writes `Predicted_Type`/`Predicted_Label`, and reports metrics when ground truth exists.
- `models/gnss_masked_convlstm_model.h5`: trained model (about 169 KB). Training-data version, class mapping, TensorFlow version, validation performance, and random seed are not documented.
- `models/load_model.py`: generic evaluator for an earlier temporal model using `Elevation`, `CNR`, `CNR_rate`, and `Diff_DelPr_PrRate`. The filename should eventually identify its architecture.

### Unified benchmark

- `models/unified_benchmark.py`: authoritative comparison entry point. It loads both datasets, maps them to the same four classes, uses only `Elevation`, `Azimuth`, and `CNR`, creates ten-observation windows per PRN, applies the same chronological split, trains MLP, Random Forest, BiLSTM, and the repository's `SMConvLSTM`, and writes exact metrics to `BENCHMARK_RESULTS.json`.
- `BENCHMARK_RESULTS.json`: machine-readable configuration, sample counts, aggregate metrics, and per-class F1 scores from the latest run.
- `REPRODUCTION_RESULTS.md`: formatted result tables, interpretation, limitations, and reproduction instructions.

## 8. Recommended Reading Order

1. `DataProcess/RINEX_processing_main_f.m` — feature generation.
2. `DataTag/zzy_skymask_generation_f.m` — label generation.
3. `TagDatabase/TagDatabase.csv` — schema and class distribution.
4. `models/unified_benchmark.py` — reproducible four-model comparison.
5. `REPRODUCTION_RESULTS.md` — current results and their interpretation.
6. `MLP/`, `models/RF/`, and `models/BiLSTM/` — historical standalone experiments.
7. `models/SMconLSTM_training.py` and `models/Rho_module.py` — original satellite-masked implementation.

## 9. Dependencies

MATLAB dependencies include MATLAB, Mapping Toolbox, Statistics and Machine Learning Toolbox, and the bundled goGPS, GTSAM, satnav3p01, and ray-tracing code.

Python dependencies include `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `tensorflow`, and `keras`. Exact versions have not been recovered; a future reproducibility pass should add `requirements.txt` or `environment.yml`.

## 10. Known Issues

- Absolute `C:/Users/ZHANG/...` paths remain in several scripts.
- Four-class and five-class implementations coexist.
- `pred_Uncertainty` is not a real probability/confidence estimate.
- The unified benchmark has a fixed configuration and random seed, but there is still no dependency lock file or automated test suite.
- The unified SMConvLSTM comparison uses per-PRN `1 × 1` spatial windows for identical targets; it therefore does not measure the possible benefit of cross-satellite spatial context.

## 11. Third-party Code

goGPS, GTSAM, ParforProgMon, satnav3p01, and other supporting files originate from their respective upstream projects or research codebases. Use and redistribution must comply with the licences and citation requirements in their directories. Inclusion here does not constitute relicensing.
