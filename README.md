# GNSS Signal Classifier

Research code for classifying GNSS signal propagation conditions in urban environments. The project combines RINEX observations, broadcast ephemerides, reference trajectories, and 3D building models to generate per-satellite features and ray-tracing labels. MLP, Random Forest, BiLSTM, ConvLSTM, and spatially masked ConvLSTM (SMConvLSTM) models are then used for classification.

> This is a research archive. Some scripts contain machine-specific paths, experimental implementations, and inconsistent class definitions. Raw UBX/RINEX data, personal trajectories, and most training CSV files are not included.

## Table of Contents

1. [Research Objectives](#1-research-objectives)
2. [Signal Classes](#2-signal-classes)
3. [Dataset Schema](#3-dataset-schema)
4. [`DataProcess/`](#5-dataprocess)
5. [`DataTag/`](#6-datatag)
6. [Data, Baselines, and Visualization](#7-data-baselines-and-visualization)
7. [`models/`](#8-models)
8. [Recommended Reading Order](#9-recommended-reading-order)
9. [Dependencies](#10-dependencies)
10. [Known Issues](#11-known-issues)
11. [Third-party Code](#13-third-party-code)

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

Some later Python models use four outputs, and some scripts use `Uncertainty` rather than `Type` as the target. Verify each input CSV's class mapping before reproducing an experiment.

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

The later datasets additionally use `Site`, `CNR_rate`, `Diff_DelPr_PrRate`, `SecDeri_pr`, `Uncertainty`, and receiver-velocity fields. The exact differences and intended model associations are documented below.


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

### `TagDatabase/data_b_output.csv`

The four-class dataset read by both Random Forest scripts through their original path `System_ongoing/output/data_b.csv`:

- **9,208 observations, 30 columns, approximately 2.26 MB**;
- adds `CNR_rate`, `Diff_DelPr_PrRate`, `SecDeri_pr`, and `Site` for temporal and site-aware modelling;
- uses weighted least-squares position fields (`WLS_*`) instead of the original `LS_*` fields;
- removes `Max_PRN` and `Max_Residual`; and
- maps `Type` to `0: Multi_Path`, `1: LOS`, `2: Reflection`, and `3: Diffraction`.

The class counts are 2,052, 2,477, 2,076, and 2,603 respectively. Compared with `TagDatabase.csv`, this is a much smaller and more balanced model-ready dataset.

### `TagDatabase/data_b_dataset.csv`

The version read by `models/BiLSTM/biLSTM.2.py` through its original path `System_ongoing/Dataset/data_b.csv`. It has the same 9,208 observations, sites, timestamps, and feature schema as `data_b_output.csv`, but the files are **not identical**:

- classes 0 and 1 are unchanged;
- class 2 contains 1,749 rather than 2,076 observations; and
- class 3 contains 2,930 rather than 2,603 observations.

This means 327 labels were reassigned from class 2 to class 3 in the BiLSTM copy. Both versions are retained to preserve reproducibility; they must not be silently substituted for one another. The repository does not yet contain documentation explaining why those labels were changed.

### `TagDatabase/ublox_3_tag.csv`

The later continuous u-blox experiment used directly by `models/SMconLSTM_training.py`:

- **20,076 observations, 34 columns, approximately 7.66 MB**;
- covers GPS times 133292–136799, suggesting a single continuous collection rather than a multi-experiment database;
- adds receiver velocity (`VRx`, `VRy`, `VRz`), temporal derivative features, `Uncertainty`, and `los`; and
- is reshaped by the trainer into a `sample × time × PRN × feature` tensor with an availability mask.

Although the file retains `Type`, SMConvLSTM is trained against `Uncertainty`. The observed relationship is:

| `Type` | `Uncertainty` | Observations |
|---:|---:|---:|
| 0 | 2 | 1,375 |
| 1 | 0 | 12,780 |
| 2 | 3 | 4,617 |
| 3 | 0 | 538 |
| 3 | 1 | 766 |

`Uncertainty` is therefore a derived categorical target, not a probability or confidence score. In particular, `Type=3` is split between two uncertainty classes. The physical rule used for that split has not yet been recovered and should not be inferred solely from the numeric values.

### Dataset lineage and model association

```text
TagDatabase.csv       -> early five-class archive -> MLP
data_b_output.csv     -> balanced four-class set  -> Random Forest
data_b_dataset.csv    -> relabelled four-class set -> BiLSTM
ublox_3_tag.csv       -> later masked sequence set -> SMConvLSTM
```

These datasets may reveal experimental locations and trajectories. They should be anonymized or removed before the repository is made public.

### `MLP/MLP_1.py`

Early multilayer-perceptron baseline using mainly `CNR` and `Elevation` to predict `Type`. It includes standardization, balancing, training, validation, and decision-boundary visualization. Its CSV path is a placeholder.

### `Visualized_Path/to kml.f.py`

Reads ECEF/positioning results from CSV, converts them to latitude, longitude, and height, and creates a Google Earth KML trajectory. A clearer future name is `export_trajectory_to_kml.py`.

## 7. `models/`

### Random Forest

- `models/RF/RF.py`: builds fixed windows grouped by site, satellite, and continuous time; flattens `Elevation`, `CNR`, `CNR_rate`, and `Diff_DelPr_PrRate`; uses the window majority class; and reports metrics, confusion matrix, and feature importance.
- `models/RF/RF_nogroup.py`: extracts grouped statistics such as means and standard deviations, allowing statistical aggregation to be compared with flattened windows.

### BiLSTM

- `models/BiLSTM/biLSTM.2.py`: the most complete Python training entry point. It creates overlapping sequences, standardizes inputs, builds bidirectional/standard LSTM and dropout layers, uses focal loss, reports metrics and permutation importance, plots training results, and saves the Keras model. Paths are hard-coded; `train_bilstm.py` would be a clearer name.
- `models/BiLSTM/biLSTM.2 - 副本.py`: an older variant retained for development history. Move it to `archive/` or remove it after confirming no unique experiment depends on it.
- `models/BiLSTM/BiLSTM.m`: MATLAB two-layer BiLSTM prototype with class weights and five-class output. It saves `trained_gnss_model.mat` but depends on missing helpers, and its focal-loss comments do not fully match the configuration.

### Base ConvLSTM

- `models/Base_convLSTM/base_conv_lstm.py`: low-level `ConvLSTMCell` and `ConvLSTM` implementation managing convolutional recurrent state, shapes, dropout, and initialization; it is a layer, not a training entry point.
- `models/Base_convLSTM/base_conv_lstm - 副本.py`: exact historical duplicate of `base_conv_lstm.py`.
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

## 8. Recommended Reading Order

1. `DataProcess/RINEX_processing_main_f.m` — feature generation.
2. `DataTag/zzy_skymask_generation_f.m` — label generation.
3. `TagDatabase/TagDatabase.csv` — schema and class distribution.
4. `MLP/MLP_1.py` or `models/RF/` — conventional baselines.
5. `models/BiLSTM/biLSTM.2.py` — main temporal classifier.
6. `models/SMconLSTM_training.py` and `models/Rho_module.py` — satellite-masked model.
7. `models/load_model_SMconvLSTM.py` — inference.
8. `predict_gnss_quality.m` — planned GNSS/INS integration.

## 9. Dependencies

MATLAB dependencies include MATLAB, Mapping Toolbox, Statistics and Machine Learning Toolbox, and the bundled goGPS, GTSAM, satnav3p01, and ray-tracing code.

Python dependencies include `numpy`, `pandas`, `scipy`, `matplotlib`, `scikit-learn`, `tensorflow`, and `keras`. Exact versions have not been recovered; a future reproducibility pass should add `requirements.txt` or `environment.yml`.

## 10. Known Issues

- Absolute `C:/Users/ZHANG/...` paths remain in several scripts.
- `Type` and `Uncertainty` are not consistently defined.
- Four-class and five-class implementations coexist.
- `predict_gnss_quality.m` references a missing `predict_gnss.py`.
- `pred_Uncertainty` is not a real probability/confidence estimate.
- Historical `- 副本.py` files and multiple Masked ConvLSTM versions remain.
- Third-party MATLAB libraries are duplicated.
- There is no unified configuration, random seed, dependency lock file, or test suite.
- `Gaussian_mixture_1.py` is better suited to the GNSS Error Map repository.

## 11. Third-party Code

goGPS, GTSAM, ParforProgMon, satnav3p01, and other supporting files originate from their respective upstream projects or research codebases. Use and redistribution must comply with the licences and citation requirements in their directories. Inclusion here does not constitute relicensing.
