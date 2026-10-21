"""Run a reproducible four-model benchmark on both repository datasets.

All models use the same three features, label mapping, PRN-wise windows, and
chronological 80/20 split.  Run from the repository root with:

    python models/unified_benchmark.py
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM

from Rho_module import SMConvLSTM


SEED = 42
FEATURES = ["Elevation", "Azimuth", "CNR"]
WINDOW = 10
TRAIN_FRACTION = 0.8
CLASS_NAMES = ["NLOS/Multipath", "LOS", "Reflection", "Diffraction"]
ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "TagDatabase": ROOT / "TagDatabase" / "TagDatabase.csv",
    "data_b": ROOT / "TagDatabase" / "data_b.csv",
}


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def make_windows(path: Path):
    df = pd.read_csv(path, usecols=["GPS_Time", "PRN", "Type", *FEATURES])
    df = df.dropna(subset=["GPS_Time", "PRN", "Type", *FEATURES]).copy()
    # The large archive has an extra LOS+diffraction class.  The common
    # four-class benchmark folds it into diffraction.
    df["label"] = df["Type"].astype(int).replace({4: 3})
    df = df.sort_values(["PRN", "GPS_Time"], kind="stable")

    train_x, train_y, test_x, test_y = [], [], [], []
    for _, group in df.groupby("PRN", sort=False):
        x = group[FEATURES].to_numpy(np.float32)
        y = group["label"].to_numpy(np.int32)
        n = len(group) - WINDOW + 1
        if n < 2:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(x, (WINDOW, x.shape[1]))[:, 0]
        targets = y[WINDOW - 1 :]
        cut = max(1, min(n - 1, int(n * TRAIN_FRACTION)))
        train_x.append(windows[:cut]); train_y.append(targets[:cut])
        test_x.append(windows[cut:]); test_y.append(targets[cut:])

    return tuple(np.concatenate(parts) for parts in (train_x, train_y, test_x, test_y))


def scale_sequences(x_train, x_test):
    scaler = StandardScaler().fit(x_train.reshape(-1, len(FEATURES)))
    train = scaler.transform(x_train.reshape(-1, len(FEATURES))).reshape(x_train.shape)
    test = scaler.transform(x_test.reshape(-1, len(FEATURES))).reshape(x_test.shape)
    return train.astype(np.float32), test.astype(np.float32)


def class_weights(y):
    classes = np.arange(4)
    values = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes.tolist(), values.tolist()))


def mlp_model():
    inputs = Input((len(FEATURES),))
    x = Dense(64, activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    return Model(inputs, Dense(4, activation="softmax")(x), name="MLP")


def bilstm_model():
    inputs = Input((WINDOW, len(FEATURES)))
    x = Bidirectional(LSTM(32))(inputs)
    x = Dropout(0.2)(x)
    return Model(inputs, Dense(4, activation="softmax")(x), name="BiLSTM")


def smconvlstm_model():
    inputs = Input((WINDOW, 1, 1, len(FEATURES)), name="features")
    mask = Input((WINDOW, 1, 1), name="spatial_mask")
    x = SMConvLSTM(rank=2, filters=8, kernel_size=(1, 1), padding="same",
                   return_sequences=False)(inputs, Smask=mask)
    x = tf.keras.layers.Flatten()(x)
    return Model([inputs, mask], Dense(4, activation="softmax")(x), name="SMConvLSTM")


def fit_neural(model, train_input, y_train, test_input, y_test):
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_input, y_train, validation_data=(test_input, y_test), epochs=15,
        batch_size=512, class_weight=class_weights(y_train), verbose=2,
        callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    )
    return np.argmax(model.predict(test_input, batch_size=1024, verbose=0), axis=1)


def metrics(y_true, y_pred):
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_class = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(4), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_class_f1": {CLASS_NAMES[i]: float(per_class[2][i]) for i in range(4)},
    }


def run_dataset(name, path):
    print(f"\n===== {name}: {path.name} =====")
    x_train, y_train, x_test, y_test = make_windows(path)
    x_train, x_test = scale_sequences(x_train, x_test)
    print(f"windows: train={len(y_train):,}, test={len(y_test):,}; features={FEATURES}")
    results = {}

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
                                random_state=SEED, min_samples_leaf=2)
    rf.fit(x_train[:, -1], y_train)
    results["Random Forest"] = metrics(y_test, rf.predict(x_test[:, -1]))
    print("Random Forest", results["Random Forest"])

    set_seed(); model = mlp_model()
    pred = fit_neural(model, x_train[:, -1], y_train, x_test[:, -1], y_test)
    results["MLP"] = metrics(y_test, pred); print("MLP", results["MLP"])

    set_seed(); model = bilstm_model()
    pred = fit_neural(model, x_train, y_train, x_test, y_test)
    results["BiLSTM"] = metrics(y_test, pred); print("BiLSTM", results["BiLSTM"])

    train_5d = x_train[:, :, None, None, :]
    test_5d = x_test[:, :, None, None, :]
    train_mask = np.ones(train_5d.shape[:-1], dtype=np.float32)
    test_mask = np.ones(test_5d.shape[:-1], dtype=np.float32)
    set_seed(); model = smconvlstm_model()
    pred = fit_neural(model, [train_5d, train_mask], y_train, [test_5d, test_mask], y_test)
    results["SMConvLSTM"] = metrics(y_test, pred); print("SMConvLSTM", results["SMConvLSTM"])

    return {"rows": int(pd.read_csv(path, usecols=["Type"]).shape[0]),
            "train_windows": int(len(y_train)), "test_windows": int(len(y_test)), "models": results}


def main():
    set_seed()
    output = {
        "configuration": {"seed": SEED, "features": FEATURES, "window": WINDOW,
                          "split": "chronological 80/20 within each PRN",
                          "label_mapping": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1],
                                            "2": CLASS_NAMES[2], "3/4": CLASS_NAMES[3]}},
        "datasets": {},
    }
    for name, path in DATASETS.items():
        output["datasets"][name] = run_dataset(name, path)
    out_path = ROOT / "BENCHMARK_RESULTS.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
