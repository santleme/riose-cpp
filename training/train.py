#!/usr/bin/env python3
"""Train/evaluate a cow activity classifier on fixed, non-overlapping IMU windows.

The default data source is the Kaggle dataset `fandaoerji/cow-nose-ring-data-set`.
The script also supports local CSVs and synthetic smoke data for offline development.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier

DATASET = "fandaoerji/cow-nose-ring-data-set"
DATASET_LICENSE = "CC0-1.0"
LABELS = {0: "feeding", 1: "rumination", 2: "standing", 3: "lying", 4: "walking"}
AXES = ["acc_x", "acc_y", "acc_z"]
REQUIRED = {"date", "time", "cow_num", *AXES, "behavior"}
WINDOW_SAMPLES = 50  # 5 s at the dataset's 10 Hz sample rate
SAMPLING_HZ = 10
SAMPLE_INTERVAL_SECONDS = 1.0 / SAMPLING_HZ
# Allow modest timestamp quantization/jitter while ensuring a missing 10 Hz
# sample (0.2 s between retained rows) always breaks a window.
SAMPLE_INTERVAL_TOLERANCE_SECONDS = 0.025
FEATURE_NAMES = [
    *(f"{axis}_{stat}" for axis in AXES for stat in ("mean", "std", "min", "max")),
    "mag_mean", "mag_std", "mag_min", "mag_max", "mag_rms", "signal_magnitude_area",
]


def fetch_kaggle_data(cache_dir: Path) -> Path:
    """Fetch the public, CC0-licensed Kaggle CSV using the official CLI."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "data.csv"
    if csv_path.exists():
        return csv_path
    with tempfile.TemporaryDirectory(prefix="campo-vivo-kaggle-") as tmp:
        subprocess.run(
            ["kaggle", "datasets", "download", DATASET, "-f", "data.csv", "-p", tmp],
            check=True,
        )
        archive = next(Path(tmp).glob("*.zip"))
        with zipfile.ZipFile(archive) as zf:
            member = next(n for n in zf.namelist() if n.endswith("data.csv"))
            with zf.open(member) as src, csv_path.open("wb") as dst:
                dst.write(src.read())
    return csv_path


def kaggle_mounted_data() -> Path | None:
    """Find Kaggle's attached dataset without downloading it a second time."""
    input_dir = Path("/kaggle/input")
    if input_dir.is_dir():
        matches = list(input_dir.glob("**/data.csv"))
        if matches:
            return matches[0]
    return None


def _window_features(acc: np.ndarray) -> np.ndarray:
    """Summarize a 5-second tri-axial window with scale/orientation-aware stats."""
    axes = [acc[:, i] for i in range(3)]
    magnitude = np.linalg.norm(acc, axis=1)
    features = []
    for values in axes:
        features.extend((values.mean(), values.std(), values.min(), values.max()))
    features.extend((magnitude.mean(), magnitude.std(), magnitude.min(), magnitude.max(),
                     np.sqrt(np.mean(magnitude ** 2)), np.mean(np.abs(acc))))
    return np.asarray(features, dtype=np.float32)


def _parse_sample_timestamps(frame: pd.DataFrame) -> pd.Series:
    """Parse the dataset's date plus either seconds-of-day or clock time."""
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    raw_time = frame["time"]
    numeric_seconds = pd.to_numeric(raw_time, errors="coerce")
    numeric_mask = numeric_seconds.notna()
    # Numeric timestamps in the source/synthetic fixtures are seconds since
    # midnight; textual clock values use HH:MM:SS[.fraction].
    timedeltas = pd.Series(pd.NaT, index=frame.index, dtype="timedelta64[ns]")
    if numeric_mask.any():
        timedeltas.loc[numeric_mask] = pd.to_timedelta(numeric_seconds.loc[numeric_mask], unit="s")
    textual_mask = ~numeric_mask & raw_time.notna()
    if textual_mask.any():
        timedeltas.loc[textual_mask] = pd.to_timedelta(raw_time.loc[textual_mask].astype(str), errors="coerce")
    return dates + timedeltas


def _extract_windows(frame: pd.DataFrame, window_samples: int = WINDOW_SAMPLES,
                     minimum_label_fraction: float = 0.8
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Extract windows only from contiguous, timestamp-verified 10 Hz runs."""
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    frame = frame.copy()
    frame["cow_num"] = frame["cow_num"].astype(str)
    frame["_source_order"] = np.arange(len(frame), dtype=np.int64)
    frame["_timestamp"] = _parse_sample_timestamps(frame)
    frame["_day"] = frame["_timestamp"].dt.normalize()
    for column in (*AXES, "behavior"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    timestamp_valid = frame["_timestamp"].notna()
    data_valid = frame[[*AXES, "behavior"]].notna().all(axis=1)
    diagnostics = {
        "input_rows": int(len(frame)),
        "invalid_timestamp_rows": int((~timestamp_valid).sum()),
        "missing_sensor_or_label_rows": int((timestamp_valid & ~data_valid).sum()),
        "reordered_rows": 0,
        "duplicate_timestamp_breaks": 0,
        "sampling_gap_breaks": 0,
        "contiguous_runs": 0,
        "candidate_windows": 0,
        "discarded_mixed_label_windows": 0,
        "accepted_windows": 0,
        "sampling_hz": SAMPLING_HZ,
        "sample_interval_tolerance_seconds": SAMPLE_INTERVAL_TOLERANCE_SECONDS,
    }
    frame = frame.loc[timestamp_valid & data_valid].copy()
    frame["behavior"] = frame["behavior"].astype(int)
    features: list[np.ndarray] = []
    labels: list[int] = []
    animals: list[str] = []
    # Sort within each cow/day because CSV row order is not a sampling guarantee.
    # A source-order diagnostic preserves visibility into shuffled input.
    for (animal, _day), group in frame.groupby(["cow_num", "_day"], sort=True):
        original_order = group["_source_order"].to_numpy()
        ordered = group.sort_values(["_timestamp", "_source_order"], kind="stable").reset_index(drop=True)
        diagnostics["reordered_rows"] += int(np.count_nonzero(original_order != ordered["_source_order"].to_numpy()))
        timestamps = ordered["_timestamp"].astype("int64").to_numpy()
        # Split on duplicates and on intervals outside the timestamp precision
        # tolerance around the documented 100 ms sample interval. This also
        # prevents a dropped NaN sensor row from being bridged after dropna.
        deltas = np.diff(timestamps) / 1_000_000_000
        boundaries = [0]
        for index, delta in enumerate(deltas, start=1):
            if delta <= 0:
                diagnostics["duplicate_timestamp_breaks"] += 1
                boundaries.append(index)
            elif abs(delta - SAMPLE_INTERVAL_SECONDS) > SAMPLE_INTERVAL_TOLERANCE_SECONDS:
                diagnostics["sampling_gap_breaks"] += 1
                boundaries.append(index)
        boundaries.append(len(ordered))
        for run_start, run_end in zip(boundaries, boundaries[1:]):
            if run_start == run_end:
                continue
            diagnostics["contiguous_runs"] += 1
            run = ordered.iloc[run_start:run_end]
            acc = run[AXES].to_numpy(dtype=np.float32)
            y = run["behavior"].to_numpy(dtype=np.int64)
            full_windows = len(y) // window_samples
            for index in range(full_windows):
                start = index * window_samples
                segment_labels = y[start:start + window_samples]
                diagnostics["candidate_windows"] += 1
                counts = Counter(segment_labels.tolist())
                label, count = counts.most_common(1)[0]
                if count / window_samples < minimum_label_fraction:
                    diagnostics["discarded_mixed_label_windows"] += 1
                    continue
                features.append(_window_features(acc[start:start + window_samples]))
                labels.append(int(label))
                animals.append(animal)
    diagnostics["accepted_windows"] = len(features)
    if not features:
        raise ValueError("No contiguous labeled windows found; check timestamps, 10 Hz sampling, and CSV schema")
    return np.stack(features), np.asarray(labels), np.asarray(animals), diagnostics


def extract_windows(frame: pd.DataFrame, window_samples: int = WINDOW_SAMPLES,
                    minimum_label_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return features, majority labels, and animal IDs from contiguous 10 Hz windows."""
    x, y, animals, _ = _extract_windows(frame, window_samples, minimum_label_fraction)
    return x, y, animals


def synthetic_frame(seed: int = 23, samples_per_animal: int = 4000) -> pd.DataFrame:
    """Make deterministic labelled IMU-like data for an offline pipeline smoke run."""
    rng = np.random.default_rng(seed)
    rows = []
    for animal in map(str, range(1, 5)):
        base_shift = rng.normal(0, 0.05, size=3)
        for i in range(samples_per_animal):
            label = (i // WINDOW_SAMPLES // 4) % len(LABELS)
            phase = i / 10.0
            centers = np.asarray([
                [0.04, 0.04, 1.00], [0.12, 0.04, 0.96], [0.02, 0.02, 1.00],
                [0.03, 0.03, 0.97], [0.20, 0.10, 0.90],
            ])
            amp = np.asarray([0.03, 0.04, 0.01, 0.02, 0.18])[label]
            point = centers[label] + base_shift + rng.normal(0, amp, 3)
            point[0] += amp * np.sin(phase * 3)
            rows.append(("2026-01-01", f"{i / 10:.1f}", animal, *point, label))
    return pd.DataFrame(rows, columns=["date", "time", "cow_num", *AXES, "behavior"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, help="Input data.csv. Defaults to Kaggle download.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/cow-nose-ring"))
    output_default = Path("/kaggle/working/training/outputs") if Path("/kaggle/working").is_dir() else Path("training/outputs")
    parser.add_argument("--output-dir", type=Path, default=output_default)
    parser.add_argument("--holdout-animal", default="6", help="Animal ID reserved entirely for evaluation.")
    parser.add_argument("--validation-animal", default="1", help="Training animal reserved for model selection.")
    parser.add_argument("--smoke-synthetic", action="store_true", help="Use generated data; never for model claims.")
    parser.add_argument("--trees", type=int, default=300, help="Number of trees for Extra Trees and Random Forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional depth cap for tree baselines.")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    if args.smoke_synthetic:
        frame = synthetic_frame(args.seed)
        source = "deterministic synthetic smoke data"
        license_id = "synthetic"
    else:
        path = args.data or kaggle_mounted_data() or fetch_kaggle_data(args.cache_dir)
        frame = pd.read_csv(path)
        source = f"{DATASET} (data.csv)"
        license_id = DATASET_LICENSE

    x, y, animals, extraction_diagnostics = _extract_windows(frame)
    test_mask = animals == str(args.holdout_animal)
    train_mask = ~test_mask
    validation_mask = animals == str(args.validation_animal)
    fit_mask = train_mask & ~validation_mask
    if not fit_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError(f"Holdout animal {args.holdout_animal!r} is absent or leaves no training data")
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]
    x_fit, y_fit = x[fit_mask], y[fit_mask]
    x_validation, y_validation = x[validation_mask], y[validation_mask]
    if set(np.unique(y_test)) - set(np.unique(y_train)):
        raise ValueError("Holdout animal contains a class absent from training animals")
    if set(np.unique(y_validation)) - set(np.unique(y_fit)):
        raise ValueError("Validation animal contains a class absent from fit animals")

    def make_models():
        return {
        "extra_trees": ExtraTreesClassifier(n_estimators=args.trees, max_depth=args.max_depth,
                                             class_weight="balanced", max_features=0.8,
                                             min_samples_leaf=2, n_jobs=-1, random_state=args.seed),
        "random_forest": RandomForestClassifier(n_estimators=args.trees, max_depth=args.max_depth,
                                                  class_weight="balanced_subsample",
                                                  max_features=0.8, min_samples_leaf=2,
                                                  n_jobs=-1, random_state=args.seed),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(96, 48), activation="relu",
                    alpha=1e-3, batch_size=256, learning_rate_init=1e-3, max_iter=80,
                    early_stopping=True, validation_fraction=0.12, n_iter_no_change=8,
                    random_state=args.seed)),
        }

    validation_scores = {}
    for name, model in make_models().items():
        model.fit(x_fit, y_fit)
        validation_prediction = model.predict(x_validation)
        validation_scores[name] = float(f1_score(y_validation, validation_prediction,
                                                  average="macro", zero_division=0))

    # A small neural baseline can use the assigned Kaggle GPU. It is optional so
    # the exact same script still runs on a lightweight local CPU environment.
    torch_payload = None
    torch_validation_score = None
    if _cuda_available():
        _, torch_validation, _ = _train_torch_mlp(x_fit, y_fit, x_validation, y_validation, args.seed)
        torch_validation_score = torch_validation["macro_f1"]
        validation_scores["torch_mlp_cuda"] = torch_validation_score

    # Model selection uses animal 1 only; the animal 6 test set stays untouched
    # until the selected model and all comparison baselines are refit.
    best_name = max(validation_scores, key=validation_scores.get)
    results = {}
    fitted = {}
    for name, model in make_models().items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        results[name] = _metrics(y_test, prediction, validation_scores[name])
        fitted[name] = model
    if _cuda_available():
        torch_model, torch_result, torch_payload = _train_torch_mlp(x_train, y_train, x_test, y_test, args.seed)
        torch_result["validation_macro_f1"] = validation_scores["torch_mlp_cuda"]
        results["torch_mlp_cuda"] = torch_result
        fitted["torch_mlp_cuda"] = torch_model
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if best_name == "torch_mlp_cuda":
        import torch
        torch.save(torch_payload, args.output_dir / "activity_classifier.pt")
    else:
        joblib.dump(fitted[best_name], args.output_dir / "activity_classifier.joblib", compress=3)
    metadata = {
        "source": source, "license": license_id, "sampling_hz": SAMPLING_HZ, "window_seconds": 5,
        "window_samples": WINDOW_SAMPLES, "minimum_majority_fraction": 0.8,
        "window_extraction": extraction_diagnostics,
        "feature_names": FEATURE_NAMES, "label_map": LABELS, "seed": args.seed,
        "holdout_animal": str(args.holdout_animal),
        "train_animals": sorted(set(animals[train_mask].tolist())),
        "test_animals": sorted(set(animals[test_mask].tolist())),
        "validation_animal": str(args.validation_animal),
        "fit_animals": sorted(set(animals[fit_mask].tolist())),
        "windows": {"total": int(len(y)), "fit": int(fit_mask.sum()), "validation": int(validation_mask.sum()),
                    "train": int(train_mask.sum()), "test": int(test_mask.sum()),
                    "fit_class_counts": {LABELS[int(k)]: int(v) for k, v in Counter(y_fit).items()},
                    "validation_class_counts": {LABELS[int(k)]: int(v) for k, v in Counter(y_validation).items()},
                    "train_class_counts": {LABELS[int(k)]: int(v) for k, v in Counter(y_train).items()},
                    "test_class_counts": {LABELS[int(k)]: int(v) for k, v in Counter(y_test).items()}},
        "models": results, "selected_model": best_name,
        "model_selection_animal": str(args.validation_animal),
        "model_artifact": "activity_classifier.pt" if best_name == "torch_mlp_cuda" else "activity_classifier.joblib",
        "evaluation_note": "Single-animal group holdout pilot; windows require timestamp-contiguous 10 Hz samples. Historical Kaggle metrics predate this continuity fix and require a rerun. Use leave-one-animal-out cross-validation before biological/generalization claims.",
        "runtime": {"cuda_available": bool(_cuda_available()), "device": "cuda" if _cuda_available() else "cpu",
                    "gpu_name": _gpu_name()},
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"source": source, "license": license_id, "selected_model": best_name,
                      "test_animals": metadata["test_animals"], "windows": metadata["windows"],
                      "macro_f1": {n: round(v["macro_f1"], 5) for n, v in results.items()},
                      "output_dir": str(args.output_dir)}, indent=2))


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_name() -> str | None:
    if not _cuda_available():
        return None
    import torch
    return torch.cuda.get_device_name(0)


def _metrics(y_true: np.ndarray, prediction: np.ndarray, validation_macro_f1: float) -> dict:
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    return {
        "validation_macro_f1": float(validation_macro_f1),
        "macro_f1": float(f1_score(y_true, prediction, labels=sorted(LABELS), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, labels=sorted(LABELS), average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, prediction, labels=sorted(LABELS),
            target_names=[LABELS[i] for i in sorted(LABELS)], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=sorted(LABELS)).tolist(),
    }


def _train_torch_mlp(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray,
                     y_test: np.ndarray, seed: int):
    """Fit a compact neural model on CUDA and return checkpoint + held-out metrics."""
    import torch
    from torch import nn
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    scaler = StandardScaler().fit(x_train)
    xt = torch.tensor(scaler.transform(x_train), dtype=torch.float32, device=device)
    xv = torch.tensor(scaler.transform(x_test), dtype=torch.float32, device=device)
    classes = np.asarray(sorted(LABELS), dtype=np.int64)
    to_index = {int(label): i for i, label in enumerate(classes)}
    yt = torch.tensor([to_index[int(v)] for v in y_train], dtype=torch.long, device=device)
    class_counts = np.bincount(yt.cpu().numpy(), minlength=len(classes))
    weights = len(yt) / (len(classes) * np.maximum(class_counts, 1))
    loss_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    net = nn.Sequential(nn.Linear(x_train.shape[1], 96), nn.ReLU(), nn.Dropout(0.15),
                        nn.Linear(96, 48), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(48, len(classes))).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=loss_weights)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _epoch in range(45):
        net.train()
        order = torch.randperm(len(xt), generator=generator)
        for batch in order.split(1024):
            indices = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(xt[indices]), yt[indices])
            loss.backward()
            optimizer.step()
    net.eval()
    with torch.no_grad():
        predicted_index = net(xv).argmax(1).cpu().numpy()
    prediction = classes[predicted_index]
    result = {
        "macro_f1": float(f1_score(y_test, prediction, labels=classes, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, prediction, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, prediction, labels=classes,
            target_names=[LABELS[int(i)] for i in classes], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=classes).tolist(),
    }
    checkpoint = {"state_dict": net.cpu().state_dict(), "scaler_mean": scaler.mean_.tolist(),
                  "scaler_scale": scaler.scale_.tolist(), "classes": classes.tolist(),
                  "feature_names": FEATURE_NAMES, "hidden_sizes": [96, 48], "seed": seed}
    return net, result, checkpoint


if __name__ == "__main__":
    main()
