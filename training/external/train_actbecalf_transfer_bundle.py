#!/usr/bin/env python3
"""Train/evaluate a cow activity classifier on fixed, non-overlapping IMU windows.

The default data source is the Kaggle dataset `fandaoerji/cow-nose-ring-data-set`.
The script also supports local CSVs and synthetic smoke data for offline development.
"""
from __future__ import annotations

import argparse
import json
import hashlib
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
DATASET_CSV_SHA256 = "1baf4a55433777522bb65022416566f9269ced58b53e0b6746669ee539c8cb0f"
LABELS = {0: "feeding", 1: "rumination", 2: "standing", 3: "lying", 4: "walking"}
AXES = ["acc_x", "acc_y", "acc_z"]
REQUIRED = {"date", "time", "cow_num", *AXES, "behavior"}
WINDOW_SAMPLES = 50  # 5 s at the dataset's 10 Hz sample rate
SAMPLING_HZ = 10
FEATURE_NAMES = [
    *(f"{axis}_{stat}" for axis in AXES for stat in ("mean", "std", "min", "max")),
    "mag_mean", "mag_std", "mag_min", "mag_max", "mag_rms", "signal_magnitude_area",
]


def fetch_kaggle_data(cache_dir: Path) -> Path:
    """Fetch the public, CC0-licensed Kaggle CSV using the official CLI."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / "data.csv"
    if csv_path.exists():
        verify_dataset_hash(csv_path)
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
    verify_dataset_hash(csv_path)
    return csv_path


def verify_dataset_hash(path: Path, allow_unpinned: bool = False) -> str:
    """Verify that a real Kaggle run uses the audited, immutable CSV snapshot."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATASET_CSV_SHA256 and not allow_unpinned:
        raise ValueError(
            f"Unpinned Kaggle data at {path}: SHA-256 {digest}; expected "
            f"{DATASET_CSV_SHA256}. Audit and explicitly opt in before using a new version."
        )
    return digest


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
        "timestamp_blocks": 0,
        "coarse_timestamp_rows": 0,
        "ambiguous_timestamp_blocks": 0,
        "sampling_gap_breaks": 0,
        "contiguous_runs": 0,
        "candidate_windows": 0,
        "discarded_mixed_label_windows": 0,
        "accepted_windows": 0,
        "sampling_hz_assumed_from_dataset_card": SAMPLING_HZ,
        "timestamp_resolution_note": "The Kaggle CSV stores whole-second clock values repeated across multiple sensor rows; within-second sample order is inferred from CSV order and the documented sampling rate.",
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
        # The actual Kaggle CSV stores whole-second timestamps repeated across
        # sensor rows (usually 20 rows per timestamp, then +2 seconds). These
        # are coarse timestamp buckets, not duplicate samples. Infer a block's
        # expected row count from elapsed seconds * documented 10 Hz. Exclude
        # blocks whose count disagrees, so missing/extra rows cannot be bridged.
        changes = np.flatnonzero(np.diff(timestamps) != 0) + 1
        block_starts = np.concatenate(([0], changes))
        block_ends = np.concatenate((changes, [len(ordered)]))
        block_times = timestamps[block_starts]
        block_counts = block_ends - block_starts
        diagnostics["timestamp_blocks"] += int(len(block_starts))
        diagnostics["coarse_timestamp_rows"] += int(np.sum(block_counts[block_counts > 1]))
        if len(block_starts) == 1:
            diagnostics["ambiguous_timestamp_blocks"] += 1
            continue

        elapsed_seconds = np.diff(block_times) / 1_000_000_000
        expected_counts = np.rint(elapsed_seconds * SAMPLING_HZ).astype(np.int64)
        valid_blocks = np.zeros(len(block_counts), dtype=bool)
        valid_blocks[:-1] = (elapsed_seconds > 0) & (block_counts[:-1] == expected_counts)
        # The final block has no following timestamp; use the previous observed
        # interval as its expected duration. This is an assumption made explicit
        # by the coarse timestamp resolution and the dataset's 10 Hz metadata.
        valid_blocks[-1] = elapsed_seconds[-1] > 0 and block_counts[-1] == expected_counts[-1]
        diagnostics["ambiguous_timestamp_blocks"] += int((~valid_blocks).sum())
        diagnostics["sampling_gap_breaks"] += int(np.count_nonzero((elapsed_seconds > 0) & (~valid_blocks[:-1])))

        block_index = 0
        while block_index < len(block_starts):
            if not valid_blocks[block_index]:
                block_index += 1
                continue
            run_start = block_index
            block_index += 1
            while block_index < len(block_starts) and valid_blocks[block_index]:
                block_index += 1
            run_start_row = int(block_starts[run_start])
            run_end_row = int(block_ends[block_index - 1])
            diagnostics["contiguous_runs"] += 1
            run = ordered.iloc[run_start_row:run_end_row]
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
    parser.add_argument("--allow-unpinned-data", action="store_true",
                        help="Allow a changed Kaggle CSV after explicitly auditing its schema, license, and contents.")
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail before loading data unless a CUDA GPU is visible to PyTorch.")
    parser.add_argument("--trees", type=int, default=300, help="Number of trees for Extra Trees and Random Forest.")
    parser.add_argument("--max-depth", type=int, default=None, help="Optional depth cap for tree baselines.")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    # Script kernels do not expose a reliable user-argument channel through
    # Kaggle's metadata. Treat an attached Kaggle dataset as a GPU run and fail
    # before data parsing unless CUDA is actually available.
    kaggle_input_attached = Path("/kaggle/input").is_dir()
    cuda_required = args.require_cuda or (kaggle_input_attached and not args.smoke_synthetic)
    if cuda_required and not _cuda_available():
        raise RuntimeError("CUDA was required but PyTorch cannot access a CUDA device")

    if args.smoke_synthetic:
        frame = synthetic_frame(args.seed)
        source = "deterministic synthetic smoke data"
        license_id = "synthetic"
    else:
        path = args.data or kaggle_mounted_data() or fetch_kaggle_data(args.cache_dir)
        data_sha256 = verify_dataset_hash(path, allow_unpinned=args.allow_unpinned_data)
        frame = pd.read_csv(path)
        source = f"{DATASET} (data.csv)"
        license_id = DATASET_LICENSE
    if args.smoke_synthetic:
        data_sha256 = None

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
        "source_sha256": data_sha256,
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
        "runtime": {"cuda_required": bool(cuda_required), "cuda_available": bool(_cuda_available()), "device": "cuda" if _cuda_available() else "cpu",
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
    optimizer_steps = 0
    epochs = 45
    for _epoch in range(epochs):
        net.train()
        order = torch.randperm(len(xt), generator=generator)
        for batch in order.split(1024):
            indices = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(xt[indices]), yt[indices])
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
    net.eval()
    with torch.no_grad():
        predicted_index = net(xv).argmax(1).cpu().numpy()
    prediction = classes[predicted_index]
    result = {
        "training": {"device": str(device), "epochs": epochs, "optimizer_steps": optimizer_steps},
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

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ZENODO_DOI = "10.5281/zenodo.13259482"
ZENODO_VERSION = "1.0"
EXPECTED_CSV_BYTES = 177_608_717
EXPECTED_ROWS = 2_467_162
EXPECTED_CALVES = 30
EXPECTED_CSV_SHA256 = "1aec91cc9454919e865be204d503899a89ebdba3ca5f81ee2cb22942f456f26d"
SAMPLE_RATE_HZ = 25
WINDOW_SECONDS = 5
WINDOW_SAMPLES = SAMPLE_RATE_HZ * WINDOW_SECONDS
REQUIRED_COLUMNS = {"dateTime", "calfid", "accX", "accY", "accZ", "behavior", "segId"}
COARSE_LABELS = {"feeding": 0, "rumination": 1, "other": 2}
KNOWN_BEHAVIORS = {
    "srs", "abnormal", "backward", "cross-suckle_other", "cross-suckle_udder", "defecation",
    "drinking", "drinking_electrolytes", "drinking_milk", "drinking|water", "eating",
    "eating_bedding", "eating_concentrates", "eating_forage", "fall", "grooming", "grooming|none",
    "grooming_lying", "headbutt", "jump", "lying", "lying-down", "mount",
    "oral_manipulation_of_pen", "play", "play_object", "rising", "rub", "rumination",
    "rumination_lying", "running", "scratch", "sniff", "sniff_lying", "sniff_walking",
    "social", "social_groom", "social_groom_lying", "social_nudge", "social_nudge_lying",
    "social_sniff", "social_sniff_lying", "standing", "stretch", "tongue_rolling", "cough",
    "tongue_rolling_lying", "urination", "vocalization", "walking",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_features(acc: np.ndarray) -> np.ndarray:
    """Match the CNR 5 s feature contract; raw axes intentionally retain domain shift."""
    axes = [acc[:, i] for i in range(3)]
    magnitude = np.linalg.norm(acc, axis=1)
    features = []
    for values in axes:
        features.extend((values.mean(), values.std(), values.min(), values.max()))
    features.extend((magnitude.mean(), magnitude.std(), magnitude.min(), magnitude.max(),
                     np.sqrt(np.mean(magnitude ** 2)), np.mean(np.abs(acc))))
    return np.asarray(features, dtype=np.float32)


def coarse_label(value: str) -> int:
    """Map the official composite label strings to a coarse activity ontology."""
    normalized = str(value).strip().casefold()
    if normalized in {"eating", "eating_bedding", "eating_concentrates", "eating_forage"}:
        return COARSE_LABELS["feeding"]
    if normalized in {"rumination", "rumination_lying"}:
        return COARSE_LABELS["rumination"]
    return COARSE_LABELS["other"]


def extract_actbecalf(path: Path, *, verify_release: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return disjoint 5 s features, coarse labels, calf IDs, and audit counts."""
    if verify_release and path.stat().st_size != EXPECTED_CSV_BYTES:
        raise ValueError(f"Unexpected ActBeCalf file size: {path.stat().st_size}")
    if verify_release:
        digest = sha256_file(path)
        if digest != EXPECTED_CSV_SHA256:
            raise ValueError(f"ActBeCalf release SHA-256 mismatch: {digest}")
    frame = pd.read_csv(path)
    if "calfId" in frame.columns and "calfid" not in frame.columns:
        frame = frame.rename(columns={"calfId": "calfid"})
    if "behaviour" in frame.columns and "behavior" not in frame.columns:
        frame = frame.rename(columns={"behaviour": "behavior"})
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"ActBeCalf is missing columns: {sorted(missing)}")
    if verify_release and len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Unexpected ActBeCalf row count: {len(frame)}")

    frame = frame.copy()
    frame["calfid"] = pd.to_numeric(frame["calfid"], errors="raise").astype(np.int32)
    for column in ("accX", "accY", "accZ"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["accX", "accY", "accZ"]].isna().any().any():
        raise ValueError("ActBeCalf contains missing/non-numeric accelerometer values")
    # The public CSV contains mixed ISO precision (whole seconds and fractional
    # seconds); pandas >=3 requires opting into per-value format inference.
    frame["_time"] = pd.to_datetime(frame["dateTime"], errors="coerce", utc=True, format="mixed")
    if frame["_time"].isna().any():
        raise ValueError("ActBeCalf contains invalid timestamps")
    if frame["behavior"].isna().any() or frame["segId"].isna().any():
        raise ValueError("ActBeCalf contains missing behavior labels or segment IDs")
    frame["segId"] = frame["segId"].astype(str)
    frame["_behavior"] = frame["behavior"].astype(str).str.strip()
    unknown = sorted(set(frame["_behavior"].str.casefold()) - KNOWN_BEHAVIORS)
    if unknown:
        raise ValueError(f"Unrecognized ActBeCalf ethogram labels: {unknown}")
    frame["_label"] = frame["_behavior"].map(coarse_label).astype(np.int8)
    frame = frame.sort_values(["calfid", "_time", "segId"], kind="stable")
    calves = sorted(frame["calfid"].unique().tolist())
    if verify_release and len(calves) != EXPECTED_CALVES:
        raise ValueError(f"Unexpected calf IDs: {calves}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    animal_ids: list[int] = []
    audit = {
        "input_rows": int(len(frame)),
        "calves": int(len(calves)),
        "segments": 0,
        "timestamp_gap_splits": 0,
        "candidate_windows": 0,
        "accepted_windows": 0,
        "discarded_short_run_samples": 0,
        "sampling_hz": SAMPLE_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "observed_ethogram_labels": {str(label): int(count) for label, count in frame["_behavior"].value_counts().sort_index().items()},
        "coarse_label_mapping": {
            "eating_and_observed_modifiers": "feeding",
            "rumination_and_lying_modifier": "rumination",
            "all_other_observed_ethogram_labels": "other",
        },
        "segment_label_conflicts": 0,
    }
    expected_ns = 1e9 / SAMPLE_RATE_HZ
    # Segment IDs may repeat across calves, so always group on both fields.
    for (calf, _segment), group in frame.groupby(["calfid", "segId"], sort=False, dropna=False):
        audit["segments"] += 1
        if group["_behavior"].nunique() != 1:
            audit["segment_label_conflicts"] += 1
            raise ValueError(f"Segment has multiple mapped labels: calf={calf}, segment={_segment}")
        group = group.sort_values("_time", kind="stable")
        # Pandas 3 may keep datetime64[us]; convert explicitly to ns so that
        # the sampling-gap threshold uses the same unit on every pandas version.
        times = group["_time"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        acc = group[["accX", "accY", "accZ"]].to_numpy(dtype=np.float32)
        label = int(group["_label"].iloc[0])
        # Split on missing samples; never join disjoint samples into a fake bout.
        diffs = np.diff(times)
        breaks = np.flatnonzero((diffs < 0.5 * expected_ns) | (diffs > 1.5 * expected_ns)) + 1
        audit["timestamp_gap_splits"] += int(len(breaks))
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(group)]))
        for start, end in zip(starts, ends):
            run = acc[int(start):int(end)]
            full_windows = len(run) // WINDOW_SAMPLES
            audit["candidate_windows"] += int(full_windows)
            audit["discarded_short_run_samples"] += int(len(run) - full_windows * WINDOW_SAMPLES)
            for index in range(full_windows):
                left = index * WINDOW_SAMPLES
                features.append(window_features(run[left:left + WINDOW_SAMPLES]))
                labels.append(label)
                animal_ids.append(int(calf))
    audit["accepted_windows"] = len(features)
    if not features:
        raise ValueError("No complete 5 s windows after respecting calf/segment boundaries")
    return np.stack(features), np.asarray(labels, dtype=np.int64), np.asarray(animal_ids, dtype=np.int32), audit


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-non-release-fixture", action="store_true")
    args = parser.parse_args()
    x, y, calves, audit = extract_actbecalf(args.csv, verify_release=not args.allow_non_release_fixture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y, calves=calves)
    audit.update({"sha256": sha256_file(args.csv), "source_doi": ZENODO_DOI, "source_version": ZENODO_VERSION})
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    from training.train import _extract_windows, verify_dataset_hash
    from training.external.actbecalf import extract_actbecalf, sha256_file
except ModuleNotFoundError:
    # Kaggle script kernels receive the generated single-file bundle; its
    # companion functions have already been defined in this module.
    pass

SEED = 20261022
HELD_OUT_CNR_ANIMAL = "6"
CLASS_NAMES = ["feeding", "rumination", "other"]
CNR_TO_COARSE = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}


def score_grouped(y_true: np.ndarray, prediction: np.ndarray, animals: np.ndarray, seed: int) -> dict:
    labels = [0, 1, 2]
    by_calf = []
    for calf in sorted(np.unique(animals)):
        mask = animals == calf
        supported = [label for label in labels if np.any(y_true[mask] == label)]
        by_calf.append(float(f1_score(y_true[mask], prediction[mask], labels=supported,
                                      average="macro", zero_division=0)))
    rng = np.random.default_rng(seed)
    unique_calves = np.unique(animals)
    pooled_boot = []
    animal_mean_boot = []
    for _ in range(2000):
        chosen = rng.choice(unique_calves, size=len(unique_calves), replace=True)
        indices = np.concatenate([np.flatnonzero(animals == calf) for calf in chosen])
        pooled_boot.append(float(f1_score(y_true[indices], prediction[indices], labels=labels,
                                          average="macro", zero_division=0)))
        sampled_animal_scores = []
        for calf in chosen:
            mask = animals == calf
            supported = [label for label in labels if np.any(y_true[mask] == label)]
            sampled_animal_scores.append(float(f1_score(y_true[mask], prediction[mask], labels=supported,
                                                         average="macro", zero_division=0)))
        animal_mean_boot.append(float(np.mean(sampled_animal_scores)))
    return {
        "macro_f1_window_pooled": float(f1_score(y_true, prediction, labels=labels, average="macro", zero_division=0)),
        "weighted_f1_window_pooled": float(f1_score(y_true, prediction, labels=labels, average="weighted", zero_division=0)),
        "macro_f1_animal_mean": float(np.mean(by_calf)),
        "macro_f1_animal_sd": float(np.std(by_calf, ddof=1)),
        "class_supported_animals_n": {CLASS_NAMES[label]: int(sum(np.any((animals == calf) & (y_true == label)) for calf in unique_calves))
                                      for label in labels},
        "macro_f1_pooled_animal_cluster_bootstrap_95ci": [float(np.quantile(pooled_boot, .025)), float(np.quantile(pooled_boot, .975))],
        "macro_f1_animal_mean_cluster_bootstrap_95ci": [float(np.quantile(animal_mean_boot, .025)), float(np.quantile(animal_mean_boot, .975))],
        "animal_class_support_min_max": {
            CLASS_NAMES[label]: [int(min(np.count_nonzero((animals == calf) & (y_true == label)) for calf in unique_calves)),
                                 int(max(np.count_nonzero((animals == calf) & (y_true == label)) for calf in unique_calves))]
            for label in labels
        },
        "classification_report": classification_report(y_true, prediction, labels=labels,
            target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels).tolist(),
        "predicted_class_counts": {CLASS_NAMES[label]: int(np.count_nonzero(prediction == label))
                                   for label in labels},
        "test_animals_n": int(len(unique_calves)),
        "window_count": int(len(y_true)),
    }


def train_extra_trees(x: np.ndarray, y: np.ndarray) -> ExtraTreesClassifier:
    # Exact frozen v5 tree family and settings; no tuning on ActBeCalf.
    model = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", max_features=0.8,
                                 min_samples_leaf=2, n_jobs=-1, random_state=23)
    return model.fit(x, y)


def train_torch_mlp(x: np.ndarray, y: np.ndarray, seed: int = SEED):
    import torch
    from torch import nn
    from sklearn.preprocessing import StandardScaler
    if not torch.cuda.is_available():
        raise RuntimeError("External transfer run requires Kaggle CUDA; refusing CPU fallback")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    scaler = StandardScaler().fit(x)
    xt = torch.tensor(scaler.transform(x), dtype=torch.float32, device="cuda")
    classes = np.asarray([0, 1, 2], dtype=np.int64)
    yt = torch.tensor(y, dtype=torch.long, device="cuda")
    counts = np.bincount(y, minlength=3)
    weights = torch.tensor(len(y) / (3 * np.maximum(counts, 1)), dtype=torch.float32, device="cuda")
    net = nn.Sequential(nn.Linear(x.shape[1], 96), nn.ReLU(), nn.Dropout(.15),
                        nn.Linear(96, 48), nn.ReLU(), nn.Dropout(.1), nn.Linear(48, 3)).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    updates = 0
    for _ in range(45):
        net.train()
        order = torch.randperm(len(xt), generator=generator)
        for batch in order.split(1024):
            index = batch.to("cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(xt[index]), yt[index])
            loss.backward()
            optimizer.step()
            updates += 1
    return net.eval(), scaler, classes, updates


def torch_predict(model, scaler, x: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        xt = torch.tensor(scaler.transform(x), dtype=torch.float32, device="cuda")
        return model(xt).argmax(1).cpu().numpy()


def _resolve_input(explicit: Path | None, file_name: str, preferred: str) -> Path:
    if explicit is not None:
        return explicit
    root = Path("/kaggle/input")
    matches = sorted(root.glob(f"**/{file_name}")) if root.exists() else []
    exact = [path for path in matches if preferred in str(path.parent).lower()]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"Expected one Kaggle input {file_name}; found {[str(p) for p in matches]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnr-csv", type=Path)
    parser.add_argument("--actbecalf-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/actbecalf-transfer"))
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Kaggle experiment")

    args.cnr_csv = _resolve_input(args.cnr_csv, "data.csv", "cow-nose-ring")
    args.actbecalf_csv = _resolve_input(args.actbecalf_csv, "AcTBeCalf.csv", "actbecalf")
    cnr_hash = verify_dataset_hash(args.cnr_csv)
    cnr = pd.read_csv(args.cnr_csv)
    x_cnr, y_cnr_original, cnr_animals, cnr_audit = _extract_windows(cnr)
    train_mask = cnr_animals != HELD_OUT_CNR_ANIMAL
    if not np.any(cnr_animals == HELD_OUT_CNR_ANIMAL):
        raise ValueError("Pinned CNR holdout animal 6 is absent")
    y_cnr = np.asarray([CNR_TO_COARSE[int(label)] for label in y_cnr_original], dtype=np.int64)
    x_train, y_train = x_cnr[train_mask], y_cnr[train_mask]
    x_external, y_external, external_animals, external_audit = extract_actbecalf(args.actbecalf_csv)
    if x_external.shape[1] != x_train.shape[1]:
        raise ValueError("External and CNR feature contracts differ")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {}
    training_class_counts = {CLASS_NAMES[label]: int(np.count_nonzero(y_train == label))
                             for label in range(3)}
    training_majority_class = int(np.argmax(np.bincount(y_train, minlength=3)))
    majority_prediction = np.full_like(y_external, training_majority_class)
    metrics["training_majority_baseline"] = score_grouped(
        y_external, majority_prediction, external_animals, SEED)
    tree = train_extra_trees(x_train, y_train)
    tree_prediction = tree.predict(x_external)
    metrics["extra_trees_v5_frozen"] = score_grouped(y_external, tree_prediction, external_animals, SEED)
    import joblib
    joblib.dump(tree, args.output_dir / "actbecalf-transfer-extra-trees.joblib", compress=3)

    mlp, scaler, classes, updates = train_torch_mlp(x_train, y_train)
    mlp_prediction = classes[torch_predict(mlp, scaler, x_external)]
    metrics["torch_mlp_v5_frozen"] = score_grouped(y_external, mlp_prediction, external_animals, SEED)
    import torch
    torch.save({"state_dict": mlp.cpu().state_dict(), "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(), "classes": classes.tolist(),
                "feature_names": ["axis means/std/min/max", "magnitude mean/std/min/max/rms", "signal magnitude area"],
                "seed": SEED, "optimizer_steps": updates}, args.output_dir / "actbecalf-transfer-mlp.pt")

    source_hashes = {
        "base_training_parser": globals().get("BASE_TRAIN_SOURCE_SHA256"),
        "actbecalf_loader": globals().get("ACTBECALF_LOADER_SOURCE_SHA256"),
        "transfer_runner": globals().get("TRANSFER_RUNNER_SOURCE_SHA256"),
    }
    if not all(source_hashes.values()):
        repo_training = Path(__file__).resolve().parents[1]
        source_paths = {"base_training_parser": repo_training / "train.py",
                        "actbecalf_loader": Path(__file__).with_name("actbecalf.py"),
                        "transfer_runner": Path(__file__)}
        source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    artifact_names = ["actbecalf-transfer-extra-trees.joblib", "actbecalf-transfer-mlp.pt"]
    artifacts = {name: sha256_file(args.output_dir / name) for name in artifact_names}

    manifest = {
        "experiment": "cnr_to_actbecalf_frozen_external_transfer_v1",
        "status": "complete_exploratory_external_cohort_test",
        "training_source": {"name": "Kaggle Cow Nose Ring Data Set", "slug": "fandaoerji/cow-nose-ring-data-set",
            "license": "CC0-1.0", "csv_sha256": cnr_hash, "cohort_holdout_excluded_from_fit": HELD_OUT_CNR_ANIMAL,
            "windows": int(len(y_train)), "animals": int(len(np.unique(cnr_animals[train_mask]))),
            "coarse_class_counts": training_class_counts,
            "majority_class": CLASS_NAMES[training_majority_class]},
        "external_test_source": {"name": "ActBeCalf", "doi": "10.5281/zenodo.13259482", "version": "1.0",
            "license": "CC-BY-4.0", "csv_sha256": sha256_file(args.actbecalf_csv), "audit": external_audit,
            "coarse_window_support": {CLASS_NAMES[label]: int(np.count_nonzero(y_external == label)) for label in range(3)}},
        "ontology_mapping": {"cnr_codes": CNR_TO_COARSE,
            "actbecalf": {"eating_and_eating_modifiers": 0, "rumination_and_posture_modifiers": 1,
                          "all_other_observed_ethogram_values": 2},
            "caveat": "Kaggle CNR's 5-code mapping is the repository's audited parser ontology; source paper reports a coarser feeding/rumination/other task. This is cross-domain activity transfer, not a decision-policy test."},
        "windowing": {"seconds": 5, "cnr_rate_hz": 10, "actbecalf_rate_hz": 25,
            "features": "same 18 per-axis and magnitude summary features; windows never cross calf, behavior segment, or timestamp gap"},
        "training_data_audit": cnr_audit,
        "models": metrics,
        "scoring_policy": {"primary": "window-pooled 3-class macro-F1",
            "animal_summary": "per-calf macro-F1 for classes with at least one labeled window for that calf; predictions to unsupported classes still reduce recall for the true class",
            "baseline": "constant training-cohort majority class, selected from CNR train labels only",
            "intervals": "2,000 bootstrap resamples of calf clusters, reported separately for pooled macro-F1 and the mean supported-class per-calf macro-F1",
            "seed": SEED},
        "code_sha256": source_hashes,
        "artifact_sha256": artifacts,
        "compute": {"training_runtime": "Kaggle GPU required", "torch_optimizer_steps": updates,
            "gpu_name": str(torch.cuda.get_device_name(0))},
        "limitations": ["one external farm/cohort only", "pre-weaned dairy calves vs adult cattle; breed, age, sensor position and frequency shift together",
            "ActBeCalf annotations are 27.4 h selected to include active bouts, so scores do not estimate natural prevalence/time budgets",
            "five-to-three-class remapping collapses every non-feeding/non-rumination class to other",
            "classification measures perception from accelerometers, not autonomous agency, physiology or causal response to management",
            "candidate settings are frozen from prior CNR runs and prior model-family selection; result is exploratory, not independent model selection"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: {"macro_f1": result["macro_f1_window_pooled"],
                            "weighted_f1": result["weighted_f1_window_pooled"],
                            "predicted_class_counts": result["predicted_class_counts"],
                            "animal_mean": result["macro_f1_animal_mean"],
                            "animal_sd": result["macro_f1_animal_sd"],
                            "pooled_ci": result["macro_f1_pooled_animal_cluster_bootstrap_95ci"],
                            "animal_mean_ci": result["macro_f1_animal_mean_cluster_bootstrap_95ci"]}
                      for name, result in metrics.items()}, indent=2))

BASE_TRAIN_SOURCE_SHA256 = '16b021f87e1f5e08be73fa0a49cffabd25be3b3e1efe0d732a6c1e1fb8cd41e0'
ACTBECALF_LOADER_SOURCE_SHA256 = '09d4d41c327fe9a2305b8f97d07338d0a006a9073405f20c73c4206a71439fe4'
TRANSFER_RUNNER_SOURCE_SHA256 = '72d1a079a3aee715974199b7d8536fcb9a966202b320fbcb1da0ec16e914a8d1'

if __name__ == "__main__":
    main()
