#!/usr/bin/env python3
"""Frozen three-class temporal activity experiment; not an animal decision policy."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import train

SEED = 20261022
COARSE_LABELS = {0: "feeding", 1: "rumination", 2: "other"}
SOURCE_TO_COARSE = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2}
EPOCHS = 35
BATCH_SIZE = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def torch_device(require_cuda: bool):
    import torch
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a visible CUDA GPU")
    return torch, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_tcn(torch, channels: int = 32):
    from torch import nn
    return nn.Sequential(
        nn.Conv1d(3, channels, kernel_size=5, padding=2), nn.GELU(),
        nn.Conv1d(channels, channels, kernel_size=5, padding=4, dilation=2), nn.GELU(),
        nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2), nn.GELU(),
        nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, 3),
    )


def fit_tcn(x_train, y_train, x_test, torch, device):
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = make_tcn(torch).to(device)
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    scale = x_train.std(axis=(0, 1), keepdims=True).clip(min=1e-5)
    train_x = torch.as_tensor(((x_train - mean) / scale).transpose(0, 2, 1), dtype=torch.float32, device=device)
    test_x = torch.as_tensor(((x_test - mean) / scale).transpose(0, 2, 1), dtype=torch.float32, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    counts = torch.bincount(train_y, minlength=3).float()
    weights = (counts.sum() / (3 * counts.clamp_min(1))).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    model.train()
    optimizer_steps = 0
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(train_y), device=device)
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
    model.eval()
    with torch.no_grad():
        predictions = model(test_x).argmax(dim=1).cpu().numpy()
    return predictions, model, mean.reshape(-1), scale.reshape(-1), optimizer_steps


def summarize(y_true, y_pred):
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=[COARSE_LABELS[i] for i in range(3)],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/outputs/activity-tcn-loso"))
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    require_cuda = args.require_cuda or Path("/kaggle/input").is_dir()
    torch, device = torch_device(require_cuda)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    data_path = args.data or train.kaggle_mounted_data()
    if data_path is None:
        data_path = train.fetch_kaggle_data(Path(".cache/cow-nose-ring"))
    data_hash = train.verify_dataset_hash(data_path)
    frame = train.pd.read_csv(data_path)
    stats_x, source_y, animals, diagnostics, raw_x = train._extract_windows(frame, include_raw=True)
    y = np.asarray([SOURCE_TO_COARSE[int(label)] for label in source_y], dtype=np.int64)
    folds = []
    ids = sorted(set(animals.tolist()))
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_files = []
    for fold_index, animal in enumerate(ids):
        test_mask = animals == animal
        x_train, x_test = stats_x[~test_mask], stats_x[test_mask]
        y_train, y_test = y[~test_mask], y[test_mask]
        majority = int(np.bincount(y_train, minlength=3).argmax())
        majority_predictions = np.full(len(y_test), majority, dtype=np.int64)
        forest = ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", max_features=0.8,
            min_samples_leaf=2, n_jobs=-1, random_state=SEED,
        )
        forest.fit(x_train, y_train)
        forest_predictions = forest.predict(x_test)
        tcn_predictions, tcn_model, mean, scale, optimizer_steps = fit_tcn(
            raw_x[~test_mask], y_train, raw_x[test_mask], torch, device,
        )
        checkpoint_path = checkpoint_dir / f"held-out-cow-{animal}.pt"
        torch.save({
            "schema_version": 1,
            "state_dict": {key: value.detach().cpu() for key, value in tcn_model.state_dict().items()},
            "normalization_mean": torch.as_tensor(mean, dtype=torch.float32),
            "normalization_scale": torch.as_tensor(scale, dtype=torch.float32),
            "class_map": COARSE_LABELS,
            "window_samples": train.WINDOW_SAMPLES,
            "sampling_hz": train.SAMPLING_HZ,
            "seed": SEED,
            "train_animals": sorted(set(animals[~test_mask].tolist())),
            "held_out_animal": animal,
            "source_csv_sha256": data_hash,
            "candidate": "temporal_cnn_frozen",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer_steps": optimizer_steps,
        }, checkpoint_path)
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if restored["state_dict"].keys() != tcn_model.state_dict().keys():
            raise RuntimeError(f"Checkpoint state mismatch for held-out cow {animal}")
        if tuple(restored["normalization_mean"].shape) != (3,) or tuple(restored["normalization_scale"].shape) != (3,):
            raise RuntimeError(f"Checkpoint normalization mismatch for held-out cow {animal}")
        checkpoint_files.append({"path": str(checkpoint_path.relative_to(args.output_dir)),
                                 "sha256": sha256(checkpoint_path), "optimizer_steps": optimizer_steps})
        fold = {
            "fold": fold_index + 1,
            "held_out_animal": animal,
            "train_animals": sorted(set(animals[~test_mask].tolist())),
            "test_window_count": int(test_mask.sum()),
            "test_class_counts": {COARSE_LABELS[i]: int(np.sum(y_test == i)) for i in range(3)},
            "models": {
                "train_majority": summarize(y_test, majority_predictions),
                "extra_trees_frozen": summarize(y_test, forest_predictions),
                "temporal_cnn_frozen": summarize(y_test, tcn_predictions),
            },
        }
        folds.append(fold)
        print(json.dumps({"fold": fold_index + 1, "animal": animal,
                          "extra_trees_macro_f1": fold["models"]["extra_trees_frozen"]["macro_f1"],
                          "temporal_cnn_macro_f1": fold["models"]["temporal_cnn_frozen"]["macro_f1"]}), flush=True)

    aggregate = {}
    for model_name in ("train_majority", "extra_trees_frozen", "temporal_cnn_frozen"):
        scores = [fold["models"][model_name]["macro_f1"] for fold in folds]
        aggregate[model_name] = {
            "macro_f1_mean_by_animal": float(np.mean(scores)),
            "macro_f1_sd_by_animal": float(np.std(scores, ddof=1)),
            "per_animal_macro_f1": scores,
            "per_class_f1_mean_by_animal": {
                label: float(np.mean([fold["models"][model_name]["classification_report"][label]["f1-score"] for fold in folds]))
                for label in COARSE_LABELS.values()
            },
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment": "cnr_three_class_temporal_cnn_loso_v1",
        "status": "completed_metrics_are_exploratory_and_post_selection",
        "dataset": train.DATASET,
        "dataset_license": train.DATASET_LICENSE,
        "source_csv_sha256": data_hash,
        "rows": int(len(frame)),
        "accepted_windows": int(len(y)),
        "window_seconds": train.WINDOW_SAMPLES / train.SAMPLING_HZ,
        "window_samples": train.WINDOW_SAMPLES,
        "coarse_label_mapping": {"0": "feeding", "1": "rumination", "2|3|4": "other"},
        "extraction_diagnostics": diagnostics,
        "split": "leave one complete cow out; no windows from a held-out animal enter training; no model selection across folds",
        "candidates": {
            "extra_trees_frozen": {"trees": 300, "class_weight": "balanced", "max_features": 0.8, "min_samples_leaf": 2},
            "temporal_cnn_frozen": {"conv_channels": 32, "kernel_sizes": [5, 5, 3], "dilations": [1, 2, 2],
                                    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW", "lr": 0.001,
                                    "class_weight": "inverse_frequency", "seed": SEED},
        },
        "runtime": {"device": str(device), "cuda_available": torch.cuda.is_available(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "folds": folds,
        "aggregate": aggregate,
        "interpretation": "Classifies observed activity from a nose-mounted accelerometer. It does not learn resource choice, motivation, physiology, spatial navigation, social response, or causal effects.",
        "limitations": [
            "Seven cows from one farm and one sensor placement; cow-level holdout is not farm-level external validation.",
            "The dataset's raw CSV uses coarse whole-second timestamps; sample order inside each verified 10 Hz block is assumed.",
            "The three-class mapping follows the source paper's feeding/rumination/other ontology; other merges walking, standing, and lying.",
            "The same cohort and candidate families have been inspected in prior work; all scores are post-selection and exploratory.",
            "This classifier is not an agent policy and cannot be directly integrated into simulated bovine decision-making without simulated/observed accelerometer input and action-context data.",
        ],
        "source_script_sha256": sha256(Path(__file__)),
        "base_training_script_sha256": getattr(train, "BASE_TRAIN_SHA256", sha256(Path(train.__file__))),
        "fold_checkpoints": checkpoint_files,
    }
    output = args.output_dir / "evaluation.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "aggregate": aggregate, "sha256": sha256(output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
