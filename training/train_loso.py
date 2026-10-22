#!/usr/bin/env python3
"""Exploratory leave-one-animal-out evaluation of the frozen v5 models.

No model family or hyperparameter is selected from an outer fold. The data is
used only for cross-animal description; this does not create an independent
acceptance cohort because the v5 models and original holdouts were inspected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

try:
    from . import train
    from .loso_protocol import leave_one_animal_out
except ImportError:  # Kaggle script kernel runs this file from the training directory.
    import train
    from loso_protocol import leave_one_animal_out


SEED = 23


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_extra_trees(seed: int) -> ExtraTreesClassifier:
    """Exact v5 ExtraTrees settings; intentionally no outer-fold tuning."""
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        max_features=0.8,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
    )


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    labels = sorted(train.LABELS)
    return {
        "macro_f1": float(f1_score(y_true, prediction, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, prediction, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            prediction,
            labels=labels,
            target_names=[train.LABELS[label] for label in labels],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=labels).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/loso-output"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    cuda_required = args.require_cuda or Path("/kaggle/input").is_dir()
    if cuda_required and not train._cuda_available():
        raise RuntimeError("CUDA is required for this LOSO rerun but is unavailable")
    data_path = args.data or train.kaggle_mounted_data() or train.fetch_kaggle_data(Path(".cache/cow-nose-ring"))
    data_sha256 = train.verify_dataset_hash(data_path)
    frame = pd.read_csv(data_path)
    features, labels, animals, extraction = train._extract_windows(frame)
    if len(features) != extraction["accepted_windows"]:
        raise AssertionError("Feature extraction count does not match parser diagnostics")

    all_labels = sorted(train.LABELS)
    folds = []
    for fold_index, split in enumerate(leave_one_animal_out(animals.tolist())):
        held_out = str(split["held_out_animal"])
        train_indices = np.asarray(split["train_indices"], dtype=np.int64)
        test_indices = np.asarray(split["test_indices"], dtype=np.int64)
        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]
        train_animals = sorted(set(animals[train_indices].tolist()))
        if held_out in train_animals or set(animals[test_indices].tolist()) != {held_out}:
            raise AssertionError("Outer fold contains animal leakage")
        class_counts = {train.LABELS[label]: int(np.sum(y_train == label)) for label in all_labels}
        absent_labels = [train.LABELS[label] for label in all_labels if class_counts[train.LABELS[label]] == 0]
        support = {train.LABELS[label]: int(np.sum(y_test == label)) for label in all_labels}

        tree = fixed_extra_trees(args.seed)
        tree.fit(x_train, y_train)
        tree_prediction = tree.predict(x_test)

        _, torch_result, _ = train._train_torch_mlp(
            x_train, y_train, x_test, y_test, args.seed,
        )
        torch_result.pop("validation_macro_f1", None)
        folds.append({
            "fold": fold_index + 1,
            "held_out_animal": held_out,
            "train_animal_count": len(train_animals),
            "train_window_count": int(len(train_indices)),
            "test_window_count": int(len(test_indices)),
            "training_class_counts": class_counts,
            "labels_absent_from_training": absent_labels,
            "test_class_support": support,
            "models": {
                "extra_trees_v5_frozen": metrics(y_test, tree_prediction),
                "torch_mlp_cuda_v5_frozen": torch_result,
            },
        })
        print(json.dumps({
            "fold": fold_index + 1,
            "held_out_animal": held_out,
            "extra_trees_macro_f1": folds[-1]["models"]["extra_trees_v5_frozen"]["macro_f1"],
            "torch_mlp_macro_f1": torch_result["macro_f1"],
            "labels_absent_from_training": absent_labels,
        }), flush=True)

    aggregate = {}
    for model_name in ("extra_trees_v5_frozen", "torch_mlp_cuda_v5_frozen"):
        fold_scores = [fold["models"][model_name]["macro_f1"] for fold in folds]
        by_class = {}
        for label in all_labels:
            scores = [fold["models"][model_name]["classification_report"][train.LABELS[label]]["f1-score"] for fold in folds]
            supports = [fold["test_class_support"][train.LABELS[label]] for fold in folds]
            by_class[train.LABELS[label]] = {
                "mean_fold_f1": float(np.mean(scores)),
                "sd_fold_f1": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "folds_with_support": int(sum(value > 0 for value in supports)),
                "total_windows": int(sum(supports)),
            }
        aggregate[model_name] = {
            "macro_f1_mean_across_animals": float(np.mean(fold_scores)),
            "macro_f1_sd_across_animals": float(np.std(fold_scores, ddof=1)),
            "per_class": by_class,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_script = Path(__file__)
    base_script = Path(train.__file__)
    manifest = {
        "experiment": "cow_activity_leave_one_animal_out_frozen_v5",
        "status": "exploratory_post_selection_no_acceptance_claim",
        "source": train.DATASET,
        "license": train.DATASET_LICENSE,
        "source_csv_sha256": data_sha256,
        "source_hash_verified": True,
        "window_extraction": extraction,
        "split_protocol": "Seven outer folds; one complete animal held out; all windows from each animal remain in exactly one fold. No outer test metric or fold label support is used for candidate or hyperparameter selection.",
        "frozen_configs": {
            "seed": args.seed,
            "extra_trees_v5_frozen": {"n_estimators": 300, "max_depth": None, "class_weight": "balanced", "max_features": 0.8, "min_samples_leaf": 2},
            "torch_mlp_cuda_v5_frozen": {"hidden_sizes": [96, 48], "epochs": 45, "batch_size": 1024, "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.001},
            "model_selection": "none across outer folds; both v5 candidates run on every fold",
        },
        "runtime": {"device": "cuda", "cuda_required": bool(cuda_required), "cuda_available": train._cuda_available(), "gpu_name": train._gpu_name()},
        "folds": folds,
        "aggregate": aggregate,
        "limitations": [
            "Post-selection/exploratory: v5 candidate families and earlier grouped holdouts were already inspected.",
            "All seven folds come from one small sensor cohort and one farm domain; this is not an independent-farm validation.",
            "Rare class support varies by fold; classes absent from training are explicitly listed and their F1 is structurally zero.",
            "Activities are observed labels, not latent need, physiological state, or causal response.",
        ],
        "source_script_sha256": sha256_file(source_script),
        "base_training_script_sha256": sha256_file(base_script),
    }
    output_path = args.output_dir / "loso-evaluation.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "aggregate": aggregate,
                      "output_sha256": sha256_file(output_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
