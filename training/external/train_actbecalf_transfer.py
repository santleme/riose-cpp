#!/usr/bin/env python3
"""Frozen CNR-to-ActBeCalf cross-cohort activity transfer challenge.

The test cohort is not used for model selection, fitting, normalization, or
thresholds. Training data and feature contract come from the Kaggle Cow Nose
Ring study; ActBeCalf is used only as a separately licensed external test.
"""
from __future__ import annotations

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


if __name__ == "__main__":
    main()
