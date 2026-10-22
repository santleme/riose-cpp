#!/usr/bin/env python3
"""Frozen-protocol CPU ExtraTrees iteration for Nellore next-visit gaps.

Hyperparameters are selected only on the animal-disjoint validation partition.
That single frozen configuration is evaluated once on the animal holdout and
once on the predeclared chronological holdout. This predicts sensor intervals;
it is not a biological model or animal policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from train_gap_model import (
    SEED, baseline_predictions, build_samples, evaluate, make_splits, matrices,
    read_source, resolve_data, sha256, split_rows,
)

GRID = [
    {"n_estimators": 80, "max_features": feature_fraction, "min_samples_leaf": leaf,
     "max_depth": None, "criterion": "squared_error"}
    for leaf in (4, 12) for feature_fraction in (0.7, 1.0)
]


def fit(x: np.ndarray, y: np.ndarray, params: dict, seed: int) -> ExtraTreesRegressor:
    model = ExtraTreesRegressor(
        **params, random_state=seed, n_jobs=-1, bootstrap=False,
    )
    model.fit(x, np.log1p(y))
    return model


def predict(model: ExtraTreesRegressor, x: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(model.predict(x), 0.0, math.log1p(24.0)))


def score_rows(model, x, y, rows):
    return evaluate(y, predict(model, x), rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("training/visits/runs/extratrees-v1"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events, source = read_source(resolve_data(args.data))
    samples, audit = build_samples(events)
    animal_split, time_split = make_splits(samples, args.seed)
    animal_parts, time_parts, purged = split_rows(samples, animal_split, time_split)

    train_rows, x_train, y_train = matrices(samples, animal_parts["train"])
    val_rows, x_val, y_val = matrices(samples, animal_parts["validation"])
    val_baselines, baseline_fit = baseline_predictions(train_rows, y_train, val_rows)
    val_baseline_scores = {key: evaluate(y_val, pred, val_rows) for key, pred in val_baselines.items()}
    selected_baseline = "training_hourly_median"  # frozen strong reference, not selected on test

    # Preregistered small grid. The sole selection metric is macro-animal MAE
    # on held-out validation animals; no test data is read in this loop.
    selection = []
    for index, params in enumerate(GRID):
        model = fit(x_train, y_train, params, args.seed + index)
        metrics = score_rows(model, x_val, y_val, val_rows)
        selection.append({
            "params": params,
            "validation_metrics": metrics,
            "seed": args.seed + index,
        })
    selected = min(selection, key=lambda item: item["validation_metrics"]["macro_animal_mae_hours"])
    frozen_params = selected["params"]
    # Freeze the grid winner and fit once per evaluation view. The validation
    # animal group is not added to training, preserving a simple fixed split.
    outputs = {}
    for name, parts in (("animal_holdout", animal_parts), ("future_time_holdout", time_parts)):
        tr, xtr, ytr = matrices(samples, parts["train"])
        va, xva, yva = matrices(samples, parts["validation"])
        te, xte, yte = matrices(samples, parts["test"])
        if min(len(tr), len(va), len(te)) == 0:
            raise ValueError(f"Empty {name} partition")
        # Test outcomes are first accessed here, after the single parameter
        # configuration and comparison baseline have been frozen.
        final_model = fit(xtr, ytr, frozen_params, args.seed + 1000 + (name == "future_time_holdout"))
        val_metrics = score_rows(final_model, xva, yva, va)
        test_metrics = score_rows(final_model, xte, yte, te)
        test_baselines, fitted_baseline = baseline_predictions(tr, ytr, te)
        baseline_metrics = {key: evaluate(yte, value, te) for key, value in test_baselines.items()}
        metrics = {
            "extra_trees": test_metrics,
            **baseline_metrics,
        }
        delta = np.asarray([
            metrics["extra_trees"]["per_animal"][animal]["mae_hours"]
            - metrics[selected_baseline]["per_animal"][animal]["mae_hours"]
            for animal in sorted(metrics["extra_trees"]["per_animal"])
        ])
        rng = np.random.default_rng(args.seed + 202 + (name == "future_time_holdout"))
        draws = np.asarray([
            np.mean(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(2000)
        ])
        checkpoint = args.output_dir / f"extratrees_{name}.joblib"
        from joblib import dump
        dump({
            "model": final_model,
            "features": __import__("train_gap_model").FEATURES,
            "target_transform": "log1p(hours)",
            "selected_params": frozen_params,
            "not_a_policy": True,
        }, checkpoint, compress=3)
        outputs[name] = {
            "rows": {"train": len(tr), "validation": len(va), "test": len(te)},
            "validation_metrics_fixed_model": val_metrics,
            "test_metrics": metrics,
            "comparison_baseline_frozen_before_test": selected_baseline,
            "delta_macro_animal_mae_hours_extra_trees_minus_hourly_median": float(delta.mean()),
            "paired_animal_bootstrap_95_ci_delta": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint.read_bytes()),
            "baseline_fit_train_only": fitted_baseline,
        }

    animal_ids = {key: sorted(animal for animal, split in animal_split.items() if split == key)
                  for key in ("train", "validation", "test")}
    time_dates = {key: sorted(date.isoformat() for date, split in time_split.items() if split == key)
                  for key in ("train", "validation", "test")}
    manifest = {
        "experiment": "nellore_next_visit_gap_extratrees_v1",
        "dataset": source,
        "data_audit": audit,
        "protocol": {
            "target": "next event IN minus current event OUT, in hours; negative overlaps excluded; zero retained; terminal events right-censored",
            "feature_boundary": "same past-only event features as the MLP; no animal/equipment IDs or future data",
            "animal_split_ids": animal_ids,
            "chronological_split_dates": time_dates,
            "chronological_cross_partition_targets_purged": purged,
            "selection": "ExtraTrees grid selected once by animal-disjoint validation macro-animal MAE in hours; temporal validation does not select or alter parameters; test partitions opened only after freeze",
            "grid": GRID,
            "selection_results_animal_validation_only": selection,
            "selected_params": frozen_params,
            "comparison_baseline": selected_baseline,
            "baselines_reported": ["training_global_median", "training_hourly_median", "persistence_previous_gap"],
            "uncertainty": "paired bootstrap by animal, 2000 resamples; negative delta favours ExtraTrees",
            "interpretation": "Holdout metrics are descriptive and exploratory/post-selection; no candidate acceptance claim is available from these data.",
            "independent_evaluation_required": "Pre-register candidate families and protocol, then evaluate on a new farm or cohort.",
            "policy_exported": False,
            "seed": args.seed,
        },
        "results": outputs,
        "evaluation_status": "exploratory_post_selection_no_acceptance_claim",
        "runtime": {"device": "CPU", "scikit_learn": __import__("sklearn").__version__, "numpy": np.__version__},
        "license_note": "Dataset source is CC-BY-NC-SA-4.0. No dataset content is bundled. Checkpoint redistribution/commercial use requires attribution and license review.",
        "manifest_hashes": {},
    }
    path = args.output_dir / "evaluation.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"evaluation_status": manifest["evaluation_status"], "selected_params": frozen_params,
                      "results": {name: {"tree": item["test_metrics"]["extra_trees"],
                                          "hourly": item["test_metrics"][selected_baseline],
                                          "persistence": item["test_metrics"]["persistence_previous_gap"],
                                          "delta_ci": item["paired_animal_bootstrap_95_ci_delta"]}
                                  for name, item in outputs.items()},
                      "evaluation": str(path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
