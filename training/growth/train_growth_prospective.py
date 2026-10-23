#!/usr/bin/env python3
"""Prospective rolling-origin ADG benchmark using only information available at interval start.

This is a predictive benchmark on one Nellore herd/paddock, not a mechanistic
growth model or an estimate of supplement/climate causal effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_DOI = "10.17632/3wdjvxmvsp.7"
EXPECTED_SHA256 = "300e274f497d6a19f4f2f9df69cf6dcb0dff03aac8c59cee799bd45404769e24"
SEED = 20261023
FEATURES = ["START_WEIGHT", "AMOUNT_DAYS", "DOY_SIN", "DOY_COS", "PREVIOUS_ADG"]
TARGET = "ADG"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_data_path(path: Path, *, input_root: Path = Path("/kaggle/input")) -> Path:
    """Resolve Kaggle's mount by pinned filename without assuming its folder slug."""
    if path.is_file():
        return path
    if input_root.is_dir():
        matches = sorted(input_root.glob("**/Complete_Dataset.csv"))
        if len(matches) == 1:
            return matches[0]
        mounted_files = sorted(str(candidate.relative_to(input_root))
                               for candidate in input_root.glob("**/*") if candidate.is_file())
        if len(matches) > 1:
            raise FileNotFoundError(f"Expected one pinned CSV under {input_root}; found {matches}")
        raise FileNotFoundError(f"No Complete_Dataset.csv under {input_root}; mounted files: {mounted_files}")
    raise FileNotFoundError(f"Input file {path} does not exist and Kaggle mount {input_root} is unavailable")


def prepare_frame(path: Path, *, verify_hash: bool = True) -> tuple[pd.DataFrame, dict]:
    digest = sha256_file(path)
    if verify_hash and digest != EXPECTED_SHA256:
        raise ValueError(f"Pinned UFMS CSV SHA-256 mismatch: {digest}")
    raw = pd.read_csv(path)
    required = {"ANIMAL", "PERIOD", "START_DATE", "FINAL_DATE", "AMOUNT_DAYS", "START_WEIGHT", "FINAL_WEIGHT", "ADG"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required UFMS columns: {missing}")
    frame = raw.copy()
    for column in ("ANIMAL", "PERIOD", "AMOUNT_DAYS", "START_WEIGHT", TARGET):
        frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    frame["START_DATE"] = pd.to_datetime(frame["START_DATE"], errors="coerce")
    frame["FINAL_DATE"] = pd.to_datetime(frame["FINAL_DATE"], errors="coerce")
    frame = frame.dropna(subset=["ANIMAL", "PERIOD", "AMOUNT_DAYS", "START_WEIGHT", TARGET, "START_DATE", "FINAL_DATE"]).copy()
    if frame.empty:
        raise ValueError("No complete animal-period records remain")
    frame[["ANIMAL", "PERIOD"]] = frame[["ANIMAL", "PERIOD"]].astype(int)
    duration = (frame["FINAL_DATE"] - frame["START_DATE"]).dt.days
    if not np.allclose(duration, frame["AMOUNT_DAYS"], atol=1):
        raise ValueError("AMOUNT_DAYS does not agree with the start/end dates")
    derived_adg = (pd.to_numeric(raw["FINAL_WEIGHT"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
                   - pd.to_numeric(raw["START_WEIGHT"].astype(str).str.replace(",", ".", regex=False), errors="coerce"))
    target_from_weights = derived_adg / pd.to_numeric(raw["AMOUNT_DAYS"].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    correlation = float(np.corrcoef(frame[TARGET], target_from_weights.loc[frame.index])[0, 1])
    frame["DOY_SIN"] = np.sin(2 * np.pi * frame["START_DATE"].dt.day_of_year / 365.25)
    frame["DOY_COS"] = np.cos(2 * np.pi * frame["START_DATE"].dt.day_of_year / 365.25)
    frame = frame.sort_values(["ANIMAL", "START_DATE", "PERIOD"], kind="stable").reset_index(drop=True)
    frame["PREVIOUS_ADG"] = frame.groupby("ANIMAL", sort=False)[TARGET].shift(1)
    frame["PREVIOUS_FINAL_DATE"] = frame.groupby("ANIMAL", sort=False)["FINAL_DATE"].shift(1)
    frame["PREVIOUS_PERIOD"] = frame.groupby("ANIMAL", sort=False)["PERIOD"].shift(1)
    contiguous = (
        frame["PREVIOUS_ADG"].notna()
        & frame["PREVIOUS_PERIOD"].eq(frame["PERIOD"] - 1)
        & frame["PREVIOUS_FINAL_DATE"].eq(frame["START_DATE"])
    )
    frame.loc[~contiguous, "PREVIOUS_ADG"] = np.nan
    audit = {
        "sha256": digest,
        "rows": int(len(frame)),
        "animals": int(frame["ANIMAL"].nunique()),
        "periods": sorted(map(int, frame["PERIOD"].unique())),
        "target_adg_vs_weight_derived_correlation": correlation,
        "rows_with_valid_previous_period_adg": int(frame["PREVIOUS_ADG"].notna().sum()),
        "features": FEATURES,
        "excluded_interval_or_post_outcome_columns": [
            "FINAL_WEIGHT", "weather summaries", "vegetation indices", "realized supplement (ADS)",
            "feeder attendance and duration", "all visit totals",
        ],
        "lag_rule": "previous ADG is allowed only when the same animal's immediately preceding period is present and its FINAL_DATE equals the current START_DATE",
    }
    return frame, audit


def model_candidates() -> dict[str, object]:
    return {
        "training_mean": DummyRegressor(strategy="mean"),
        "ridge_start_state_and_previous_adg": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


def period_metrics(rows: list[dict]) -> dict:
    y = np.asarray([row["observed"] for row in rows], dtype=float)
    p = np.asarray([row["prediction"] for row in rows], dtype=float)
    return {
        "n": len(rows),
        "mae_kg_per_day": float(mean_absolute_error(y, p)),
        "rmse_kg_per_day": float(mean_squared_error(y, p) ** 0.5),
        "bias_kg_per_day": float(np.mean(p - y)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
    }


def paired_period_bootstrap(prediction_rows: list[dict], baseline_rows: list[dict], seed: int, replicates: int = 10000) -> dict:
    prediction_by_period = {period: values for period, values in _mae_by_period(prediction_rows).items()}
    baseline_by_period = {period: values for period, values in _mae_by_period(baseline_rows).items()}
    periods = sorted(set(prediction_by_period) & set(baseline_by_period))
    differences = np.asarray([prediction_by_period[p] - baseline_by_period[p] for p in periods], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(differences, size=(replicates, len(differences)), replace=True).mean(axis=1)
    return {
        "test_periods_n": len(periods),
        "periods": periods,
        "mean_period_mae_difference_model_minus_baseline": float(differences.mean()),
        "bootstrap_95ci_equal_weight_periods": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
        "seed": seed,
        "replicates": replicates,
        "effective_clusters_note": "Period clusters share a single herd, paddock, and management context; this interval is exploratory, not farm-level uncertainty.",
    }


def _mae_by_period(rows: list[dict]) -> dict[int, float]:
    result = {}
    for period in sorted({row["period"] for row in rows}):
        selected = [row for row in rows if row["period"] == period]
        result[period] = float(mean_absolute_error(
            [row["observed"] for row in selected], [row["prediction"] for row in selected]))
    return result


def rolling_origin(frame: pd.DataFrame, *, first_test_period: int = 7) -> tuple[dict, object]:
    usable = frame.dropna(subset=FEATURES + [TARGET]).copy()
    periods = sorted(map(int, usable["PERIOD"].unique()))
    if len(periods) < 8 or first_test_period not in periods:
        raise ValueError("Need at least eight usable periods and a valid first test period")
    models = model_candidates()
    predictions: dict[str, list[dict]] = {name: [] for name in models}
    predictions["previous_adg_persistence"] = []
    split_records = []
    for test_period in [period for period in periods if period >= first_test_period]:
        training = usable.loc[usable["PERIOD"] < test_period]
        test = usable.loc[usable["PERIOD"] == test_period]
        if training.empty or test.empty or training["PERIOD"].max() >= test_period:
            raise ValueError(f"Invalid temporal split at period {test_period}")
        x_train, y_train = training[FEATURES], training[TARGET].to_numpy(dtype=float)
        x_test = test[FEATURES]
        actual = test[TARGET].to_numpy(dtype=float)
        for name, prototype in models.items():
            candidate = __import__("sklearn.base", fromlist=["clone"]).clone(prototype)
            candidate.fit(x_train, y_train)
            predicted = candidate.predict(x_test)
            predictions[name].extend({
                "animal": int(animal), "period": int(test_period), "observed": float(obs), "prediction": float(pred)
            } for animal, obs, pred in zip(test["ANIMAL"], actual, predicted))
            if name == "ridge_start_state_and_previous_adg" and test_period == periods[-1]:
                final_model = candidate
        predictions["previous_adg_persistence"].extend({
            "animal": int(animal), "period": int(test_period), "observed": float(obs), "prediction": float(pred)
        } for animal, obs, pred in zip(test["ANIMAL"], actual, test["PREVIOUS_ADG"]))
        split_records.append({
            "train_periods": sorted(map(int, training["PERIOD"].unique())),
            "test_period": int(test_period),
            "train_rows": int(len(training)),
            "test_rows": int(len(test)),
            "test_animals": sorted(map(int, test["ANIMAL"].unique())),
        })
    if "final_model" not in locals():
        raise ValueError("Final rolling-origin model was not fitted")
    metrics = {
        name: {
            **period_metrics(rows),
            "period_mae_kg_per_day": {str(p): mae for p, mae in _mae_by_period(rows).items()},
        }
        for name, rows in predictions.items()
    }
    metrics["ridge_start_state_and_previous_adg"]["paired_period_bootstrap_vs_baselines"] = {
        baseline_name: paired_period_bootstrap(predictions["ridge_start_state_and_previous_adg"],
                                                predictions[baseline_name], SEED)
        for baseline_name in ("training_mean", "previous_adg_persistence")
    }
    return {"splits": split_records, "models": metrics}, final_model


def run(data_path: Path, output_dir: Path, *, verify_hash: bool = True) -> dict:
    frame, audit = prepare_frame(data_path, verify_hash=verify_hash)
    results, model = rolling_origin(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "prospective_adg_ridge.joblib", compress=3)
    manifest = {
        "experiment": "nelore_ufms_prospective_adg_rolling_origin_v1",
        "status": "complete_exploratory_single_herd_benchmark",
        "source": {"name": "UFMS Nelore supplementation dataset", "doi": DATASET_DOI,
                   "version": 7, "license": "CC BY 4.0", "sha256": audit["sha256"]},
        "target": {"name": TARGET, "unit": "kg/day", "definition": "published mean ADG during each weighing interval"},
        "data_audit": audit,
        "design": {
            "split": "expanding-window rolling origin by whole experimental period; each test period is later than every training period",
            "first_test_period": results["splits"][0]["test_period"],
            "test_periods": [split["test_period"] for split in results["splits"]],
            "feature_information_cutoff": "interval start; previous ADG only from the immediately prior period ending exactly at current start date",
            "candidate_selection": "none; fixed Ridge alpha=10 is compared to global training mean and prior-period persistence",
            "repeated_animals": "animals recur across periods; uncertainty is summarized by the period-level paired bootstrap, not by animal-row bootstrap",
            "seed": SEED,
        },
        "results": results,
        "limitations": [
            "one herd, paddock, and 15 period contexts; no farm-level external validation",
            "the response remains an observed interval outcome derived almost exactly from measured weight change",
            "the dataset is observational and does not identify a causal supplement or weather effect",
            "interval duration is treated as the scheduled horizon known at interval start; this must be confirmed against operational deployment",
            "no weather, pasture index, realized supplement, feeder attendance, final weight, or post-outcome measurement enters the model",
            "results are predictive baselines, not a mechanistic growth or energy-balance equation",
        ],
    }
    manifest["artifact_sha256"] = {"prospective_adg_ridge.joblib": sha256_file(output_dir / "prospective_adg_ridge.joblib")}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("/kaggle/input/campo-vivo-nelore-ufms-v7/Complete_Dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/prospective-adg"))
    parser.add_argument("--allow-unpinned-fixture", action="store_true")
    args = parser.parse_args()
    data_path = resolve_data_path(args.data)
    result = run(data_path, args.output_dir, verify_hash=not args.allow_unpinned_fixture)
    print(json.dumps({"experiment": result["experiment"], "results": result["results"]["models"]}, indent=2))


if __name__ == "__main__":
    main()
