#!/usr/bin/env python3
"""Reproducible Nelore interval-growth benchmark on real herd data.

The target is average daily gain (ADG, kg/day). Inputs are limited to variables
that describe the interval or are known at its start; animal feeder attendance
and duration are deliberately excluded because they are observed during the
outcome interval. This is a predictive, observational benchmark, not a causal
estimate of supplement or weather effects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_ID = "3wdjvxmvsp"
DATASET_VERSION = 7
DATASET_DOI = "10.17632/3wdjvxmvsp.7"
DATASET_API = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
EXPECTED_SHA256 = "300e274f497d6a19f4f2f9df69cf6dcb0dff03aac8c59cee799bd45404769e24"
TARGET = "ADG"
WEATHER_FEATURES = ["TEMP_AVG", "RAIN_AVG", "RAD_SOL_AVG", "EVAPOT", "HUM_REL_AVG"]
PASTURE_FEATURES = ["NDVI_INDEX", "NDWI_INDEX", "EVI_INDEX", "LAI_INDEX", "SAVI_INDEX"]
FEATURES = ["START_WEIGHT", "AMOUNT_DAYS", "ADS", "DOY_SIN", "DOY_COS", *WEATHER_FEATURES, *PASTURE_FEATURES]
EXCLUDED_POST_START = ["SUP_TIME", "SUP_TOTAL", "TOTAL_TIME", "TOTAL_ATTENDANCE", "TOTAL_DAYS"]
SEED = 20261022


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kaggle_csv(data: Path | None, dataset_slug: str | None) -> Path:
    """Resolve data from an attached Kaggle dataset, or download its named CSV."""
    if data is not None:
        return data
    input_dir = Path("/kaggle/input")
    matches = sorted(input_dir.glob("**/Complete_Dataset.csv")) if input_dir.exists() else []
    if matches:
        return matches[0]
    dataset_slug = dataset_slug or os.getenv("CAMPO_VIVO_GROWTH_KAGGLE_DATASET")
    if not dataset_slug:
        raise RuntimeError(
            "No Complete_Dataset.csv attached under /kaggle/input and no Kaggle dataset slug supplied. "
            "The original dataset is not currently mirrored in this repository; attach a verified Kaggle "
            "mirror or pass --kaggle-dataset owner/slug."
        )
    with tempfile.TemporaryDirectory(prefix="campo-vivo-kaggle-growth-") as temporary:
        subprocess.run(["kaggle", "datasets", "download", dataset_slug, "-p", temporary], check=True)
        import zipfile
        archive = next(Path(temporary).glob("*.zip"))
        destination = Path(temporary) / "Complete_Dataset.csv"
        with zipfile.ZipFile(archive) as zipped:
            names = [n for n in zipped.namelist() if Path(n).name.lower() == "complete_dataset.csv"]
            if not names:
                raise ValueError(f"{dataset_slug} archive has no Complete_Dataset.csv")
            destination.write_bytes(zipped.read(names[0]))
        # The temp directory would be deleted at return, so preserve the source in a local cache.
        cache = Path(".cache/nelore-growth")
        cache.mkdir(parents=True, exist_ok=True)
        saved = cache / f"{_sha256(destination)}.csv"
        saved.write_bytes(destination.read_bytes())
        return saved


def download_mendeley(path: Path) -> dict[str, Any]:
    """Fetch the cited version from the canonical public Mendeley Data API."""
    import requests

    response = requests.get(DATASET_API, timeout=30)
    response.raise_for_status()
    metadata = response.json()
    if metadata.get("version") != DATASET_VERSION or metadata.get("doi", {}).get("id") != DATASET_DOI:
        raise RuntimeError("Dataset metadata does not match the pinned DOI/version")
    file_entry = next((item for item in metadata["files"] if item["filename"] == "Complete_Dataset.csv"), None)
    if not file_entry:
        raise RuntimeError("Pinned Mendeley version has no Complete_Dataset.csv")
    expected = file_entry["content_details"]["sha256_hash"]
    if expected != EXPECTED_SHA256:
        raise RuntimeError("Published file hash differs from the pinned manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(file_entry["content_details"]["download_url"], stream=True, timeout=60) as download:
        download.raise_for_status()
        with path.open("wb") as stream:
            for chunk in download.iter_content(1024 * 1024):
                if chunk:
                    stream.write(chunk)
    if _sha256(path) != EXPECTED_SHA256:
        path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded CSV SHA-256 does not match the pinned release")
    license_info = metadata.get("data_licence", {})
    return {
        "source": "Mendeley Data",
        "dataset": metadata["name"],
        "doi": DATASET_DOI,
        "version": DATASET_VERSION,
        "license": license_info.get("short_name"),
        "url": f"https://data.mendeley.com/datasets/{DATASET_ID}/{DATASET_VERSION}",
        "sha256": EXPECTED_SHA256,
    }


def load_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Some numeric fields in the published CSV use decimal commas; comma-separated
    # thousands are not used in the numeric measurement columns.
    raw = pd.read_csv(path, dtype={"SUP_TIME": "string"})
    required = {"START_DATE", "FINAL_DATE", "PERIOD", "ANIMAL", "AMOUNT_DAYS", "START_WEIGHT", "FINAL_WEIGHT", TARGET, "ADS", *WEATHER_FEATURES, *PASTURE_FEATURES}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = raw.copy()
    for column in ["PERIOD", "ANIMAL", "AMOUNT_DAYS", "START_WEIGHT", "FINAL_WEIGHT", TARGET, "ADS", *WEATHER_FEATURES, *PASTURE_FEATURES]:
        frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    frame["START_DATE"] = pd.to_datetime(frame["START_DATE"], errors="coerce")
    frame["FINAL_DATE"] = pd.to_datetime(frame["FINAL_DATE"], errors="coerce")
    frame["DOY_SIN"] = np.sin(2 * np.pi * frame["START_DATE"].dt.day_of_year / 365.25)
    frame["DOY_COS"] = np.cos(2 * np.pi * frame["START_DATE"].dt.day_of_year / 365.25)
    before = len(frame)
    frame = frame.dropna(subset=[TARGET, *FEATURES, "START_DATE", "FINAL_DATE"]).copy()
    frame["ANIMAL"] = frame["ANIMAL"].astype(int)
    frame["PERIOD"] = frame["PERIOD"].astype(int)
    # Basic consistency checks catch parse/version mistakes and prevent future outcomes as inputs.
    if frame.empty or frame["ANIMAL"].nunique() < 10 or frame["PERIOD"].nunique() < 8:
        raise ValueError("Too few animals or periods after parsing; refusing to report a benchmark")
    duration = (frame["FINAL_DATE"] - frame["START_DATE"]).dt.days
    if not np.allclose(duration, frame["AMOUNT_DAYS"], atol=1):
        raise ValueError("AMOUNT_DAYS is inconsistent with date range")
    derived_adg = (frame["FINAL_WEIGHT"] - frame["START_WEIGHT"]) / frame["AMOUNT_DAYS"]
    correlation = float(np.corrcoef(frame[TARGET], derived_adg)[0, 1])
    return frame.sort_values(["PERIOD", "ANIMAL"]).reset_index(drop=True), {
        "rows_received": before,
        "rows_used": len(frame),
        "dropped_rows": before - len(frame),
        "animals": int(frame["ANIMAL"].nunique()),
        "periods": int(frame["PERIOD"].nunique()),
        "target_vs_weight_derived_adg_correlation": correlation,
        "post_start_features_excluded": EXCLUDED_POST_START + ["FINAL_WEIGHT", "ADG"],
        "all_features": FEATURES,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def models(seed: int) -> dict[str, Any]:
    return {
        "train_mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=4, max_features=0.8, random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, l2_regularization=5.0, random_state=seed),
    }


def score(y_true: np.ndarray, prediction: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    per_period = {}
    for period, indices in frame.groupby("PERIOD").groups.items():
        local = np.asarray(list(indices), dtype=int)
        # groups keys are absolute row indices from frame passed to this function.
        per_period[str(int(period))] = {
            "n": int(len(local)),
            "mae_kg_per_day": float(mean_absolute_error(y_true[local], prediction[local])),
        }
    return {
        "n": int(len(y_true)),
        "mae_kg_per_day": float(mean_absolute_error(y_true, prediction)),
        "rmse_kg_per_day": float(mean_squared_error(y_true, prediction) ** 0.5),
        "r2": float(r2_score(y_true, prediction)) if len(y_true) > 1 else None,
        "bias_kg_per_day": float(np.mean(prediction - y_true)),
        "period_metrics": per_period,
    }


def evaluate_split(frame: pd.DataFrame, train_mask: np.ndarray, test_mask: np.ndarray, label: str, seed: int, out: Path) -> dict[str, Any]:
    train = frame.loc[train_mask].reset_index(drop=True)
    test = frame.loc[test_mask].reset_index(drop=True)
    if len(train) < 50 or len(test) < 20:
        raise ValueError(f"Split {label!r} too small (train={len(train)}, test={len(test)})")
    x_train, y_train = train[FEATURES], train[TARGET].to_numpy()
    x_test, y_test = test[FEATURES], test[TARGET].to_numpy()
    candidate_results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for name, model in models(seed).items():
        model.fit(x_train, y_train)
        fitted[name] = model
        prediction = model.predict(x_test)
        candidate_results[name] = score(y_test, prediction, test)
    return {
        "split": label,
        "train_animals": sorted(map(int, train["ANIMAL"].unique())),
        "test_animals": sorted(map(int, test["ANIMAL"].unique())),
        "train_periods": sorted(map(int, train["PERIOD"].unique())),
        "test_periods": sorted(map(int, test["PERIOD"].unique())),
        "models": candidate_results,
        "fitted": fitted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="CSV input (for example /kaggle/input/.../Complete_Dataset.csv)")
    parser.add_argument("--source", choices=["kaggle", "mendeley"], default="kaggle", help="Prefer the verified Kaggle mirror/attachment; use the canonical Mendeley source only when explicitly requested.")
    parser.add_argument("--kaggle-dataset", help="Verified Kaggle owner/slug; alternatively attach Complete_Dataset.csv under /kaggle/input")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/nelore-growth"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/growth/runs/nelore-v7"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-animal-fraction", type=float, default=0.2)
    args = parser.parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.data:
        data_path = args.data
        data_hash = _sha256(data_path)
        if data_hash != EXPECTED_SHA256:
            raise RuntimeError(
                "Explicit CSV input does not match pinned Mendeley v7 Complete_Dataset.csv; "
                "refusing to train on a file with unverified provenance"
            )
        source = {
            "source": "local or Kaggle-mounted input",
            "dataset": "UFMS Nelore supplementation dataset",
            "doi": DATASET_DOI,
            "version": DATASET_VERSION,
            "license": "CC BY 4.0",
            "sha256": data_hash,
        }
    elif args.source == "kaggle":
        data_path = _kaggle_csv(None, args.kaggle_dataset)
        kaggle_hash = _sha256(data_path)
        if kaggle_hash != EXPECTED_SHA256:
            raise RuntimeError(
                "Kaggle input hash does not match the pinned Mendeley v7 CSV; "
                "refusing to train on an unverified mirror"
            )
        source = {
            "source": "Kaggle dataset",
            "dataset": args.kaggle_dataset or os.getenv("CAMPO_VIVO_GROWTH_KAGGLE_DATASET") or data_path.parent.name,
            "attached_input_directory": data_path.parent.name,
            "doi": DATASET_DOI,
            "license": "CC BY 4.0",
            "version": DATASET_VERSION,
            "sha256": kaggle_hash,
        }
    else:
        data_path = args.cache_dir / "Complete_Dataset.csv"
        if data_path.exists() and _sha256(data_path) == EXPECTED_SHA256:
            source = {"source": "Mendeley Data", "doi": DATASET_DOI, "version": DATASET_VERSION, "license": "CC BY 4.0", "sha256": EXPECTED_SHA256}
        else:
            data_path.unlink(missing_ok=True)
            source = download_mendeley(data_path)
    data_hash = _sha256(data_path)
    frame, audit = load_dataset(data_path)
    periods = sorted(frame["PERIOD"].unique())
    if len(periods) < 12:
        raise ValueError("At least 12 periods are required for train/validation/future-test separation")

    # Prospective time split: train early periods, model-select on middle periods,
    # and evaluate once on the latest periods. Current interval weather/NDVI are
    # observed covariates, so describe this honestly as interval-conditioned nowcast.
    train_end = int(len(periods) * 0.60)
    validation_end = round(len(periods) * 0.73)
    train_periods = periods[:train_end]
    validation_periods = periods[train_end:validation_end]
    test_periods = periods[validation_end:]
    if not validation_periods or len(test_periods) < 2:
        raise ValueError("Not enough periods for a meaningful temporal holdout")
    temporal_train_mask = frame["PERIOD"].isin(train_periods).to_numpy()
    temporal_val_mask = frame["PERIOD"].isin(validation_periods).to_numpy()
    temporal_test_mask = frame["PERIOD"].isin(test_periods).to_numpy()

    # Fixed animal holdout, selected from IDs using the recorded seed. It is an
    # orthogonal generalization check, not hyperparameter selection.
    animals = np.array(sorted(frame["ANIMAL"].unique()))
    shuffled = animals.copy()
    np.random.default_rng(args.seed).shuffle(shuffled)
    n_test = max(1, round(len(animals) * args.test_animal_fraction))
    test_animals = set(map(int, shuffled[:n_test]))
    animal_train_mask = ~frame["ANIMAL"].isin(test_animals).to_numpy()
    animal_test_mask = ~animal_train_mask

    # Model selection occurs on the temporal validation block only.
    val_eval = evaluate_split(frame, temporal_train_mask, temporal_val_mask, "temporal_validation", args.seed, args.output_dir)
    selected = min(val_eval["models"], key=lambda name: val_eval["models"][name]["mae_kg_per_day"])
    time_test_eval = evaluate_split(frame, temporal_train_mask | temporal_val_mask, temporal_test_mask, "future_period_test", args.seed, args.output_dir)
    animal_eval = evaluate_split(frame, animal_train_mask, animal_test_mask, "unseen_animal_test", args.seed, args.output_dir)
    for result in (val_eval, time_test_eval, animal_eval):
        result.pop("fitted", None)

    # Export final selected model fit through the validation horizon; the future
    # test block is never used to fit or choose hyperparameters.
    final_train = frame.loc[temporal_train_mask | temporal_val_mask]
    final_model = models(args.seed)[selected]
    final_model.fit(final_train[FEATURES], final_train[TARGET])
    model_path = args.output_dir / "growth_adg_model.joblib"
    joblib.dump(final_model, model_path, compress=3)
    manifest = {
        "experiment": "nelore_interval_adg_v1",
        "seed": args.seed,
        "dataset": source,
        "csv_sha256": data_hash,
        "dataset_audit": audit,
        "target": {"name": TARGET, "unit": "kg/day", "meaning": "published average daily gain for a weighing interval"},
        "features": FEATURES,
        "feature_policy": "No final weight, ADG-derived fields, feeder attendance, feeder visit duration, or other post-start animal behavior enters X. Weather and remote sensing summarize the full interval, so this is an interval-conditioned nowcast, not a start-of-interval forecast.",
        "splits": {
            "temporal_train_periods": list(map(int, train_periods)),
            "temporal_validation_periods": list(map(int, validation_periods)),
            "future_test_periods": list(map(int, test_periods)),
            "animal_test_ids": sorted(test_animals),
            "animal_train_ids": sorted(set(map(int, animals)) - test_animals),
        },
        "selected_model_by_temporal_validation_mae": selected,
        "results": {
            "temporal_validation": val_eval,
            "future_period_test": time_test_eval,
            "unseen_animal_test": animal_eval,
        },
        "model_artifact": model_path.name,
        "software": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "model_compute_device": "cpu",
            "gpu_estimator_used": False,
        },
        "scientific_limits": [
            "One paddock, one herd, 26 Nelore cattle, around 15 periods; no external farm or breed validation.",
            "All cattle share period-level weather/pasture/supplement context, so effective sample size for environment and treatment effects is closer to 15 periods than 390 animal-period rows.",
            "This observational dataset does not identify causal effects of changing supplement, pasture, or weather.",
            "The deployed model predicts ADG from observed interval-average weather/remote-sensing covariates; those are not available at interval start.",
            "The published ADS column is included as an input but its operational interpretation should be confirmed against the source protocol before it is used for interventions.",
        ],
    }
    (args.output_dir / "evaluation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"selected_model": selected, "future_period_test": time_test_eval["models"][selected], "unseen_animal_test": animal_eval["models"][selected], "output_dir": str(args.output_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
