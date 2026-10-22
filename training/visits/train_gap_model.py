#!/usr/bin/env python3
"""Supervised next-visit interval benchmark for the public Kaggle Nellore log.

This predicts the next arrival gap after a feeder event. It is a narrow temporal
prediction experiment: it is not an animal policy, welfare model, or causal
estimate. No animal or equipment identifier is used as a model feature.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import shlex
import statistics
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

DATASET = "juliavalente/data-from-visits-to-the-trough-of-nellore-cattle"
DATASET_VERSION = 2
LICENSE = "CC-BY-NC-SA-4.0"
DOI = "10.1371/journal.pone.0271337"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET}"
EXPECTED_ARCHIVE_SHA256 = "fd2cfd016503dcff40be2931a6bd04a7185d020c811f6cafa08ec3925d9b5884"
EXPECTED_CSV_SHA256 = "37fbe08c882723b15fe028bcb5810fff52a41b3334ef73593814e0674e8f1408"
EXPECTED_HEADER = ["animal_id", "equip_id", "IN", "OUT"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SEED = 20261022
FEATURES = [
    "log_duration_s", "out_hour_sin", "out_hour_cos", "weekday_sin", "weekday_cos",
    "previous_gap_log_h", "previous_gap_missing", "previous_3_gap_median_log_h",
    "previous_3_gap_missing", "prior_events_6h", "prior_events_24h",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_data(path: Path | None) -> Path:
    if path is not None:
        return path
    root = Path("/kaggle/input")
    if root.exists():
        files = sorted(root.rglob("*.csv"))
        if files:
            return files[0]
    with tempfile.TemporaryDirectory(prefix="campo-visits-kaggle-") as temporary:
        subprocess.run(["kaggle", "datasets", "download", DATASET, "-p", temporary], check=True)
        archive = next(Path(temporary).glob("*.zip"))
        saved_dir = Path(".cache/nellore-visits")
        saved_dir.mkdir(parents=True, exist_ok=True)
        saved = saved_dir / f"{sha256(archive.read_bytes())}.zip"
        saved.write_bytes(archive.read_bytes())
        return saved


def read_source(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    archive_hash = sha256(raw) if zipfile.is_zipfile(path) else None
    if archive_hash:
        if archive_hash != EXPECTED_ARCHIVE_SHA256:
            raise ValueError(f"Kaggle archive hash differs from pinned v2: {archive_hash}")
        with zipfile.ZipFile(path) as zipped:
            if zipped.testzip() is not None:
                raise ValueError("Corrupt Kaggle zip archive")
            candidates = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
            if len(candidates) != 1:
                raise ValueError(f"Expected one CSV in source archive, found {len(candidates)}")
            member = candidates[0]
            csv_bytes = zipped.read(member)
    else:
        member = path.name
        csv_bytes = raw
    csv_hash = sha256(csv_bytes)
    if csv_hash != EXPECTED_CSV_SHA256:
        raise ValueError(f"CSV hash differs from pinned Kaggle v2: {csv_hash}")
    text = csv_bytes.decode("utf-8-sig")
    lines = text.splitlines()
    if not lines or shlex.split(lines[0]) != EXPECTED_HEADER:
        raise ValueError("Unexpected dataset header; expected animal_id, equip_id, IN, OUT")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = shlex.split(line)
        if len(fields) != 4:
            raise ValueError(f"Row {line_number}: expected exactly 4 fields, got {len(fields)}")
        animal, equipment, in_text, out_text = fields
        try:
            start = dt.datetime.strptime(in_text, TIME_FORMAT)
            end = dt.datetime.strptime(out_text, TIME_FORMAT)
        except ValueError as exc:
            raise ValueError(f"Row {line_number}: invalid local IN/OUT timestamp: {exc}") from exc
        duration_s = (end - start).total_seconds()
        if not animal or not equipment or duration_s < 0:
            raise ValueError(f"Row {line_number}: empty ID or OUT preceding IN")
        events.append({"animal": animal, "equipment": equipment, "start": start, "end": end, "duration_s": int(duration_s)})
    if not events:
        raise ValueError("No events parsed")
    return events, {
        "source_url": DATASET_URL,
        "source_version": DATASET_VERSION,
        "license": LICENSE,
        "primary_paper_doi": DOI,
        "archive_filename": path.name,
        "archive_sha256": archive_hash,
        "csv_member": member,
        "csv_sha256": csv_hash,
        "csv_bytes": len(csv_bytes),
        "raw_columns": EXPECTED_HEADER,
    }


def build_samples(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_animal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_animal[event["animal"]].append(event)
    for rows in by_animal.values():
        rows.sort(key=lambda e: (e["start"], e["end"], e["equipment"]))

    samples: list[dict[str, Any]] = []
    negative_gaps = zero_gaps = 0
    overlap_duration_s = []
    right_censored = 0
    left_context_missing = 0
    duplicates = 0
    for animal, rows in by_animal.items():
        seen: set[tuple[Any, ...]] = set()
        prior_valid_gaps: deque[float] = deque(maxlen=3)
        for index, current in enumerate(rows):
            signature = (current["equipment"], current["start"], current["end"])
            if signature in seen:
                duplicates += 1
            seen.add(signature)
            if index == len(rows) - 1:
                right_censored += 1
                continue
            previous_gap_h: float | None = None
            if index > 0:
                previous_gap_h = (current["start"] - rows[index - 1]["end"]).total_seconds() / 3600
                if previous_gap_h >= 0:
                    prior_valid_gaps.append(previous_gap_h)
            else:
                left_context_missing += 1

            next_event = rows[index + 1]
            target_gap_h = (next_event["start"] - current["end"]).total_seconds() / 3600
            if target_gap_h < 0:
                negative_gaps += 1
                overlap_duration_s.append(-target_gap_h * 3600)
                continue
            if target_gap_h == 0:
                zero_gaps += 1

            hour = current["end"].hour + current["end"].minute / 60 + current["end"].second / 3600
            weekday = current["end"].weekday()
            previous_3 = statistics.median(prior_valid_gaps) if prior_valid_gaps else None
            prior_events_6h = sum((current["end"] - row["start"]).total_seconds() <= 6 * 3600 for row in rows[max(0, index - 1000):index])
            prior_events_24h = sum((current["end"] - row["start"]).total_seconds() <= 24 * 3600 for row in rows[max(0, index - 1000):index])
            feature_values = [
                math.log1p(current["duration_s"]),
                math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
                math.sin(2 * math.pi * weekday / 7), math.cos(2 * math.pi * weekday / 7),
                math.log1p(previous_gap_h) if previous_gap_h is not None and previous_gap_h >= 0 else 0.0,
                float(previous_gap_h is None or previous_gap_h < 0),
                math.log1p(previous_3) if previous_3 is not None else 0.0,
                float(previous_3 is None),
                float(prior_events_6h), float(prior_events_24h),
            ]
            samples.append({
                "animal": animal,
                "date": current["start"].date(),
                "next_date": next_event["start"].date(),
                "hour": current["end"].hour,
                "x": feature_values,
                "target_h": float(target_gap_h),
            })

    durations = [event["duration_s"] for event in events]
    all_positive_gaps = [row["target_h"] for row in samples if row["target_h"] > 0]
    by_equipment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_equipment[event["equipment"]].append(event)
    same_equipment_overlap = 0
    for rows in by_equipment.values():
        rows.sort(key=lambda e: (e["start"], e["end"]))
        same_equipment_overlap += sum(rows[i]["start"] < rows[i - 1]["end"] for i in range(1, len(rows)))
    audit = {
        "event_rows": len(events),
        "animal_count": len(by_animal),
        "animals_single_event": sum(len(rows) == 1 for rows in by_animal.values()),
        "paper_reported_animal_count": 37,
        "animal_count_discrepancy": len(by_animal) != 37,
        "equipment_count": len({event["equipment"] for event in events}),
        "date_count": len({event["start"].date() for event in events}),
        "start_min_local": min(event["start"] for event in events).isoformat(sep=" "),
        "end_max_local": max(event["end"] for event in events).isoformat(sep=" "),
        "timezone": "not encoded; timestamps treated as local naive time",
        "duration_s": {
            "min": int(min(durations)), "median": float(statistics.median(durations)),
            "mean": float(statistics.mean(durations)), "max": int(max(durations)),
            "zero_count": sum(value == 0 for value in durations),
            "over_300_s_count": sum(value > 300 for value in durations),
        },
        "duplicate_events": duplicates,
        "same_equipment_overlapping_events": int(same_equipment_overlap),
        "left_context_missing_first_visit_count": left_context_missing,
        "right_censored_last_visit_count": right_censored,
        "interarrival_gap_h": {
            "target_definition": "next IN minus current OUT",
            "negative_overlap_rows_excluded": negative_gaps,
            "negative_overlap_seconds_median": float(statistics.median(overlap_duration_s)) if overlap_duration_s else None,
            "zero_gap_rows_retained": zero_gaps,
            "valid_nonnegative_targets": len(samples),
            "positive_median_h": float(statistics.median(all_positive_gaps)) if all_positive_gaps else None,
            "positive_max_h": float(max(all_positive_gaps)) if all_positive_gaps else None,
            "over_8h_count": sum(value > 8 for value in all_positive_gaps),
            "over_12h_count": sum(value > 12 for value in all_positive_gaps),
            "over_24h_count": sum(value > 24 for value in all_positive_gaps),
        },
        "limits": [
            "The Kaggle CSV records only animal/equipment IN/OUT timestamps; no intake, ethogram, water volume, pasture location or treatment labels are present.",
            "The associated study reports 37 cattle while Kaggle v2 has 38 IDs, including one single-event ID; this discrepancy remains unresolved.",
            "Negative next gaps reflect overlapping same-animal event records and are excluded from next-gap regression, not silently clipped.",
            "Last events per animal are right-censored and excluded; first events lack prior history and use an explicit missing flag.",
            "This is a 21-day feed-efficiency/fed-trough cohort, not pasture grazing or a validated routine daily activity process.",
        ],
    }
    return samples, audit


def make_splits(samples: list[dict[str, Any]], seed: int) -> tuple[dict[str, str], dict[Any, str]]:
    counts = Counter(sample["animal"] for sample in samples)
    animals = sorted(counts)
    random.Random(seed).shuffle(animals)
    n = len(animals)
    n_test = max(1, math.ceil(n * 0.2))
    n_val = max(1, math.ceil(n * 0.2))
    n_train = n - n_test - n_val
    if n_train < 1:
        raise ValueError("Not enough animals for disjoint train/validation/test")
    animal_split = {
        **{a: "train" for a in animals[:n_train]},
        **{a: "validation" for a in animals[n_train:n_train + n_val]},
        **{a: "test" for a in animals[n_train + n_val:]},
    }
    dates = sorted({sample["date"] for sample in samples} | {sample["next_date"] for sample in samples})
    if len(dates) < 3:
        raise ValueError("Need at least three dates for chronological holdout")
    n_train_dates = max(1, math.floor(len(dates) * 0.6))
    n_val_dates = max(1, math.floor(len(dates) * 0.2))
    n_val_dates = min(n_val_dates, len(dates) - n_train_dates - 1)
    time_split = {
        **{date: "train" for date in dates[:n_train_dates]},
        **{date: "validation" for date in dates[n_train_dates:n_train_dates + n_val_dates]},
        **{date: "test" for date in dates[n_train_dates + n_val_dates:]},
    }
    return animal_split, time_split


def split_rows(samples: list[dict[str, Any]], animal_split: dict[str, str], time_split: dict[Any, str]):
    animal_indices = {name: [] for name in ("train", "validation", "test")}
    time_indices = {name: [] for name in ("train", "validation", "test")}
    purged: Counter[str] = Counter()
    for index, sample in enumerate(samples):
        animal_indices[animal_split[sample["animal"]]].append(index)
        current_split = time_split.get(sample["date"])
        future_split = time_split.get(sample["next_date"])
        if current_split == future_split and current_split is not None:
            time_indices[current_split].append(index)
        else:
            purged[f"{current_split}->{future_split}"] += 1
    return animal_indices, time_indices, dict(purged)


def matrices(samples: list[dict[str, Any]], indices: list[int]):
    subset = [samples[i] for i in indices]
    x = np.asarray([row["x"] for row in subset], dtype=np.float32)
    y = np.asarray([row["target_h"] for row in subset], dtype=np.float32)
    return subset, x, y


class GapNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def train_model(x_train, y_train, x_val, y_val, seed: int, device: torch.device, output_dir: Path, label: str):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    tx = torch.tensor((x_train - x_mean) / x_std, dtype=torch.float32, device=device)
    ty = torch.tensor(np.log1p(y_train), dtype=torch.float32, device=device)
    vx = torch.tensor((x_val - x_mean) / x_std, dtype=torch.float32, device=device)
    vy = np.log1p(y_val)
    model = GapNet(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    patience = 8
    epochs_without_improvement = 0
    batch_size = 2048
    optimizer_steps = 0
    for epoch in range(1, 81):
        model.train()
        order = torch.randperm(len(tx), device=device)
        for start in range(0, len(tx), batch_size):
            batch = order[start:start + batch_size]
            prediction = model(tx[batch])
            loss = nn.functional.huber_loss(prediction, ty[batch], delta=0.5)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer_steps += 1
        model.eval()
        with torch.no_grad():
            val_prediction = model(vx).detach().cpu().numpy()
        val_log_mae = float(np.mean(np.abs(val_prediction - vy)))
        if val_log_mae < best_val - 1e-6:
            best_val = val_log_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training failed to produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = output_dir / f"gap_predictor_{label}.pt"
    torch.save({
        "state_dict": best_state,
        "input_dim": int(x_train.shape[1]),
        "features": FEATURES,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "target_transform": "log1p(hours); inverse expm1",
        "best_epoch_selected_by_validation_log_mae": best_epoch,
        "validation_log_mae": best_val,
        "seed": seed,
        "split": label,
        "not_a_policy": True,
    }, checkpoint)
    return model, x_mean, x_std, {
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_validation_log_mae": best_val,
        "training_device": str(device),
        "optimizer_steps": optimizer_steps,
        "checkpoint": checkpoint.name,
    }


def baseline_predictions(train_rows, target_hours, rows):
    global_median = float(np.median(target_hours))
    by_hour: dict[int, list[float]] = defaultdict(list)
    for row, target in zip(train_rows, target_hours):
        by_hour[row["hour"]].append(float(target))
    hour_medians = {hour: float(np.median(values)) for hour, values in by_hour.items()}
    result = {
        "training_global_median": np.full(len(rows), global_median, dtype=np.float64),
        "training_hourly_median": np.asarray([hour_medians.get(row["hour"], global_median) for row in rows]),
        "persistence_previous_gap": np.asarray([
            math.expm1(row["x"][5]) if row["x"][6] == 0 else global_median for row in rows
        ]),
    }
    return result, {"global_median_h": global_median, "hourly_medians_h": {str(k): v for k, v in sorted(hour_medians.items())}}


def evaluate(y_hours: np.ndarray, prediction: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred = np.clip(np.asarray(prediction, dtype=np.float64), 0.0, 24.0)
    target = np.asarray(y_hours, dtype=np.float64)
    per_animal = {}
    for animal in sorted({row["animal"] for row in rows}):
        mask = np.asarray([row["animal"] == animal for row in rows])
        per_animal[str(animal)] = {
            "n": int(mask.sum()),
            "mae_hours": float(np.mean(np.abs(pred[mask] - target[mask]))),
            "mae_log1p_hours": float(np.mean(np.abs(np.log1p(pred[mask]) - np.log1p(target[mask])))),
        }
    animal_maes = [value["mae_hours"] for value in per_animal.values()]
    return {
        "n": int(len(target)),
        "mae_hours_pooled": float(np.mean(np.abs(pred - target))),
        "mae_log1p_hours_pooled": float(np.mean(np.abs(np.log1p(pred) - np.log1p(target)))),
        "macro_animal_mae_hours": float(np.mean(animal_maes)),
        "per_animal": per_animal,
    }


def bootstrap_delta(model_metrics, baseline_metrics, seed: int, reps: int = 2000) -> dict[str, Any]:
    model = model_metrics["per_animal"]
    base = baseline_metrics["per_animal"]
    shared = sorted(set(model) & set(base))
    if len(shared) < 2:
        return {"animals": len(shared), "ci95_delta_macro_animal_mae_h": None}
    delta = np.asarray([model[animal]["mae_hours"] - base[animal]["mae_hours"] for animal in shared])
    rng = np.random.default_rng(seed)
    draws = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        draws[i] = np.mean(delta[rng.integers(0, len(delta), size=len(delta))])
    return {
        "animals": len(shared),
        "point_delta_macro_animal_mae_h": float(delta.mean()),
        "ci95_delta_macro_animal_mae_h": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "interpretation": "negative favours the neural model; bootstrap unit is animal",
    }


def run_split(name: str, samples, partitions, seed: int, device: torch.device, output_dir: Path):
    train_rows, x_train, y_train = matrices(samples, partitions["train"])
    val_rows, x_val, y_val = matrices(samples, partitions["validation"])
    test_rows, x_test, y_test = matrices(samples, partitions["test"])
    if min(len(train_rows), len(val_rows), len(test_rows)) == 0:
        raise ValueError(f"Empty {name} split: train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)}")
    model, x_mean, x_std, fit_meta = train_model(x_train, y_train, x_val, y_val, seed, device, output_dir, name)

    val_baselines, baseline_fit = baseline_predictions(train_rows, y_train, val_rows)
    val_result = {key: evaluate(y_val, value, val_rows) for key, value in val_baselines.items()}
    best_baseline_name = min(val_result, key=lambda key: val_result[key]["macro_animal_mae_hours"])
    model.eval()
    with torch.no_grad():
        normalized_val = torch.tensor((x_val - x_mean) / x_std, dtype=torch.float32, device=device)
        predicted_val_log = model(normalized_val).detach().cpu().numpy()
        predicted_val_h = np.expm1(np.clip(predicted_val_log, 0.0, math.log1p(24.0)))
    val_result["neural_mlp"] = evaluate(y_val, predicted_val_h, val_rows)

    test_baselines, _ = baseline_predictions(train_rows, y_train, test_rows)
    test_result = {key: evaluate(y_test, value, test_rows) for key, value in test_baselines.items()}
    with torch.no_grad():
        normalized_test = torch.tensor((x_test - x_mean) / x_std, dtype=torch.float32, device=device)
        predicted_test_log = model(normalized_test).detach().cpu().numpy()
        predicted_test_h = np.expm1(np.clip(predicted_test_log, 0.0, math.log1p(24.0)))
    test_result["neural_mlp"] = evaluate(y_test, predicted_test_h, test_rows)
    test_result["neural_vs_validation_selected_baseline_bootstrap"] = bootstrap_delta(
        test_result["neural_mlp"], test_result[best_baseline_name], seed + 113
    )
    val_compare = val_result["neural_mlp"]["macro_animal_mae_hours"] < val_result[best_baseline_name]["macro_animal_mae_hours"]
    test_compare = test_result["neural_mlp"]["macro_animal_mae_hours"] < test_result[best_baseline_name]["macro_animal_mae_hours"]
    ci = test_result["neural_vs_validation_selected_baseline_bootstrap"]["ci95_delta_macro_animal_mae_h"]
    robust_test_win = ci is not None and ci[1] < 0 and test_result["neural_mlp"]["mae_hours_pooled"] < test_result[best_baseline_name]["mae_hours_pooled"]
    return {
        "split_name": name,
        "train_rows": len(train_rows), "validation_rows": len(val_rows), "test_rows": len(test_rows),
        "fit": fit_meta,
        "baseline_selected_on_validation_macro_animal_mae": best_baseline_name,
        "baseline_fit_parameters_from_train_only": baseline_fit,
        "validation_metrics": val_result,
        "test_metrics": test_result,
        "validation_neural_beats_selected_baseline": val_compare,
        "test_neural_beats_selected_baseline": test_compare,
        "test_robust_acceptance_gate": robust_test_win,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Pinned Kaggle source CSV or its downloaded zip")
    parser.add_argument("--output-dir", type=Path, default=Path("training/visits/runs/next-gap"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_path = resolve_data(args.data)
    events, source_meta = read_source(source_path)
    samples, audit = build_samples(events)
    animal_split, time_split = make_splits(samples, args.seed)
    animal_rows, time_rows, time_purged = split_rows(samples, animal_split, time_split)

    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError("--require-cuda set, but torch.cuda.is_available() is false")
    if Path("/kaggle/input").exists() and not cuda_available:
        raise RuntimeError("Kaggle dataset input is mounted but CUDA is unavailable; refusing a CPU fallback")
    if args.device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and cuda_available)) else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    runtime = {
        "device": str(device),
        "cuda_available": bool(cuda_available),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "seed": args.seed,
    }
    print(json.dumps({"phase": "runtime", **runtime}, ensure_ascii=False), flush=True)
    animal_result = run_split("animal_holdout", samples, animal_rows, args.seed, device, args.output_dir)
    time_result = run_split("future_time_holdout", samples, time_rows, args.seed + 1, device, args.output_dir)

    animal_partition_ids = {name: sorted(animal for animal, split in animal_split.items() if split == name) for name in ("train", "validation", "test")}
    time_partition_dates = {name: sorted(date.isoformat() for date, split in time_split.items() if split == name) for name in ("train", "validation", "test")}
    # The model is a research-only event-time predictor. It is never emitted as an animal policy.
    accepted = bool(animal_result["test_robust_acceptance_gate"] and time_result["test_robust_acceptance_gate"])
    artifact_hashes = {}
    for result in (animal_result, time_result):
        path = args.output_dir / result["fit"]["checkpoint"]
        artifact_hashes[path.name] = sha256(path.read_bytes())
    manifest = {
        "experiment": "nellore_next_visit_gap_supervised_v1",
        "dataset": source_meta,
        "data_audit": audit,
        "label": {
            "target": "next event IN minus current event OUT, hours",
            "transform": "log1p(gap_hours)",
            "negative_overlaps": "excluded from regression and counted in audit; not clipped to zero",
            "zero_gaps": "retained",
            "right_censored_last_events": "excluded; one per animal with events",
            "long_gaps": "retained without winsorization; separately counted above 8h, 12h and 24h",
        },
        "features": FEATURES,
        "feature_policy": "Only current completed event and prior event history; no animal_id/equip_id, no future rows, no intake or behavior labels.",
        "splits": {
            "animal_group_seed": args.seed,
            "animal_ids": animal_partition_ids,
            "time_dates": time_partition_dates,
            "time_rows_purged_across_date_partitions": time_purged,
            "animal_row_counts": {name: len(rows) for name, rows in animal_rows.items()},
            "time_row_counts": {name: len(rows) for name, rows in time_rows.items()},
        },
        "protocol": {
            "baseline_selection": "lowest validation macro-animal MAE among global training median, training hourly median, and previous-gap persistence",
            "neural_checkpoint_selection": "lowest validation MAE in log1p hours; test never used for epochs, hyperparameters, or model selection",
            "uncertainty": "paired bootstrap over animals (2000 resamples) for macro-animal MAE delta; negative delta favours MLP",
            "policy_exported": False,
            "policy_acceptance_gate": "candidate needs pooled test MAE improvement and 95% animal-bootstrap upper bound below zero on both held-out-animal and future-time tests; this remains predictive, not causal or a policy",
        },
        "results": {"animal_holdout": animal_result, "future_time_holdout": time_result},
        "candidate_status": "passes_predictive_gate_only_not_a_policy" if accepted else "no_evidence_of_robust_generalization_over_baselines",
        "runtime": runtime,
        "checkpoints": artifact_hashes,
        "scientific_limits": [
            "Single 21-day feed-efficiency cohort; no external herd/farm, pasture, treatment, or physiological labels.",
            "Kaggle v2 has 38 distinct IDs while the paper reports 37; one ID has one event and contributes no interval label.",
            "A trough event is an observed equipment visit, not a validated eating/drinking/rumination event or internal need.",
            "The source license is CC BY-NC-SA 4.0; weights derived from these labels remain subject to license review before redistribution or commercial use.",
        ],
    }
    evaluation_path = args.output_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "phase": "complete", "runtime": runtime,
        "candidate_status": manifest["candidate_status"],
        "animal_test_mlp": animal_result["test_metrics"]["neural_mlp"],
        "time_test_mlp": time_result["test_metrics"]["neural_mlp"],
        "evaluation": str(evaluation_path),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
