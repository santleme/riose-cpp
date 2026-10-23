#!/usr/bin/env python3
"""Audited 5 s window extraction for the official ActBeCalf release.

The extraction deliberately keeps each calf/segment separate and never stitches
short segments or timestamp gaps. This is an external activity-transfer set,
not a sample of daily behavior prevalence.
"""
from __future__ import annotations

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


if __name__ == "__main__":
    main()
