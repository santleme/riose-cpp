#!/usr/bin/env python3
"""Fetch and validate the public Kaggle Nellore feed-bin events (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import shlex
import statistics
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PATH = ROOT / "data/raw/kaggle/nellore-trough-v2.zip"
NORMALIZED_PATH = ROOT / "data/processed/nellore-trough-events.csv"
SPLITS_PATH = ROOT / "data/processed/nellore-trough-splits.csv"
REPORT_PATH = ROOT / "data/reports/nellore-trough-audit.json"
DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "juliavalente/data-from-visits-to-the-trough-of-nellore-cattle"
)
EXPECTED_HEADER = ["animal_id", "equip_id", "IN", "OUT"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_archive(destination: Path) -> bytes:
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={"User-Agent": "CampoVivoDatasetPipeline/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        if "zip" not in content_type.lower():
            raise ValueError(f"Kaggle download returned unexpected content type: {content_type}")
        archive = response.read()
    zipfile.ZipFile(io.BytesIO(archive)).testzip()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(archive)
    return archive


def read_archive(path: Path) -> tuple[bytes, str, list[dict[str, str]]]:
    archive = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        members = [info for info in zf.infolist() if not info.is_dir()]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one data file in archive, found {len(members)}")
        info = members[0]
        text = zf.read(info.filename).decode("utf-8-sig")

    lines = text.splitlines()
    if not lines:
        raise ValueError("The dataset file is empty")
    header = shlex.split(lines[0])
    if header != EXPECTED_HEADER:
        raise ValueError(f"Unexpected columns: {header!r}; expected {EXPECTED_HEADER!r}")

    events: list[dict[str, str]] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = shlex.split(line)
        if len(fields) != len(EXPECTED_HEADER):
            raise ValueError(f"Line {line_number}: expected 4 whitespace-quoted fields, got {len(fields)}")
        events.append(dict(zip(EXPECTED_HEADER, fields)))
    return archive, info.filename, events


def parse_events(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    parsed: list[dict[str, object]] = []
    for line_number, row in enumerate(rows, start=2):
        animal = row["animal_id"].strip()
        equipment = row["equip_id"].strip()
        try:
            start = dt.datetime.strptime(row["IN"], TIME_FORMAT)
            end = dt.datetime.strptime(row["OUT"], TIME_FORMAT)
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: invalid local timestamp: {exc}") from exc
        duration = (end - start).total_seconds()
        if not animal or not equipment:
            raise ValueError(f"Line {line_number}: animal_id and equip_id must be non-empty")
        if duration < 0:
            raise ValueError(f"Line {line_number}: OUT precedes IN")
        parsed.append(
            {
                "animal_id": animal,
                "equip_id": equipment,
                "start": start,
                "end": end,
                "duration_s": int(duration),
            }
        )
    return parsed


def _hash_order(value: str, seed: str) -> bytes:
    return hashlib.sha256(f"{seed}:{value}".encode()).digest()


def make_animal_splits(events: list[dict[str, object]], seed: str = "campo-vivo-v1") -> dict[str, str]:
    animals = sorted({str(row["animal_id"]) for row in events}, key=lambda x: _hash_order(x, seed))
    n = len(animals)
    if n < 3:
        raise ValueError("Animal holdout requires at least three distinct animal IDs")
    n_test = max(1, math.ceil(n * 0.2))
    n_validation = max(1, math.ceil(n * 0.2))
    n_train = n - n_test - n_validation
    if n_train < 1:
        raise ValueError("Not enough animals for train/validation/test partitions")
    return {
        **{animal: "train" for animal in animals[:n_train]},
        **{animal: "validation" for animal in animals[n_train : n_train + n_validation]},
        **{animal: "test" for animal in animals[n_train + n_validation :]},
    }


def make_time_splits(events: list[dict[str, object]]) -> tuple[dict[dt.date, str], list[dt.date]]:
    dates = sorted({row["start"].date() for row in events})
    if len(dates) < 3:
        raise ValueError("Chronological split requires at least three distinct dates")
    n = len(dates)
    train_end = max(1, math.floor(n * 0.6))
    validation_end = max(train_end + 1, math.floor(n * 0.8))
    validation_end = min(validation_end, n - 1)
    split_by_date = {
        **{date: "train" for date in dates[:train_end]},
        **{date: "validation" for date in dates[train_end:validation_end]},
        **{date: "test" for date in dates[validation_end:]},
    }
    return split_by_date, dates


def validate_splits(
    events: list[dict[str, object]],
    animal_split: dict[str, str],
    time_split: dict[dt.date, str],
    dates: list[dt.date],
) -> None:
    animal_sets = {
        name: {animal for animal, split in animal_split.items() if split == name}
        for name in ("train", "validation", "test")
    }
    if animal_sets["train"] & animal_sets["validation"] or animal_sets["train"] & animal_sets["test"] or animal_sets["validation"] & animal_sets["test"]:
        raise AssertionError("Animal groups overlap across animal-held-out partitions")
    if set(animal_split) != {str(row["animal_id"]) for row in events}:
        raise AssertionError("Animal split assignment does not cover all animals")
    if len(time_split) != len(set(time_split)) or set(time_split) != set(dates):
        raise AssertionError("Temporal assignment does not cover each observed date exactly once")
    if not (max(day for day, name in time_split.items() if name == "train") < min(day for day, name in time_split.items() if name == "validation")):
        raise AssertionError("Validation dates must occur after train dates")
    if not (max(day for day, name in time_split.items() if name == "validation") < min(day for day, name in time_split.items() if name == "test")):
        raise AssertionError("Test dates must occur after validation dates")


def time_partition_for_event(row: dict[str, object], time_split: dict[dt.date, str]) -> str:
    start_partition = time_split[row["start"].date()]
    end_partition = time_split.get(row["end"].date())
    return start_partition if end_partition == start_partition else "purged"


def audit(events: list[dict[str, object]], archive: bytes, member: str) -> dict[str, object]:
    animals = sorted({str(row["animal_id"]) for row in events})
    equipment = sorted({str(row["equip_id"]) for row in events})
    dates = sorted({row["start"].date() for row in events})
    durations = [int(row["duration_s"]) for row in events]
    overnight_events = sum(row["start"].date() != row["end"].date() for row in events)
    animal_counts = Counter(str(row["animal_id"]) for row in events)
    equipment_counts = Counter(str(row["equip_id"]) for row in events)
    starts_by_equipment: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in events:
        starts_by_equipment[str(row["equip_id"])].append(row)
    same_bin_gaps = []
    for rows in starts_by_equipment.values():
        ordered = sorted(rows, key=lambda row: row["start"])
        same_bin_gaps.extend(
            (ordered[index]["start"] - ordered[index - 1]["end"]).total_seconds()
            for index in range(1, len(ordered))
        )

    animal_split = make_animal_splits(events)
    time_split, dates = make_time_splits(events)
    validate_splits(events, animal_split, time_split, dates)
    raw_order_breaks = sum(
        events[index]["start"] < events[index - 1]["start"]
        for index in range(1, len(events))
    )
    cross_partition_midnight_events = sum(
        time_split[row["start"].date()] != time_split[row["end"].date()]
        for row in events
        if row["end"].date() in time_split
    )

    report: dict[str, object] = {
        "dataset_id": "nellore-trough-events-2021",
        "source_version": 2,
        "license": "CC-BY-NC-SA-4.0",
        "archive_member": member,
        "archive_bytes": len(archive),
        "archive_sha256": sha256_bytes(archive),
        "row_count": len(events),
        "column_names": EXPECTED_HEADER,
        "columns_added_by_pipeline": ["duration_s"],
        "columns_not_available_in_kaggle_archive": ["consumption", "pen", "behavior_label", "water_volume", "gps"],
        "animal_count": len(animals),
        "paper_reported_animal_count": 37,
        "animal_count_mismatch": len(animals) != 37,
        "single_event_animal_count": sum(1 for count in animal_counts.values() if count == 1),
        "equipment_count": len(equipment),
        "equipment_event_counts": dict(sorted(equipment_counts.items())),
        "date_count": len(dates),
        "date_start_local": dates[0].isoformat(),
        "date_end_local": dates[-1].isoformat(),
        "timezone": "not encoded in source; timestamps retained as local naive datetime",
        "events_crossing_local_midnight": overnight_events,
        "raw_row_order_breaks_by_start_time": raw_order_breaks,
        "midnight_events_crossing_chronological_partition": cross_partition_midnight_events,
        "duration_seconds": {
            "min": min(durations),
            "median": statistics.median(durations),
            "mean": statistics.mean(durations),
            "max": max(durations),
            "zero_duration_count": sum(value == 0 for value in durations),
            "over_300_seconds_count": sum(value > 300 for value in durations),
        },
        "duplicate_event_count": len(events)
        - len({(row["animal_id"], row["equip_id"], row["start"], row["end"]) for row in events}),
        "same_equipment_gap_seconds": {
            "negative_overlap_count": sum(value < 0 for value in same_bin_gaps),
            "zero_to_10_count": sum(0 <= value <= 10 for value in same_bin_gaps),
            "note": "candidate temporal proximity only; not verified aggression/replacement labels",
        },
        "splits": {
            "animal_group": {
                name: sum(1 for value in animal_split.values() if value == name)
                for name in ("train", "validation", "test")
            },
            "animal_group_event_rows": {
                name: sum(1 for row in events if animal_split[str(row["animal_id"])] == name)
                for name in ("train", "validation", "test")
            },
            "chronological_dates": {
                name: sum(1 for value in time_split.values() if value == name)
                for name in ("train", "validation", "test")
            },
            "chronological_event_rows": {
                name: sum(
                    1
                    for row in events
                    if time_split[row["start"].date()] == name
                    and time_split[row["end"].date()] == name
                )
                for name in ("train", "validation", "test")
            },
            "chronological_purged_rows": cross_partition_midnight_events,
            "seed": "campo-vivo-v1",
            "limits": [
                "animal holdout evaluates this one cohort only, not a new farm or management system",
                "chronological holdout includes animals seen earlier and is a future-within-cohort check",
                "event rows are repeated measures; random row splits are not supported",
            ],
        },
        "warnings": [
            "paper reports 37 animals but Kaggle v2 has 38 distinct IDs; one ID occurs once",
            "raw Kaggle archive has four columns; do not infer intake or behavior labels",
            "this is one feedlot/paddock cohort with a 21-day window, not pasture-grazing movement data",
            "CC BY-NC-SA 4.0 prohibits commercial use under the stated dataset license; obtain legal review before product/model redistribution",
        ],
    }
    return report


def write_outputs(events: list[dict[str, object]], report: dict[str, object]) -> None:
    NORMALIZED_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NORMALIZED_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "animal_id", "equip_id", "IN_local", "OUT_local", "duration_s"])
        for row in sorted(events, key=lambda item: (item["start"], item["equip_id"], item["animal_id"])):
            event_id = sha256_bytes(
                f"{row['animal_id']}|{row['equip_id']}|{row['start']}|{row['end']}".encode()
            )
            writer.writerow(
                [
                    event_id,
                    row["animal_id"],
                    row["equip_id"],
                    row["start"].isoformat(sep=" "),
                    row["end"].isoformat(sep=" "),
                    row["duration_s"],
                ]
            )

    animal_split = make_animal_splits(events)
    time_split, _ = make_time_splits(events)
    with SPLITS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "animal_split", "time_split"])
        for row in events:
            event_id = sha256_bytes(
                f"{row['animal_id']}|{row['equip_id']}|{row['start']}|{row['end']}".encode()
            )
            row_time_split = time_partition_for_event(row, time_split)
            writer.writerow(
                [event_id, animal_split[str(row["animal_id"])], row_time_split]
            )
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to a downloaded Kaggle zip archive")
    parser.add_argument("--download", action="store_true", help="Fetch the public Kaggle v2 archive")
    args = parser.parse_args(argv)
    path = args.input or ARCHIVE_PATH
    if args.download:
        archive = download_archive(path)
        print(f"downloaded {len(archive)} bytes to {path}")
    elif not path.exists():
        parser.error(f"archive not found: {path}; pass --download or --input PATH")

    archive, member, rows = read_archive(path)
    events = parse_events(rows)
    if len(events) != 90211:
        raise ValueError(f"Expected 90,211 rows for Kaggle v2; found {len(events)}")
    report = audit(events, archive, member)
    write_outputs(events, report)
    print(json.dumps({key: report[key] for key in ("row_count", "animal_count", "date_count", "archive_sha256", "warnings")}, indent=2))
    print(f"wrote {NORMALIZED_PATH.relative_to(ROOT)}, {SPLITS_PATH.relative_to(ROOT)}, {REPORT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile, AssertionError) as exc:
        print(f"data pipeline error: {exc}", file=sys.stderr)
        raise SystemExit(2)
