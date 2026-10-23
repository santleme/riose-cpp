from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training.external.actbecalf import extract_actbecalf, coarse_label, window_features


def _fixture() -> pd.DataFrame:
    rows = []
    start = pd.Timestamp("2024-01-01T00:00:00Z")
    # One complete 5 s bout, then two short segments that must not be stitched.
    specs = [(1, 1, 125, "Eating", 0), (1, 2, 60, "Rumination", 125),
             (1, 3, 65, "Rumination", 185), (2, 1, 125, "Lying", 250)]
    for calf, segment, n, label, offset in specs:
        for i in range(n):
            stamp = start + pd.Timedelta(seconds=(offset + i) / 25)
            rows.append((stamp.isoformat(), calf, float(i), 0.1, 1.0, label, segment))
    return pd.DataFrame(rows, columns=["dateTime", "calfid", "accX", "accY", "accZ", "behavior", "segId"])


def test_coarse_ethogram_mapping():
    assert coarse_label(" Eating ") == 0
    assert coarse_label("eating_forage") == 0
    assert coarse_label("eating_bedding") == 0
    assert coarse_label("Rumination") == 1
    assert coarse_label("rumination_lying") == 1
    assert coarse_label("Drinking") == 2
    assert coarse_label("Drinking milk") == 2


def test_windows_do_not_stitch_segments_and_use_animal_ids():
    frame = _fixture()
    frame = frame.rename(columns={"behavior": "behaviour"})
    path = __import__("pathlib").Path("/tmp/campo-actbecalf-test.csv")
    frame.to_csv(path, index=False)
    try:
        x, y, calves, audit = extract_actbecalf(path, verify_release=False)
    finally:
        path.unlink(missing_ok=True)
    assert x.shape == (2, 18)
    assert y.tolist() == [0, 2]
    assert calves.tolist() == [1, 2]
    assert audit["segments"] == 4
    assert audit["accepted_windows"] == 2
    assert audit["discarded_short_run_samples"] == 125


def test_feature_contract_is_18_floats():
    features = window_features(np.ones((125, 3), dtype=np.float32))
    assert features.shape == (18,)
    assert np.isfinite(features).all()
    assert np.isclose(features[12], np.sqrt(3))


def test_unknown_or_missing_labels_are_not_silently_mapped_to_other():
    for value in ("made-up label", None):
        frame = _fixture()
        frame.loc[0, "behavior"] = value
        path = __import__("pathlib").Path("/tmp/campo-actbecalf-invalid.csv")
        frame.to_csv(path, index=False)
        try:
            with pytest.raises(ValueError):
                extract_actbecalf(path, verify_release=False)
        finally:
            path.unlink(missing_ok=True)
