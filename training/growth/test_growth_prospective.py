from pathlib import Path

import numpy as np
import pandas as pd

from training.growth.train_growth_prospective import FEATURES, prepare_frame, resolve_data_path, rolling_origin


def _fixture(path: Path) -> Path:
    rows = []
    first = pd.Timestamp("2024-01-01")
    for animal in (1, 2):
        weight = 200.0 + animal * 20
        for period in range(1, 11):
            start = first + pd.Timedelta(days=(period - 1) * 7)
            end = start + pd.Timedelta(days=7)
            adg = 0.3 + animal * 0.1 + period * 0.01
            rows.append({
                "ANIMAL": animal, "PERIOD": period, "START_DATE": start.date().isoformat(),
                "FINAL_DATE": end.date().isoformat(), "AMOUNT_DAYS": 7,
                "START_WEIGHT": weight, "FINAL_WEIGHT": weight + adg * 7, "ADG": adg,
                "ADS": 99, "TEMP_AVG": 99, "NDVI_INDEX": 99,
            })
            weight += adg * 7
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_only_information_available_at_start_is_used(tmp_path):
    path = _fixture(tmp_path / "fixture.csv")
    frame, audit = prepare_frame(path, verify_hash=False)

    assert FEATURES == ["START_WEIGHT", "AMOUNT_DAYS", "DOY_SIN", "DOY_COS", "PREVIOUS_ADG"]
    assert not {"FINAL_WEIGHT", "ADG", "ADS", "TEMP_AVG", "NDVI_INDEX"}.intersection(FEATURES)
    period_one = frame.loc[frame["PERIOD"] == 1]
    period_two = frame.loc[frame["PERIOD"] == 2]
    assert period_one["PREVIOUS_ADG"].isna().all()
    assert np.allclose(period_two["PREVIOUS_ADG"], period_two["ANIMAL"].map({1: .41, 2: .51}))
    assert audit["rows_with_valid_previous_period_adg"] == 18


def test_rolling_origin_never_trains_on_test_or_future_periods(tmp_path):
    frame, _ = prepare_frame(_fixture(tmp_path / "fixture.csv"), verify_hash=False)

    result, _ = rolling_origin(frame, first_test_period=7)

    assert [split["test_period"] for split in result["splits"]] == [7, 8, 9, 10]
    for split in result["splits"]:
        assert max(split["train_periods"]) < split["test_period"]
        assert split["test_rows"] == 2
    assert set(result["models"]["ridge_start_state_and_previous_adg"]["paired_period_bootstrap_vs_baselines"]) == {
        "training_mean", "previous_adg_persistence"
    }


def test_kaggle_input_resolves_by_verified_filename_not_mount_slug(tmp_path):
    input_root = tmp_path / "input"
    mounted = input_root / "arbitrary-dataset-mount" / "Complete_Dataset.csv"
    mounted.parent.mkdir(parents=True)
    mounted.write_text("fixture")

    assert resolve_data_path(Path("/kaggle/input/expected-slug/Complete_Dataset.csv"),
                             input_root=input_root) == mounted
