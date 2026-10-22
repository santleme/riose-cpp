"""Synthetic checks for timestamp-safe IMU window extraction."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from training.train import _extract_windows


def rows(times: list[float], missing_sensor_at: set[int] | None = None) -> pd.DataFrame:
    missing_sensor_at = missing_sensor_at or set()
    values = []
    for index, seconds in enumerate(times):
        acc_x = np.nan if index in missing_sensor_at else float(index)
        values.append({
            "date": "2026-01-01", "time": seconds, "cow_num": "cow-a",
            "acc_x": acc_x, "acc_y": 0.0, "acc_z": 1.0, "behavior": 0,
        })
    return pd.DataFrame(values)


class ExtractWindowsContinuityTests(unittest.TestCase):
    def test_ordered_10_hz_rows_form_full_windows(self):
        x, y, animals, diag = _extract_windows(rows([0.0, 0.1, 0.2, 0.3]), window_samples=2)
        self.assertEqual(x.shape, (2, 18))
        self.assertEqual(y.tolist(), [0, 0])
        self.assertEqual(animals.tolist(), ["cow-a", "cow-a"])
        self.assertEqual(diag["sampling_gap_breaks"], 0)

    def test_shuffled_rows_are_sorted_before_windowing(self):
        ordered = rows([0.0, 0.1, 0.2, 0.3])
        shuffled = ordered.iloc[[2, 0, 3, 1]].reset_index(drop=True)
        ordered_x, ordered_y, _, _ = _extract_windows(ordered, window_samples=2)
        shuffled_x, shuffled_y, _, diag = _extract_windows(shuffled, window_samples=2)
        np.testing.assert_allclose(shuffled_x, ordered_x)
        np.testing.assert_array_equal(shuffled_y, ordered_y)
        self.assertGreater(diag["reordered_rows"], 0)

    def test_gap_starts_new_run_and_is_never_bridged(self):
        x, _, _, diag = _extract_windows(rows([0.0, 0.1, 0.2, 0.5, 0.6, 0.7]), window_samples=2)
        self.assertEqual(len(x), 2)  # Three windows would bridge across the 0.3 s gap.
        self.assertEqual(diag["sampling_gap_breaks"], 1)

    def test_duplicate_timestamp_starts_new_run(self):
        x, _, _, diag = _extract_windows(rows([0.0, 0.1, 0.1, 0.2, 0.3]), window_samples=2)
        self.assertEqual(len(x), 2)
        self.assertEqual(diag["duplicate_timestamp_breaks"], 1)

    def test_missing_sensor_row_is_dropped_and_gap_is_not_bridged(self):
        frame = rows([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], missing_sensor_at={2})
        x, _, _, diag = _extract_windows(frame, window_samples=3)
        self.assertEqual(len(x), 1)
        # The only accepted window is rows after the gap (indices 3, 4, 5).
        self.assertAlmostEqual(float(x[0, 0]), 4.0)
        self.assertEqual(diag["missing_sensor_or_label_rows"], 1)
        self.assertEqual(diag["sampling_gap_breaks"], 1)


if __name__ == "__main__":
    unittest.main()
