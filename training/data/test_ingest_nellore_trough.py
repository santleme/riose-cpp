import datetime as dt
import unittest

try:
    from .ingest_nellore_trough import (
        make_animal_splits,
        make_time_splits,
        parse_events,
        time_partition_for_event,
        validate_splits,
    )
except ImportError:  # Also allow running this file directly from training/data.
    from ingest_nellore_trough import (
        make_animal_splits,
        make_time_splits,
        parse_events,
        time_partition_for_event,
        validate_splits,
    )


class NelloreTroughPipelineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"animal_id": str(animal), "equip_id": "1", "IN": f"2021-06-{day:02d} 10:00:00", "OUT": f"2021-06-{day:02d} 10:00:30"}
            for animal in range(1, 11)
            for day in range(1, 7)
        ]

    def test_duration_is_derived_and_nonnegative(self):
        events = parse_events(self.rows)
        self.assertTrue(all(row["duration_s"] == 30 for row in events))

    def test_negative_duration_is_rejected(self):
        row = dict(self.rows[0], **{"OUT": "2021-06-01 09:59:59"})
        with self.assertRaisesRegex(ValueError, "OUT precedes IN"):
            parse_events([row])

    def test_animal_and_time_splits_are_independent_and_ordered(self):
        events = parse_events(self.rows)
        groups = make_animal_splits(events)
        times, dates = make_time_splits(events)
        validate_splits(events, groups, times, dates)
        self.assertEqual(set(groups.values()), {"train", "validation", "test"})
        self.assertEqual(set(times.values()), {"train", "validation", "test"})
        self.assertLess(
            max(day for day, split in times.items() if split == "train"),
            min(day for day, split in times.items() if split == "validation"),
        )

    def test_event_crossing_a_temporal_partition_is_purged(self):
        row = {
            "animal_id": "1",
            "equip_id": "1",
            "IN": "2021-06-02 23:59:50",
            "OUT": "2021-06-03 00:00:10",
        }
        event = parse_events([row])[0]
        partitions = {
            dt.date(2021, 6, 2): "train",
            dt.date(2021, 6, 3): "validation",
        }
        self.assertEqual(time_partition_for_event(event, partitions), "purged")


if __name__ == "__main__":
    unittest.main()
