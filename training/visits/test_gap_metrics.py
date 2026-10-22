import unittest

import numpy as np

from gap_metrics import evaluate


class PredictionClippingMetricsTest(unittest.TestCase):
    def test_reports_clipped_predictions_without_clipping_targets(self):
        target = np.asarray([30.0, 10.0, 0.0])
        prediction = np.asarray([30.0, -2.0, 12.0])
        rows = [{"animal": "a"}, {"animal": "a"}, {"animal": "b"}]

        metrics = evaluate(target, prediction, rows)

        self.assertAlmostEqual(metrics["mae_hours_pooled"], 28.0 / 3.0)
        self.assertEqual(metrics["prediction_clipping"]["upper_clipped_count"], 1)
        self.assertAlmostEqual(metrics["prediction_clipping"]["upper_clipped_fraction"], 1 / 3)
        self.assertEqual(metrics["prediction_clipping"]["lower_clipped_count"], 1)
        self.assertAlmostEqual(metrics["prediction_clipping"]["clipped_fraction"], 2 / 3)
        self.assertFalse(metrics["prediction_clipping"]["targets_clipped"])
        self.assertEqual(target[0], 30.0)


if __name__ == "__main__":
    unittest.main()
