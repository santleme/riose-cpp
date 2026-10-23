import numpy as np

from training.external.train_actbecalf_transfer import score_grouped


def test_grouped_scores_expose_one_class_collapse():
    y = np.asarray([0, 0, 1, 2, 2])
    prediction = np.full_like(y, 2)
    animals = np.asarray([1, 1, 1, 2, 2])

    metrics = score_grouped(y, prediction, animals, seed=19)

    assert np.isclose(metrics["macro_f1_window_pooled"], 2 / 10.5)
    assert metrics["predicted_class_counts"] == {"feeding": 0, "rumination": 0, "other": 5}
    assert metrics["macro_f1_animal_mean"] == 0.5
