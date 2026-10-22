"""Scoring utilities for next-visit interval prediction."""
from __future__ import annotations

from typing import Any

import numpy as np


def evaluate(y_hours: np.ndarray, prediction: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score clipped predictions and report clipping; targets remain uncapped."""
    raw_pred = np.asarray(prediction, dtype=np.float64)
    lower_clipped = raw_pred < 0.0
    upper_clipped = raw_pred > 24.0
    clipped = np.logical_or(lower_clipped, upper_clipped)
    pred = np.clip(raw_pred, 0.0, 24.0)
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
        "prediction_clipping": {
            "lower_bound_hours": 0.0,
            "upper_bound_hours": 24.0,
            "lower_clipped_count": int(lower_clipped.sum()),
            "lower_clipped_fraction": float(lower_clipped.mean()) if len(raw_pred) else 0.0,
            "upper_clipped_count": int(upper_clipped.sum()),
            "upper_clipped_fraction": float(upper_clipped.mean()) if len(raw_pred) else 0.0,
            "clipped_count": int(clipped.sum()),
            "clipped_fraction": float(clipped.mean()) if len(raw_pred) else 0.0,
            "targets_clipped": False,
        },
        "mae_hours_pooled": float(np.mean(np.abs(pred - target))),
        "mae_log1p_hours_pooled": float(np.mean(np.abs(np.log1p(pred) - np.log1p(target)))),
        "macro_animal_mae_hours": float(np.mean(animal_maes)),
        "per_animal": per_animal,
    }
