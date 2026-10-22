"""Small Python inference adapter for Campo Vivo's activity window model."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from .train import FEATURE_NAMES, WINDOW_SAMPLES, _window_features
except ImportError:  # Direct script-style use with training/ on PYTHONPATH.
    from train import FEATURE_NAMES, WINDOW_SAMPLES, _window_features


class ActivityPredictor:
    """Load a saved artifact and classify one 5-second, 10 Hz IMU window.

    `predict_window` expects shape `(50, 3)` in the same axis order and units as
    the dataset's `acc_x`, `acc_y`, `acc_z` columns (g). The adapter returns the
    public activity label string and class scores.
    """

    LABELS = {0: "feeding", 1: "rumination", 2: "standing", 3: "lying", 4: "walking"}

    def __init__(self, artifact: str | Path):
        self.path = Path(artifact)
        self.model: Any = None
        self.checkpoint: dict[str, Any] | None = None
        if self.path.suffix == ".joblib":
            self.model = joblib.load(self.path)
        elif self.path.suffix == ".pt":
            import torch
            self.torch = torch
            self.checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
            self.net = torch.nn.Sequential(
                torch.nn.Linear(len(FEATURE_NAMES), 96), torch.nn.ReLU(), torch.nn.Dropout(0.15),
                torch.nn.Linear(96, 48), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(48, len(self.checkpoint["classes"])),
            )
            self.net.load_state_dict(self.checkpoint["state_dict"])
            self.net.eval()
        else:
            raise ValueError("Artifact must be a .joblib or .pt file")

    def predict_window(self, accelerometer_g: np.ndarray) -> dict[str, Any]:
        data = np.asarray(accelerometer_g, dtype=np.float32)
        if data.shape != (WINDOW_SAMPLES, 3):
            raise ValueError(f"Expected ({WINDOW_SAMPLES}, 3), got {data.shape}")
        features = _window_features(data).reshape(1, -1)
        if self.model is not None:
            probabilities = self.model.predict_proba(features)[0]
            classes = self.model.classes_
        else:
            checkpoint = self.checkpoint
            normalized = (features - np.asarray(checkpoint["scaler_mean"])) / np.asarray(checkpoint["scaler_scale"])
            with self.torch.no_grad():
                logits = self.net(self.torch.tensor(normalized, dtype=self.torch.float32))[0]
                probabilities = self.torch.softmax(logits, dim=0).numpy()
            classes = checkpoint["classes"]
        scores = {self.LABELS[int(label)]: float(score) for label, score in zip(classes, probabilities)}
        label = max(scores, key=scores.get)
        return {"activity": label, "scores": scores}
