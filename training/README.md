# Bovino activity baseline

Reproducible supervised baseline for coarse activity recognition from 3-axis cow IMU signals. It is an observation classifier, **not** a model of motivation, health, welfare, or causal environmental response.

## Data

Default source: [Cow nose ring data set on Kaggle](https://www.kaggle.com/datasets/fandaoerji/cow-nose-ring-data-set). The Kaggle metadata declares **CC0-1.0**. The uploader describes seven free-moving cows, 10 Hz accelerometer data, 1,252,522 rows, and labels `0 feeding`, `1 rumination`, `2 standing`, `3 lying`, `4 walking`. The raw dataset is not committed; fetching is handled by the official Kaggle CLI.

The script groups each animal/day independently, extracts non-overlapping 5-second windows, discards windows where the most common frame label is below 80%, and uses per-axis and acceleration-magnitude statistics. It reserves cow 1 for model selection and cow 6 for the final animal-held-out evaluation. It compares Extra Trees, Random Forest, and an MLP; on a CUDA runtime it also fits a small PyTorch MLP on the GPU. It records validation and test macro-F1, weighted-F1, per-class report, confusion matrix, class counts, split IDs, and runtime.

This is a small imbalanced dataset: walking, feeding, and rumination dominate, while standing and lying are rare. Even an animal-held-out score is only a pilot on one of seven animals. Use leave-one-animal-out evaluation and prospective field validation before making generalization or biological claims. The labels describe observed activity, not internal state or causal response to feed, water, pasture, weather, or social context.

## Kaggle run (2026-10-22)

Kernel version 2 completed with `NvidiaTeslaT4` requested and `cuda_available=true`. Animal 1 selected the MLP; animal 6 remained the test group. Test macro-F1 was 0.491 for the selected MLP (weighted-F1 0.814). Extra Trees scored 0.488 macro-F1, Random Forest 0.485, and the CUDA PyTorch MLP 0.510. The higher Torch test score does not replace the validation-selected model: its animal-1 validation macro-F1 was 0.420 versus 0.543 for the selected MLP. Tiny classes explain much of the instability: the validation animal had only three lying windows and no standing windows; the test animal had 14 lying and 8 standing windows. See [evaluation.json](outputs/evaluation.json), [Kaggle run metadata](experiments/kaggle-cuda-run.json), and the captured [kernel log](experiments/kaggle-cuda-run.log). The private [Kaggle kernel](https://www.kaggle.com/code/pascalcoimbra/campo-vivo-cow-activity-baseline) is the remote execution record.

## Local run

```bash
python -m pip install -r training/requirements.txt
python training/train.py
```

With a local CSV:

```bash
python training/train.py --data /path/to/data.csv --output-dir training/outputs
```

For an offline smoke run only (not scientific results):

```bash
python -m pip install -r training/requirements.txt
python training/train.py --smoke-synthetic --holdout-animal 4
```

The Python adapter loads either serialized scikit-learn or PyTorch artifact:

```python
import numpy as np
from training.infer import ActivityPredictor

predictor = ActivityPredictor("training/outputs/activity_classifier.joblib")
result = predictor.predict_window(np.zeros((50, 3), dtype=np.float32))
```

Input is 50 consecutive samples at 10 Hz in `acc_x, acc_y, acc_z` order and in units of g. The inference adapter and current artifacts are Python-specific. A browser deployment should export/convert the selected model and verify numerical parity before integrating it into the game.

## Kaggle GPU run

`kernel-metadata.json` attaches the public source dataset and requests a T4 GPU. The Kaggle script records whether CUDA is active; its PyTorch MLP branch uses CUDA when available. On successful completion, use `kaggle kernels output <owner>/<slug> -p training/kaggle-output` to retrieve `evaluation.json` and model weights.

Kaggle official docs: [API authentication and CLI](https://www.kaggle.com/docs/api), [Kaggle CLI kernel workflow](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), and [kernel metadata fields](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).
