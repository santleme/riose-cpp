# Bovino activity baseline

Reproducible supervised baseline for coarse activity recognition from 3-axis cow IMU signals. It is an observation classifier, **not** a model of motivation, health, welfare, or causal environmental response.

## Data

Default source: [Cow nose ring data set on Kaggle](https://www.kaggle.com/datasets/fandaoerji/cow-nose-ring-data-set). The Kaggle metadata declares **CC0-1.0**. The card describes seven free-moving cows, 10 Hz accelerometer data, 1,252,522 rows, and labels `0 feeding`, `1 rumination`, `2 standing`, `3 lying`, `4 walking`. The downloaded CSV audited on 2026-10-22 has 1,252,536 rows (14 above the card count) and whole-second timestamps. The associated 2024 paper reports eleven study cows and evaluates a three-class ontology (`feeding`, `rumination`, `other`), so the relationship between paper population, Kaggle subset, and five CSV codes remains unresolved. Pin and audit the actual CSV before training; do not treat the paper's metrics as directly comparable to the five-label benchmark. The raw dataset is not committed; fetching is handled by the official Kaggle CLI. [Source paper](https://www.mdpi.com/2076-2615/14/8/1187).

The pipeline pins the audited CSV by SHA-256 (`1baf4a…c8cb0f`) and rejects changed contents by default, including a changed file mounted by Kaggle. For an explicitly audited new version, pass `--allow-unpinned-data`; its hash is then recorded in `evaluation.json`. This is a guard against silent Kaggle dataset updates, not a substitute for checking provenance, licensing, schema, and label semantics.

The script parses and sorts timestamps per animal/day. Since the real CSV has whole-second time values repeated across rows (normally 20 rows, followed by a +2 s timestamp), these are treated as coarse buckets, not duplicate samples. A block is admitted only when its row count matches elapsed seconds × documented 10 Hz; incomplete, ambiguous, or gapped blocks break runs and are excluded. Row order within a coarse bucket is assumed to be the sensor sample order because the file has no sub-second timestamps; this assumption cannot be verified from the CSV. Dropped rows with missing sensor values reduce block counts and therefore cannot be silently bridged. The parser then extracts non-overlapping 5-second windows and discards windows where the most common frame label is below 80%. The frozen CSV produced 18,446 accepted windows after this preprocessing. The evaluation metadata records row drops, reorderings, timestamp blocks, ambiguous/gap counts, and accepted/discarded windows. It reserves cow 1 for model selection and cow 6 for the final animal-held-out evaluation. It compares Extra Trees, Random Forest, and an MLP; on a CUDA runtime it also fits a small PyTorch MLP on the GPU.

This is a small imbalanced dataset: walking, feeding, and rumination dominate, while standing and lying are rare. Even an animal-held-out score is only a pilot on one of seven animals. Use leave-one-animal-out evaluation and prospective field validation before making generalization or biological claims. The labels describe observed activity, not internal state or causal response to feed, water, pasture, weather, or social context.

## Kaggle run (2026-10-22)

Kernel version 2 completed with `NvidiaTeslaT4` requested and `cuda_available=true`. Those metrics were produced before timestamp-continuity validation was added; they are historical results and must not be attributed to the fixed preprocessing. Rerun the Kaggle experiment and regenerate `evaluation.json` before using any updated scores or model artifacts. The old scores are preserved for provenance: animal 1 selected the MLP; animal 6 was the test group. Test macro-F1 was 0.491 for the selected MLP (weighted-F1 0.814); Extra Trees scored 0.488, Random Forest 0.485, and the CUDA PyTorch MLP 0.510. See [the captured evaluation](outputs/evaluation.json), [Kaggle run metadata](experiments/kaggle-cuda-run.json), and [kernel log](experiments/kaggle-cuda-run.log). The private [Kaggle kernel](https://www.kaggle.com/code/pascalcoimbra/campo-vivo-cow-activity-baseline) is the remote execution record.

Kernel version 3 failed after about 510 seconds during preprocessing, before fitting any estimator. Its submitted parser interpreted repeated whole-second values as duplicate 100 ms timestamps and returned no valid windows. The current code instead audits whole-second timestamp blocks against row count and the documented 10 Hz rate; the fixed parser was run against the frozen CSV locally and extracted 18,446 windows. The CSV hash, row/label counts, timestamp diagnostics, and v3 failure log hash are recorded in [the data audit](experiments/cow-nose-ring-data-audit.json) and [run status](experiments/cow-ring-kaggle-v3-status.json). V3 produced no model metrics or checkpoints.

Kernel version 4 completed on a Tesla T4 using the corrected coarse-timestamp parser. It found 1,252,536 rows and extracted 18,446 accepted windows, matching the local audit. The v4 script did not embed a dataset SHA-256 in its evaluation output; the metrics are associated with the attached public dataset and matching parser diagnostics, while exact file-hash verification is deferred to the pinned v5 run. The existing animal 1/6 split and candidate family were already inspected, so these are exploratory repeat-run metrics, not independent validation or acceptance evidence. Validation selected Extra Trees (macro-F1 0.6931); its animal 6 macro-F1 was 0.4761 (weighted-F1 0.7913). Random Forest scored 0.4699 / 0.7773, scikit-learn MLP 0.4709 / 0.7802, and the CUDA PyTorch MLP 0.5435 / 0.7716. The CUDA model had only 0.3621 validation macro-F1, so the higher test score does not change the validation-selected model. Animal 6 has only 8 standing and 12 lying windows; Extra Trees scored zero F1 for both classes. See aggregate [v4 run manifest](experiments/cow-ring-kaggle-v4-manifest.json); weights, raw log, full evaluation JSON, confusion matrices, and per-animal outputs remain private in Kaggle.

V4 confirms CUDA and `Tesla T4` in its evaluation metadata. It predates explicit optimizer-step instrumentation: from the fixed 45-epoch, batch-1024 loop and training row counts, 630 optimizer steps are inferred for validation fitting and 630 for the refit, 1,260 total; this count was not directly logged. The pinned v5 script adds direct step and dataset-hash reporting. Kaggle execution must also enforce the `--require-cuda` guard before data loading; do not treat a GPU request alone as proof that CUDA was available.

Kernel version 5 completed on a Tesla T4 with the script's mounted-Kaggle CUDA guard active. It verified the frozen CSV SHA-256 (`1baf4a…c8cb0f`), reproduced the 18,446-window parser audit and the v4 metrics, and directly recorded 630 optimizer updates for the final CUDA fit. The temporary validation-fit model's 630 updates are derived from its sample count and fixed loop, not emitted as a separate counter. The selected Extra Trees checkpoint SHA-256 is identical to v4. V5 made no architecture, hyperparameter, or model-selection changes; because the same animal holdout and candidate families had already been inspected, the rerun is reproducibility evidence only, not independent validation or acceptance evidence. See the aggregate [v5 run manifest](experiments/cow-ring-kaggle-v5-manifest.json); weights, full evaluation JSON, and logs remain private in Kaggle.

The next exploratory protocol is leave-one-animal-out on the same seven-animal cohort. `train_loso.py` evaluates the v5 Extra Trees and CUDA MLP configurations unchanged in all seven outer folds; it does not tune or select a model from a fold. It verifies the same CSV SHA and requires CUDA when running with the Kaggle input mount. Per-animal and per-class metrics, fold class support, and labels absent from each fold's training data are recorded. This is still post-selection because the model families and original grouped holdouts have already been inspected; it is not an independent acceptance estimate. The private run preserves v5 artifacts and uses metadata in [cow-ring-kaggle-loso-v6-metadata.json](experiments/cow-ring-kaggle-loso-v6-metadata.json).

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
