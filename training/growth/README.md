# Nelore growth benchmark

Small, reproducible regression experiment for the UFMS Nelore supplementation dataset. This predicts the published average daily gain (ADG, kg/day) for a weighing interval. It does **not** identify a causal supplement response and is not yet a validated physiology model.

## Dataset and provenance

Canonical record: [Mendeley Data v7, DOI 10.17632/3wdjvxmvsp.7](https://data.mendeley.com/datasets/3wdjvxmvsp/7). The API reports CC BY 4.0. Version 7's `Complete_Dataset.csv` is pinned by SHA-256 in the training script. It contains 26 Nelore animals in one 16 ha paddock at UFMS, with 15 weighing periods from December 2022 to October 2023. The published description reports weight/supplementation, feeder visits, pasture spectral indices and weather. See the original data paper [in Data in Brief / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11144722/).

Kaggle is preferred for the training runtime. Put a verified copy of `Complete_Dataset.csv` in a Kaggle Dataset and attach it to the notebook/kernel, or pass `--kaggle-dataset owner/slug` after verifying its source hash and license. No Kaggle mirror is assumed by this repository. When there is no attached Kaggle source, use the pinned canonical Mendeley artifact explicitly for local development. The pipeline will not silently substitute a similarly named dataset.

## Run

```bash
python -m pip install -r training/growth/requirements.txt
python training/growth/train_growth.py --source mendeley
```

For Kaggle, attach the verified data and run:

```bash
python training/growth/train_growth.py --source kaggle
```

Or with an explicitly verified Kaggle slug:

```bash
python training/growth/train_growth.py --source kaggle --kaggle-dataset owner/verified-nelore-dataset
```

An explicit `--data /path/to/Complete_Dataset.csv` accepts a Kaggle-mounted path or a local file. The run saves the manifest, split IDs, metrics and fitted model under `training/growth/runs/nelore-v7/` (override with `--output-dir`). The script requires no GPU. With 390 animal-period records and 15 independent time contexts, a GPU neural model would add complexity without resolving the data's effective sample-size or generalization limits; Kaggle T4 availability is useful for later sequential models and larger datasets, not a quality criterion for this baseline.

## Experiment design

- Target: published interval ADG in kg/day.
- Features: starting body weight, interval length, supplementation average, day-of-year, observed interval weather and satellite pasture indices.
- Excluded: final weight, ADG itself, supplement total and feeder attendance/time/days, which are derived or measured during the outcome interval.
- Baselines: training-mean regressor, Ridge, Extra Trees and HistGradientBoosting.
- Temporal split: first 60% of periods for fitting, next 13% for model selection and latest 27% as a future-period test. Metrics are listed by period because all animals share environmental/treatment context.
- Animal split: seeded, disjoint animals for a separate generalization check. This split is diagnostic and does not choose the exported model.
- Primary metrics: MAE/RMSE in kg/day, bias and R². Split manifests record all animals and periods.

The temporal score is an **interval-conditioned nowcast**: weather and satellite features summarize the entire interval and are not known when the interval begins. A prospective simulator forecast must instead accept weather forecasts/planned management or use only covariates known at the start. ADS should be checked against the original experimental protocol before interpreting it as a manipulable dose.

## First completed run

The pinned dataset ran locally and on private Kaggle kernel `pascalcoimbra/campo-vivo-nelore-growth-baseline-v1` (version 2 complete; Kaggle T4 requested, input copy `pascalcoimbra/campo-vivo-nelore-ufms-v7`). Version 2 checks the CSV SHA-256 before training. Its metrics match the local run to floating-point precision. The run receipt with output hashes, `evaluation.json`, serialized model and raw Kaggle log are under [`runs/kaggle-v2/`](runs/kaggle-v2/); a first successful execution is also retained in `runs/kaggle-v1/`.

| Split/model | MAE (kg/day) | RMSE (kg/day) | R² |
|---|---:|---:|---:|
| Temporal validation, selected HistGradientBoosting | 0.277 | 0.400 | 0.041 |
| Future periods 12–15, HistGradientBoosting selected on validation | 0.455 | 0.569 | −0.181 |
| Future periods 12–15, Extra Trees comparison | 0.406 | 0.507 | 0.060 |
| Future periods 12–15, training-mean baseline | 0.511 | 0.642 | −0.505 |
| Unseen-animal test, HistGradientBoosting | 0.302 | 0.380 | 0.336 |
| Unseen-animal test, training-mean baseline | 0.363 | 0.467 | −0.000 |

The future-period test is worse than the validation score and has negative R² for the selected model. Extra Trees happens to do better on this future block, despite being worse on validation. That is evidence of unstable selection with only two validation periods, not evidence for picking the model that scored best on test. The separate animal holdout is not a farm-level validation and shares the same 15 period contexts. Dataset audit found the published ADG is almost exactly `(FINAL_WEIGHT - START_WEIGHT) / AMOUNT_DAYS` (correlation 0.998); final weight is excluded from features, but the target remains a derived interval outcome. The model is only a predictive benchmark and is not ready to drive biology or management interventions.

The Kaggle runtime requested a T4, but this baseline consists of scikit-learn tabular estimators that run on CPU; **no estimator used the GPU**. That is the appropriate compute choice for 390 records. The run verifies Kaggle packaging, input attachment, output/checkpoint recovery and numerical reproducibility; a larger sequential model should use the GPU only when its data scale and validated splits justify it. Local and Kaggle metrics match within `1e-12`; after reloading both serialized checkpoints, predictions on held-out future periods match exactly (maximum absolute difference 0.0), despite different artifact hashes across Python 3.12 and Python 3.13 runtimes.

## Limits and next validation

Rows repeat the same weather, pasture and supplementation context for all cattle in a period; thus environmental effects have roughly 15 period-level units, not 390 independent observations. There is one herd/paddock, no external farm, and no randomized intervention identifiable from this table. Use the predictions as a benchmark adapter, not as established BR-CORTE or NASEM physiology. The next useful data would be an independent herd or paddock with measured body weight, forage mass/composition, actual supplement intake and prospective weather; validate by holding out both farm and time. A mechanistic growth/nutrient balance must remain separately versioned and calibrated.

The private Kaggle kernel and private verified input mirror were used for this run. Credentials and remote access tokens are not included in this repository. The local command below can reproduce the same split and metrics from the canonical public Mendeley v7 file.

## Prospective prediction experiment

`train_growth_prospective.py` separates prediction from the interval-conditioned nowcast above. Each fold trains on earlier complete periods and predicts the next whole period. Features are restricted to starting body weight, the scheduled interval duration, calendar season, and the immediately previous measured ADG, which is admitted only when that prior period ends exactly at the current start date. Interval weather, vegetation indices, realized supplementation, feeder visits, final weight, and the target itself are excluded from the current feature vector.

The fixed Ridge candidate is compared to two predeclared baselines: training-period mean and previous-period ADG persistence. It reports per-period MAE and a paired bootstrap over period clusters; the latter is exploratory because all periods belong to one herd and paddock. It is not an independent-farm uncertainty interval and cannot support causal claims. The fit uses the 390-row UFMS v7 data but only nine later period contexts are tested; if its interval against the strongest baseline crosses zero or improvements are below 10%, it fails the proposed acceptance gate.

The first complete prospective Kaggle run (private kernel version 2, CPU) reproduced the local results to a maximum absolute metric difference of `1.2e-15`. Ridge reached MAE `0.431 kg/d`, versus `0.462 kg/d` for the stronger pooled baseline (training mean) and `0.561 kg/d` for previous-ADG persistence. The relative improvement over the strongest baseline is only 6.7%; the paired period-bootstrap 95% CI for MAE difference is `[-0.141, 0.089] kg/d`, crossing zero. It therefore **fails** the proposed acceptance gate and must not be integrated as an accepted growth model. Its future-period R² is `−0.123`. The exact output, checkpoint, log, source hashes and failed first attempt are preserved under [`runs/kaggle-prospective-v2/`](runs/kaggle-prospective-v2/).

Local reproduction:

```bash
python training/growth/train_growth_prospective.py \
  --data .cache/nelore-growth/Complete_Dataset.csv \
  --output-dir training/growth/runs/prospective-adg-v1
```

Kaggle execution uses a private, SHA-verified mirror and disables GPU because the fixed tabular regressors are tiny CPU workloads:

```bash
mkdir -p /tmp/campo-vivo-growth-prospective
cp training/growth/train_growth_prospective.py /tmp/campo-vivo-growth-prospective/
cp training/growth/kaggle-prospective/kernel-metadata.json /tmp/campo-vivo-growth-prospective/kernel-metadata.json
kaggle kernels push -p /tmp/campo-vivo-growth-prospective
```
