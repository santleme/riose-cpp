# Nellore trough next-visit interval benchmark

## Preregistered question and boundary

Can event-local context and a short history predict the **next non-overlapping time-to-return** to an instrumented trough better than persistence and a time-of-day median, on this one cohort? This is only supervised prediction of a sensor-event interval. It does not identify eating, water intake, motivation, social hierarchy, welfare, a biological need, a causal management response, or an animal action policy.

### Dataset and license

Source: [Kaggle v2](https://www.kaggle.com/datasets/juliavalente/data-from-visits-to-the-trough-of-nellore-cattle), license `CC-BY-NC-SA-4.0`, paper DOI [10.1371/journal.pone.0271337](https://doi.org/10.1371/journal.pone.0271337). The public archive contains only `animal_id`, `equip_id`, `IN`, and `OUT`. It has no intake, behavior ethogram, GPS, water volume, or treatment variable. The model uses no animal/equipment ID. The dataset is attached directly from Kaggle to a **private** Kaggle kernel; the dataset is not re-uploaded. Raw data is never written into this repository.

Kaggle v2 file hashes are pinned in [`train_gap_model.py`](train_gap_model.py). In the inspected release, the compressed archive SHA-256 is `fd2cfd016503dcff40be2931a6bd04a7185d020c811f6cafa08ec3925d9b5884`, and the extracted CSV SHA-256 is `37fbe08c882723b15fe028bcb5810fff52a41b3334ef73593814e0674e8f1408`. The Kaggle license metadata is `CC-BY-NC-SA-4.0`; attribution and non-commercial/share-alike conditions must remain attached to any data-derived artifact. Do not use the checkpoint commercially without license review.

### Fixed target and event validity

Each sample is the current completed event plus past-only history. The target is `next_event.IN - current_event.OUT`, measured in hours. Negative intervals indicate overlapping same-animal records and are excluded, never clipped. Zero gaps remain valid. The last event per animal has no observed successor and is right-censored, so it has no regression target. The first event has no prior interval; it remains eligible with an explicit missing-history flag. Long positive gaps remain uncapped and are reported separately above 8h, 12h and 24h. Duration is `OUT-IN`; negative durations fail ingestion. Timestamps have no encoded timezone and are preserved as local naive datetimes.

The feature list is fixed in the script: log duration, cyclical event-end hour and weekday, previous positive gap, missing flags, median of the prior three valid gaps, and prior-event counts in six and 24 hours. No feature reads the next event, event ID, animal ID, equipment ID, absolute date, or any outcome from after the current event.

### Frozen splits and selection

The script uses one registered seed (`20261022`) and creates two independent evaluation views:

1. **Animal holdout:** cattle IDs are shuffled deterministically, then assigned to disjoint 60/20/20 train/validation/test groups. All dates for a test animal remain outside fitting and model selection. Exact IDs and per-animal outputs remain attached only to the private Kaggle run; the public manifest records fold sizes and aggregate metrics only.
2. **Temporal holdout:** event dates are ordered and split chronologically 60/20/20. A sample is purged when its next event falls in a different date partition, preventing a target from crossing train/validation/test boundaries. The animals are shared across chronological partitions; this is a within-cohort future-time check, not unseen-herd validation.

Baselines are the training-set global median, training-set hour-of-day median with global fallback, and previous-gap persistence. The baseline is selected by validation macro-animal MAE. A compact MLP (11→32→16→1) learns `log1p(gap_hours)` with Huber loss and early stopping; its checkpoint/epoch is selected only by validation log-gap MAE. The test partitions never select features, epochs, hyperparameters, or candidates. Metrics include pooled and per-animal MAE in hours and log1p-hours. A paired 2,000-draw bootstrap over animals gives a 95% CI for macro-animal MAE difference against the validation-selected baseline.

The fixed acceptance gate requires the MLP to beat the selected baseline on pooled test MAE and have the animal-bootstrap 95% CI upper bound below zero on **both** animal and time tests. This gate evaluates a predictive candidate only; it never turns this regressor into an animal policy. If the gate fails, report that result and retain any checkpoint solely as a private experiment artifact.

### Results and second candidate

The private Kaggle kernel `pascalcoimbra/campo-vivo-nellore-next-visit-gap-benchmark`, version 1, completed with `device=cuda`, GPU `Tesla T4`, PyTorch `2.11.0+cu128`, and nonzero optimizer steps (260 animal split; 676 temporal split). MLP pooled MAE was 0.1880 h against the training hourly median at 0.1635 h on held-out animals, and 0.1919 h against 0.1722 h on the future dates. The paired macro-animal MAE deltas (model minus baseline) were positive, with 95% animal-bootstrap intervals [0.0205, 0.0301] h and [0.0172, 0.0209] h. The temporal view's status is **exploratory** because its outcomes were inspected during an earlier ExtraTrees iteration whose tuning validation included all dates. Do not treat those dates as a still-blinded independent holdout. This cohort also cannot establish external-farm generalization.

An ExtraTrees CPU iteration compared four fixed parameter candidates; the single winner was chosen using validation macro-animal MAE, then the animal holdout and temporal dates were evaluated. On held-out animals the tree reached 0.2221 h pooled MAE against 0.1635 h for the hourly median and 0.3063 h for persistence; the macro-animal delta against hourly median was +0.0577 h (95% animal-bootstrap CI [+0.0511, +0.0643]). On the already-inspected temporal dates it reached 0.2323 h against 0.1722 h hourly and 0.3335 h persistence; this time result is exploratory and not an independent holdout. Neither candidate is accepted or integrated as an animal policy.

For this public repository, retain only code, protocol, aggregate metrics, limitations, and hashes. Do not commit the CSV, full per-animal output, model checkpoint, or joblib artifact. The source and derived model license implications are CC BY-NC-SA-4.0 and have not been cleared for redistribution or commercial use; checkpoints remain recoverable only through the private Kaggle kernel.

### Kaggle GPU requirement

`kernel-metadata.json` attaches the original public Kaggle dataset to a private kernel and requests a GPU. `train_gap_model.py` selects CUDA automatically, logs `cuda_available`, device name, CUDA/PyTorch versions and fails rather than silently running on CPU when Kaggle data is mounted but CUDA is unavailable. The run output must prove the model update occurred on `cuda`; a GPU request in metadata alone is not sufficient evidence.

For local reproduction, install `requirements.txt` and run:

```bash
python training/visits/train_gap_model.py --data /path/to/visits_to_the_trough.csv --device cpu
```

On Kaggle, the mounted file is found automatically. Use an output directory outside the repository for local runs. Raw CSV/archive, per-animal evaluation output, and checkpoints must remain outside source control. For Kaggle, outputs remain attached to the private kernel. Aggregate results and output hashes are recorded in `runs/experiment-manifest.json`.

## Acceptance and limits

The source is a single 21-day feed-efficiency trial. The associated paper reports 37 animals; Kaggle v2 has 38 distinct IDs, including one ID with one event. The CSV’s 90,211 rows are repeated events, not independent animals. In the source audit, there were 645 overlapping successive records, 11 exact zero gaps, 38 right-censored terminal visits, 17 zero-duration events and 3,471 events longer than five minutes. These are data-quality and sampling properties, not behavior labels. The main score is conditional on non-overlapping records; results cannot validate real-world feeding physiology or generalize beyond this cohort.
